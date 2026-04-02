"""
rag_core.py
-----------
Self-contained RAG module: retrieval + memory + chain assembly.
Designed to be imported by any agent in the multi-agent system.

Usage:
    from src.rag.rag_core import RAGCore, MetadataFilter

    rag = RAGCore(session_id="user_42")
    result = rag.run(
        query="What is the return policy?",
        filter=MetadataFilter(parent_section="Returns & Refunds")
    )
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory


# ---------------------------------------------------------------------------
# Config — must match what was used during ingestion
# ---------------------------------------------------------------------------

CHROMA_DIR        = "data/chroma_db"
COLLECTION_NAME   = "recursive_espazo_nature"
EMBED_MODEL       = "llama3.1"          # must match the model used during ingestion
LLM_MODEL         = "llama3.1"
TOP_K             = 6
SCORE_THRESHOLD   = 0.35
MEMORY_MAX_TOKENS = 1500   # token threshold before buffer → summary


# ---------------------------------------------------------------------------
# MetadataFilter
# ---------------------------------------------------------------------------

@dataclass
class MetadataFilter:
    section:        Optional[str] = None
    parent_section: Optional[str] = None
    page_gte:       Optional[int] = None
    page_lte:       Optional[int] = None
    keyword:        Optional[str] = None

    def to_chroma_where(self) -> Optional[dict]:
        clauses = []
        if self.section:
            clauses.append({"section":        {"$eq": self.section}})
        if self.parent_section:
            clauses.append({"parent_section": {"$eq": self.parent_section}})
        if self.page_gte is not None:
            clauses.append({"page":           {"$gte": self.page_gte}})
        if self.page_lte is not None:
            clauses.append({"page":           {"$lte": self.page_lte}})
        if self.keyword:
            clauses.append({"keywords":       {"$contains": self.keyword}})
        if not clauses:  return None
        if len(clauses) == 1: return clauses[0]
        return {"$and": clauses}


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    content:        str
    score:          float
    section:        str
    parent_section: str
    keywords:       list[str]
    page:           int
    index_start:    int

    @classmethod
    def from_doc(cls, doc: Document, score: float) -> "RetrievedChunk":
        m = doc.metadata
        return cls(
            content        = doc.page_content,
            score          = round(score, 4),
            section        = m.get("section", ""),
            parent_section = m.get("parent_section", ""),
            keywords       = [k.strip() for k in m.get("keywords", "").split(",") if k.strip()],
            page           = int(m.get("page", 0)),
            index_start    = int(m.get("index_start", 0)),
        )

    def as_context_block(self) -> str:
        return (
            f"[Source: {self.parent_section} › {self.section} "
            f"| Page {self.page} | Score {self.score}]\n"
            f"{self.content}"
        )

    def as_source_ref(self) -> dict:
        return {
            "section":        self.section,
            "parent_section": self.parent_section,
            "page":           self.page,
            "score":          self.score,
        }


# ---------------------------------------------------------------------------
# RAGCore — the full pipeline in one class
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful customer service agent for Espazo Nature.
Espazo Nature is a company that provides glamping services in Galicia, Spain.
Answer using ONLY the context below. If the answer isn't there, say:
"I don't have that information — please contact support."
Be concise, friendly, and cite the section name when relevant.
If the question is in Spanish, answer in Spanish. If in English, answer in English.

{memory_context}

Context:
{context}"""


class RAGCore:
    """
    Full RAG pipeline: retrieval + memory + generation.

    One instance per session (carries memory state).
    Thread-safe for a single session; instantiate separately per user.

    Parameters
    ----------
    session_id       : unique ID for this conversation (for logging/memory keying)
    chroma_dir       : path to persisted Chroma store
    collection_name  : Chroma collection to query
    embed_model      : must match the model used during ingestion
    llm_model        : Ollama model for generation + summarisation
    top_k            : number of chunks to retrieve
    score_threshold  : minimum cosine similarity to keep a chunk
    memory_max_tokens: token budget before ConversationSummaryBufferMemory
                       starts compressing old turns into a summary
    """

    def __init__(
        self,
        session_id:        str   = "default",
        chroma_dir:        str   = CHROMA_DIR,
        collection_name:   str   = COLLECTION_NAME,
        embed_model:       str   = EMBED_MODEL,
        llm_model:         str   = LLM_MODEL,
        top_k:             int   = TOP_K,
        score_threshold:   float = SCORE_THRESHOLD,
        memory_max_tokens: int   = MEMORY_MAX_TOKENS,
    ):
        self.session_id      = session_id
        self.top_k           = top_k
        self.score_threshold = score_threshold

        # LLM (shared for generation + summarisation)
        self.llm = ChatOllama(model=llm_model, temperature=0.1)

        # Embeddings + Vector store
        self.embeddings   = OllamaEmbeddings(model=embed_model)
        self.vectorstore  = Chroma(
            collection_name    = collection_name,
            embedding_function = self.embeddings,
            persist_directory  = chroma_dir,
        )

        # Memory: sliding buffer → auto-summary once max_token_limit hit
        self.memory = ConversationSummaryBufferMemory(
            llm                 = self.llm,
            max_token_limit     = memory_max_tokens,
            return_messages     = True,
            memory_key          = "chat_history",
            output_key          = "answer",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query:  str,
        filter: Optional[MetadataFilter] = None,
    ) -> dict:
        """
        Run one RAG turn.

        Returns
        -------
        {
            "answer":  str,
            "sources": list[dict],   # section, parent_section, page, score
            "chunks":  list[RetrievedChunk],
        }
        """
        # 1. Retrieve
        chunks  = self._retrieve(query, filter)
        context = self._build_context(chunks)

        # 2. Load memory
        mem_vars       = self.memory.load_memory_variables({})
        memory_context = self._format_memory(mem_vars.get("chat_history", []))

        # 3. Build prompt + run LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}"),
        ])

        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context":        context,
            "memory_context": memory_context,
            "chat_history":   mem_vars.get("chat_history", []),
            "query":          query,
        })

        # 4. Save turn to memory (triggers summarisation if over token limit)
        self.memory.save_context(
            {"input":  query},
            {"answer": answer},
        )

        return {
            "answer":  answer,
            "sources": [c.as_source_ref() for c in chunks],
            "chunks":  chunks,
        }

    def get_history_summary(self) -> str:
        """Return the current memory summary (for handoff to other agents)."""
        mem_vars = self.memory.load_memory_variables({})
        return self._format_memory(mem_vars.get("chat_history", []))

    def clear_memory(self):
        """Reset memory for a new session."""
        self.memory.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query:  str,
        filter: Optional[MetadataFilter],
    ) -> list[RetrievedChunk]:
        where   = filter.to_chroma_where() if filter else None
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=self.top_k, filter=where,
        )
        chunks = [
            RetrievedChunk.from_doc(doc, score)
            for doc, score in results
            if score >= self.score_threshold
        ]
        return self._keyword_rerank(query, chunks)

    def _keyword_rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        q = query.lower()
        return sorted(
            chunks,
            key=lambda c: c.score + sum(0.05 for kw in c.keywords if kw.lower() in q),
            reverse=True,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant documents found."
        return "\n\n---\n\n".join(c.as_context_block() for c in chunks)

    def _format_memory(self, messages: list) -> str:
        if not messages:
            return ""
        lines = ["[Conversation so far]"]
        for m in messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)
