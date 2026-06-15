"""
RAG Agent — LangGraph-based conversational retrieval agent.

Architecture
------------
    START → manage_memory → call_model
                               ↓  (tool_calls present?)
                            [yes] → retrieve [→ translate_context*] → call_model  (loop)
                            [no]  → END

    Greetings are handled natively by the LLM via system-prompt
    instructions — no separate classifier or pre-node is needed.

    * translate_context is an optional node activated via
      TRANSLATE_CONTEXT_ENABLED in config.py.  When enabled it translates
      retrieved Spanish documents into the user's detected language before
      the LLM generates the final answer.

Public API
----------
    generate_reply(platform, user_message, sender_id) -> str
    retrieve_documents(query) -> tuple[str, list[Document]]   (also a LangChain @tool)
    rag_agent  — compiled LangGraph graph (for streaming / direct use)
"""

from __future__ import annotations

import logging
import warnings
import os

from typing import Annotated
from langchain_core._api import deprecated

from langdetect import DetectorFactory, detect as _detect_lang, LangDetectException
DetectorFactory.seed = 0

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from src.telemetry import timed
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.infrastructure.memory.sqlite_memory import SqliteMemoryAdapter
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.retrieval.hybrid_retriever import _reciprocal_rank_fusion
from src.config import (
    COLLECTION,
    DB_PATH,
    HYBRID_K,
    MODEL_NAME,
    RRF_K,
    TOP_K,
    FILTER_CONFIDENCE_THRESHOLD,
    RELEVANCE_THRESHOLD,
    TRANSLATE_CONTEXT_ENABLED,
    MAX_HISTORY_TURNS,
    CONVERSATION_SUMMARIZATION_ENABLED,
    SYSTEM_PROMPT_PATH,
    QUERY_METADATA_PROMPT_PATH
)
from src.infrastructure.retrieval.reranker import rerank
from src.rag.metadata_extractor import find_valid_labels
from src.utils import get_sections, load_prompt
from src.infrastructure.embeddings import get_embeddings
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared infrastructure (initialised once at import time)
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", message="Relevance scores must be between")

if os.getenv("AGENT_LLM_PROVIDER", "ollama") == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    _llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
else:
    _llm = ChatOllama(model=MODEL_NAME)

embeddings = get_embeddings()

_vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

# Build BM25 index from the same documents that are in Chroma
_bm25_index = BM25Retriever(_vectorstore)
_chroma_snapshot = _vectorstore.get()


def rebuild_bm25() -> None:
    """
    Rebuild the module-level BM25 index from the current ChromaDB state.
    Called by the file watcher (FR-ING-05) after a successful re-ingestion
    so that the sparse index stays in sync with the updated vector store.
    """
    global _vectorstore, _bm25_index, _chroma_snapshot
    logger.info("Rebuilding BM25 index and reloading vectorstore after re-ingestion...")
    
    from src.config import get_active_collection_name
    active_col = get_active_collection_name()
    logger.info("Updating vectorstore to use active collection: %s", active_col)
    
    _vectorstore = Chroma(
        collection_name=active_col,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )
    _chroma_snapshot = _vectorstore.get()
    _bm25_index = BM25Retriever(_vectorstore)
    logger.info("Vectorstore and BM25 index rebuild complete.")

# ---------------------------------------------------------------------------
# Metadata extraction (query → section + keywords)
# ---------------------------------------------------------------------------


_QUERY_METADATA_PROMPT = load_prompt(QUERY_METADATA_PROMPT_PATH)


class QueryMetadata(BaseModel):
    """Structured output used to filter the vector store before retrieval."""

    finding: str = Field(
        description="The specific section mentioned in the query, or 'none' if none."
    )
    keywords: list[str] = Field(
        description=(
            "Proper nouns or keywords extracted from the query in their ORIGINAL "
            "language. DO NOT translate them."
        )
    )


_metadata_extractor = _llm.with_structured_output(QueryMetadata)


def _build_section_filter(query: str) -> dict | None:
    """
    Run the metadata extractor to derive a Chroma $where filter from the query.

    Returns None if no relevant section is found or on any error.
    """
    sections = ",".join(str(s) for s in get_sections(_vectorstore))
    system_prompt = _QUERY_METADATA_PROMPT.format(sections=sections)
    try:
        meta: QueryMetadata = _metadata_extractor.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ])
    except Exception:
        logger.exception("Metadata extraction failed for query: %r", query)
        return None

    if not meta.finding or meta.finding.lower() in ("none", ""):
        return None
    #logger.debug("metadata: %s, %s", meta.finding, meta.keywords)
    
    canonical_finding, field_type, match_score = find_valid_labels(
        finding=meta.finding,
        chroma_snapshot=_chroma_snapshot,
        logger=logger,
    )
    
    if not canonical_finding or match_score < FILTER_CONFIDENCE_THRESHOLD:
        logger.info(
            "Skipping metadata filter: best match '%s' for extracted '%s' scored %.2f (threshold: %.2f)",
            canonical_finding, meta.finding, match_score, FILTER_CONFIDENCE_THRESHOLD
        )
        return None

    logger.info("Applying metadata filter: '%s' (score: %.2f)", canonical_finding, match_score)

    # If the label appears as both a section and subsection, use an $or filter
    if field_type == "both":
        return {
            "section": {"$eq": canonical_finding},
        }
    
    # Otherwise, filter strictly by the specific field it belongs to
    return {field_type: {"$eq": canonical_finding}}


def _apply_metadata_filter(
    hits: list[tuple[Document, float]],
    chroma_filter: dict | None,
) -> list[tuple[Document, float]]:
    """
    Apply a Chroma-style ``$where`` filter to BM25 hits so both indexes
    respect the same metadata scope.

    Supports: ``{"$and": [...]}``, ``{"$or": [...]}``, and simple leaf
    clauses like ``{"section": {"$eq": value}}``.
    """
    if chroma_filter is None:
        return hits

    def _leaf_passes(doc: Document, clause: dict) -> bool:
        for field, op in clause.items():
            if not isinstance(op, dict):
                continue
            op_name, val = next(iter(op.items()))
            actual = doc.metadata.get(field, "")
            if op_name == "$eq" and actual != val:
                return False
            if op_name == "$contains" and val not in actual:
                return False
        return True

    def _passes(doc: Document, clause: dict) -> bool:
        if "$and" in clause:
            return all(_passes(doc, c) for c in clause["$and"])
        if "$or" in clause:
            return any(_passes(doc, c) for c in clause["$or"])
        return _leaf_passes(doc, clause)

    return [(doc, score) for doc, score in hits if _passes(doc, chroma_filter)]


def _translate_query_to_es(query: str, source_lang: str) -> str:
    """
    Translates a non-Spanish user query (EN, FR, DE, etc.) into Spanish for retrieval.
    """
    if source_lang == "es":
        return query
    
    translation_prompt = (
        "Translate the following user search query into Spanish. "
        "Preserve all specific entities, numbers, and proper nouns. "
        "Return ONLY the translated Spanish query, nothing else.\n\n"
        f"Query: {query}"
    )
    try:
        translated = _llm.invoke([{"role": "user", "content": translation_prompt}])
        translated_text = translated.content.strip()
        logger.info("Translated query for retrieval (%s -> es): '%s' -> '%s'", source_lang, query, translated_text)
        return translated_text
    except Exception:
        logger.exception("Query translation failed; falling back to original query.")
        return query


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
@timed("agent.retrieve_documents")
def retrieve_documents(query: str, config: RunnableConfig) -> tuple[str, list[Document]]:
    """
    Retrieve the most relevant documents for a query using a hybrid
    BM25 + dense-vector RRF pipeline followed by cross-encoder reranking.

    Returns both a serialised string (for the LLM context) and the raw
    Document objects (artifact, for downstream use).
    """
    # 0. Detect and translate query if non-Spanish
    try:
        lang_code = _detect_lang(query) if len(query.strip()) >= 3 else "es"
    except Exception:
        lang_code = "es"

    retrieval_query = query
    if lang_code != "es":
        retrieval_query = _translate_query_to_es(query, lang_code)

    # 1. Derive metadata filter + augment query with extracted keywords
    #search_filter = _build_section_filter(retrieval_query)
    configurable = config.get("configurable", {})
    section = configurable.get("section")
    #logger.debug("Search filter: %s", search_filter)
    
    # Re-extract keywords to append to the query (improves both indexes)
    """sections = ",".join(str(s) for s in get_sections(_vectorstore))
    system_prompt = _QUERY_METADATA_PROMPT.format(sections=sections)
    try:
        meta: QueryMetadata = _metadata_extractor.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": retrieval_query}
        ])
        #if meta.keywords:
            #retrieval_query = retrieval_query + " " + " ".join(meta.keywords)
    except Exception:
        logger.warning("Keyword augmentation failed; using original query.")"""

    # 2. Hybrid retrieval: dense (Chroma) + sparse (BM25) → RRF
    #    Dense uses a "soft boost": fetch with AND without the filter,
    #    then deduplicate keeping the best score. Filtered matches rank
    #    higher naturally, but unfiltered relevant docs aren't excluded.
    try:
        # --- Dense: unfiltered (broad recall) ---
        unfiltered = _vectorstore.similarity_search_with_relevance_scores(
            query=retrieval_query, k=HYBRID_K,
        )

        # --- Dense: filtered (section boost) ---
        search_filter = None
        if section:
            search_filter = {"section": {"$eq": section}}
        if search_filter:
            filtered = _vectorstore.similarity_search_with_relevance_scores(
                query=retrieval_query, k=HYBRID_K, filter=search_filter,
            )
        else:
            filtered = []

        # Merge: deduplicate by page_content, keep the higher score
        seen: dict[str, tuple[Document, float]] = {}
        for doc, score in list(filtered) + list(unfiltered):
            if score < 0.0:
                continue
            key = doc.page_content
            if key not in seen or score > seen[key][1]:
                seen[key] = (doc, score)
        dense_hits = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:HYBRID_K]

        # --- Sparse (BM25) — no hard filter either ---
        sparse_hits = _bm25_index.search(retrieval_query, k=HYBRID_K)

        rrf_docs = _reciprocal_rank_fusion(dense_hits, sparse_hits, k=RRF_K, top_n=HYBRID_K)
        docs, best_score = rerank(retrieval_query, rrf_docs, top_n=TOP_K)

    except Exception:
        logger.exception("Retrieval failed; falling back to simple retrieve.")
        docs = _vectorstore.as_retriever().invoke(retrieval_query)
        best_score = float("-inf")

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
    )
    return serialized, {"docs": docs, "best_reranker_score": best_score}


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Typed state for the RAG agent graph."""
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

_LANG_NAMES = {"es": "Spanish", "en": "English", "gl": "Galician", "fr": "French", "de": "German"}


def _detect_user_language(messages: list) -> str:
    """
    Walk backwards through the message list and return the ISO-639-1
    language code of the most recent user (HumanMessage) message.

    Falls back to ``"es"`` (Spanish) when detection fails or the
    message is too short for reliable classification.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = msg.content.strip()
            if len(text) < 3:          # too short for reliable detection
                return "es"
            try:
                return _detect_lang(text)
            except LangDetectException:
                return "es"
    return "es"





# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = load_prompt(SYSTEM_PROMPT_PATH)

_llm_with_tools = _llm.bind_tools([retrieve_documents])


class ModelCaller:
    """
    Graph node responsible for calling the LLM.
    Decoupled from static prompt constants to support prompt injection
    from the Composition Root (for A/B testing / evaluation).
    """

    def __init__(self, system_prompt_template: str) -> None:
        self.system_prompt_template = system_prompt_template

    def __call__(self, state: AgentState, config: RunnableConfig = None) -> dict:
        """Invoke the LLM, injecting the system prompt template with detected language."""
        lang_code = _detect_user_language(state["messages"])
        lang_name = _LANG_NAMES.get(lang_code, lang_code.title())
        logger.info("Detected user language: %s (%s)", lang_code, lang_name)

        # FR-AGT-04: Out-of-scope fallback — trigger only when the reranker
        # explicitly scored every candidate below the relevance threshold,
        # NOT merely when the retrieval returned no documents.  This lets
        # the LLM handle greetings / small-talk from its own knowledge even
        # if the knowledge base had nothing to offer.
        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            tool_msg = state["messages"][-1]
            artifact = getattr(tool_msg, "artifact", None) or {}
            best_score = artifact.get("best_reranker_score", None)

            if (
                best_score is not None
                and best_score != float("-inf")   # no candidates ≠ out-of-scope
                and best_score < RELEVANCE_THRESHOLD
            ):
                from src.domain.orchestrator import _OOS_RESPONSES
                from langchain_core.messages import AIMessage

                logger.info(
                    "OOS fallback: best reranker score %.4f < threshold %.4f",
                    best_score, RELEVANCE_THRESHOLD,
                )
                reply_text = _OOS_RESPONSES.get(lang_code, _OOS_RESPONSES["es"])
                return {"messages": [AIMessage(content=reply_text)]}

        # Resolve system prompt template (Composition Root default or runtime config override)
        configurable = config.get("configurable", {}) if config else {}
        prompt_tpl = configurable.get("system_prompt_template", self.system_prompt_template)

        prompt = prompt_tpl.format(response_language=lang_name)
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = _llm_with_tools.invoke(messages)
        return {"messages": [response]}


# ---------------------------------------------------------------------------
# Memory management (FR-MEM-03)
# ---------------------------------------------------------------------------

def _manage_memory(state: AgentState) -> dict:
    """
    Active path memory node.
    Performs quick synchronous safety hard truncation ONLY if message history
    exceeds the safety limit to prevent token context overflow.
    Otherwise, does nothing, leaving summarization to the background worker.
    """
    messages = state.get("messages", [])

    # Extract existing summary and conversation messages
    existing_summary_msg = None
    convo_msgs = []
    for m in messages:
        if m.id == "conversation_summary" or (m.type == "system" and m.content.startswith("Resumen de la conversación anterior:")):
            existing_summary_msg = m
        else:
            convo_msgs.append(m)

    # Safety limit:
    # If summarization is enabled, safety limit is MAX_HISTORY_TURNS * 4.
    # If disabled, safety limit is MAX_HISTORY_TURNS * 2.
    if CONVERSATION_SUMMARIZATION_ENABLED:
        safety_limit = MAX_HISTORY_TURNS * 4
    else:
        safety_limit = MAX_HISTORY_TURNS * 2

    if len(convo_msgs) <= safety_limit:
        return {"messages": []}

    # Hard truncate down to MAX_HISTORY_TURNS * 2
    logger.warning("Active path: history size (%d) exceeds safety limit (%d). Hard-truncating.", 
                   len(convo_msgs), safety_limit)
    
    cut_idx = len(convo_msgs) - (MAX_HISTORY_TURNS * 2)
    while cut_idx < len(convo_msgs) and convo_msgs[cut_idx].__class__.__name__ != "HumanMessage":
        cut_idx += 1
    to_remove = convo_msgs[:cut_idx]

    messages_to_remove = [RemoveMessage(id=m.id) for m in to_remove if m.id]
    if existing_summary_msg and existing_summary_msg.id:
        messages_to_remove.append(RemoveMessage(id=existing_summary_msg.id))
    return {"messages": messages_to_remove}


def summarize_history(agent, thread_id: str) -> None:
    """
    Background worker function that performs conversation summarization.
    Retrieves the current message history from the agent's checkpointer state,
    determines if the turns exceed MAX_HISTORY_TURNS * 2, generates a summary
    of the older turns via the LLM, and updates the thread state.
    """
    if not CONVERSATION_SUMMARIZATION_ENABLED:
        return

    logger.debug("Background thread: Starting summarization check for thread: %s", thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception as e:
        logger.debug("Could not retrieve state for thread %s (perhaps checkpointer not configured?): %s", thread_id, e)
        return

    messages = state.values.get("messages", []) if state.values else []
    if not messages:
        return

    # Extract existing summary and conversation messages
    existing_summary_msg = None
    convo_msgs = []
    for m in messages:
        if m.id == "conversation_summary" or (m.type == "system" and m.content.startswith("Resumen de la conversación anterior:")):
            existing_summary_msg = m
        else:
            convo_msgs.append(m)

    max_messages = MAX_HISTORY_TURNS * 2
    if len(convo_msgs) <= max_messages:
        logger.debug("Background thread: Thread %s messages (%d) under threshold (%d). No summary needed.", 
                     thread_id, len(convo_msgs), max_messages)
        return

    # Split into messages to remove and messages to keep
    cut_idx = len(convo_msgs) - max_messages
    while cut_idx < len(convo_msgs) and convo_msgs[cut_idx].__class__.__name__ != "HumanMessage":
        cut_idx += 1
    to_remove = convo_msgs[:cut_idx]

    if not to_remove:
        return

    try:
        summary_prompt = (
            "Resume la siguiente conversación anterior de forma muy concisa (máximo 150 palabras), "
            "capturando los puntos clave, preguntas del usuario, respuestas dadas y cualquier estado o "
            "reserva en curso. Este resumen servirá de contexto para continuar la conversación.\n\n"
        )
        if existing_summary_msg:
            summary_prompt += f"Resumen anterior:\n{existing_summary_msg.content}\n\n"

        summary_prompt += "Nuevos mensajes a resumir:\n"
        for m in to_remove:
            role = "Usuario" if m.type == "human" else "Asistente" if m.type == "ai" else m.type.capitalize()
            summary_prompt += f"{role}: {m.content}\n"

        logger.info("Background thread: Generating conversation summary for %d messages...", len(to_remove))
        summary_response = _llm.invoke(summary_prompt)
        summary_text = summary_response.content.strip()

        new_summary_msg = SystemMessage(
            content=f"Resumen de la conversación anterior: {summary_text}",
            id="conversation_summary"
        )

        messages_to_return = []
        if existing_summary_msg and existing_summary_msg.id:
            messages_to_return.append(RemoveMessage(id=existing_summary_msg.id))
        for m in to_remove:
            if m.id:
                messages_to_return.append(RemoveMessage(id=m.id))
        messages_to_return.append(new_summary_msg)

        logger.info("Background thread: Saving updated state for thread %s", thread_id)
        agent.update_state(config, {"messages": messages_to_return}, as_node="manage_memory")
        logger.info("Background thread: Summarization and state update completed successfully for thread %s.", thread_id)
    except Exception as e:
        logger.error("Background thread: Failed to generate background conversation summary for thread %s: %s", thread_id, e)



# ---------------------------------------------------------------------------
# Optional post-retrieval translation node  (Option C — FR-AGT-02)
# ---------------------------------------------------------------------------

def _translate_context(state: AgentState) -> dict:
    """
    Translate the last ``ToolMessage`` (retrieved documents) into the
    user's detected language.

    This node is only wired into the graph when
    ``TRANSLATE_CONTEXT_ENABLED = True`` in ``config.py``.
    If the user already writes in Spanish (the knowledge-base language),
    the node is a no-op.
    """
    lang_code = _detect_user_language(state["messages"])
    if lang_code == "es":
        return {"messages": []}          # docs are already in Spanish

    lang_name = _LANG_NAMES.get(lang_code, lang_code.title())

    # Find the most recent ToolMessage
    last_tool_msg: ToolMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            last_tool_msg = msg
            break

    if last_tool_msg is None or not last_tool_msg.content:
        return {"messages": []}

    translation_prompt = (
        f"Translate the following retrieved knowledge-base content into {lang_name}. "
        "Preserve all factual details, numbers, proper nouns, and formatting. "
        "Return ONLY the translated text, nothing else.\n\n"
        f"{last_tool_msg.content}"
    )

    try:
        translated = _llm.invoke([{"role": "user", "content": translation_prompt}])
        # Return a ToolMessage with the same id so add_messages replaces it
        updated = ToolMessage(
            content=translated.content,
            tool_call_id=last_tool_msg.tool_call_id,
            id=last_tool_msg.id,
        )
        logger.info("Translated retrieved context to %s", lang_name)
        return {"messages": [updated]}
    except Exception:
        logger.exception("Context translation failed; using original docs.")
        return {"messages": []}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None, *, skip_memory: bool = False, system_prompt_template: str | None = None):
    """
    Assembles and compiles the RAG agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.
                      If None, a default MemorySaver is used.
        skip_memory:  When True the ``manage_memory`` node is omitted and
                      no checkpointer is attached, producing a fully
                      stateless graph (useful for evaluation).
        system_prompt_template: Optional custom system prompt template to inject.

    Graph topology::

        START → manage_memory → call_model → [tools?]
                                                ├─ yes → retrieve [→ translate_context] → call_model
                                                └─ no  → END
    """
    workflow = StateGraph(AgentState)

    prompt_tpl = system_prompt_template or _SYSTEM_PROMPT_TEMPLATE
    workflow.add_node("call_model",       ModelCaller(prompt_tpl))
    workflow.add_node("retrieve",         ToolNode([retrieve_documents]))

    if not skip_memory:
        workflow.add_node("manage_memory", _manage_memory)
        workflow.add_edge(START, "manage_memory")
        workflow.add_edge("manage_memory", "call_model")
    else:
        workflow.add_edge(START, "call_model")

    workflow.add_conditional_edges(
        "call_model",
        tools_condition,
        {"tools": "retrieve", END: END},
    )

    if TRANSLATE_CONTEXT_ENABLED:
        workflow.add_node("translate_context", _translate_context)
        workflow.add_edge("retrieve", "translate_context")
        workflow.add_edge("translate_context", "call_model")
        logger.info("Graph built WITH post-retrieval translation node.")
    else:
        workflow.add_edge("retrieve", "call_model")

    cp = None if skip_memory else (checkpointer or MemorySaver())
    return workflow.compile(checkpointer=cp)


rag_agent = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
@deprecated(
    since="1.1.0",
    message="Use src.domain.orchestrator.RAGOrchestrator instead.",
    removal="2.0.0",
)
def generate_reply(platform: str, user_message: str, sender_id: str) -> str:

    Generate a reply for an incoming customer message.

    Args:
        platform:     ``"whatsapp"`` or ``"instagram"``
        user_message: The text the customer sent.
        sender_id:    The customer's unique ID / phone number, used as the
                      conversation thread key for memory persistence.

    Returns:
        A plain-text reply to send back to the customer.
    global rag_agent
    if rag_agent is None:
        rag_agent = build_graph(checkpointer=SqliteMemoryAdapter().get_checkpointer())
    
    logger.info("Generating reply | platform=%s sender=%s", platform, sender_id)

    result = rag_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        {"configurable": {"thread_id": sender_id}},
    )
    return result["messages"][-1].content"""


# ---------------------------------------------------------------------------
# Interactive smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    configure_logging()
    logger.info("RAG Agent — interactive mode. Type 'exit' or 'quit' to quit.")

    agent = build_graph()
    thread = "local-test"

    while True:
        try:
            user_input = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("Bye!")
            sys.exit(0)

        if user_input.lower() in ("exit", "quit", ""):
            break

        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": thread}},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
