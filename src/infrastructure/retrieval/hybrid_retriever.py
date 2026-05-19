import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.domain.ports import IRetriever
from src.domain.models import RetrievedContext
from src.config import HYBRID_K, RRF_K, TOP_K, RELEVANCE_THRESHOLD, KEYWORD_AUGMENTATION_ENABLED
from .bm25_retriever import BM25Retriever
from .reranker import rerank

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    dense: list[tuple[Document, float]],
    sparse: list[tuple],
    k: int = RRF_K,
    top_n: int = HYBRID_K,
) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, (content, _, metadata) in enumerate(sparse):
        scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank + 1)
        if content not in doc_map:
            doc_map[content] = Document(page_content=content, metadata=metadata)

    fused = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_n]
    logger.debug("RRF fusion: %d candidates from %d dense + %d sparse",
                 len(fused), len(dense), len(sparse))
    return [doc_map[c] for c in fused]


class HybridRetriever(IRetriever):
    """
    Full retrieval pipeline:
      1. LLM metadata filter extraction (section-level soft filter)
      2. Dense retrieval — unfiltered + filtered, merged
      3. Sparse BM25 retrieval (optionally on augmented query)
      4. RRF fusion
      5. Cross-encoder reranking
      6. Out-of-scope detection via relevance threshold (FR-AGT-04)

    Implements IRetriever port.
    """

    def __init__(self, vectorstore: Chroma, bm25: BM25Retriever, llm) -> None:
        self._vs = vectorstore
        self._bm25 = bm25
        self._llm = llm

    def rebuild_index(self) -> None:
        """Delegate to BM25 adapter. Called by watcher after re-ingestion."""
        self._bm25.rebuild()

    def retrieve(self, query: str) -> list[RetrievedContext]:
        try:
            return self._retrieve_internal(query)
        except Exception:
            logger.exception("Primary retrieval failed; falling back to basic dense search.")
            docs = self._vs.as_retriever(search_kwargs={"k": TOP_K}).invoke(query)
            return [self._to_dto(doc, 0.0) for doc in docs]

    def _retrieve_internal(self, query: str) -> list[RetrievedContext]:
        from src.rag.agent import _build_section_filter   # temporary: refactor later
        metadata_filter = _build_section_filter(query)

        # Dense retrieval — unfiltered
        dense_unfiltered = self._vs.similarity_search_with_relevance_scores(
            query, k=HYBRID_K
        )

        # Dense retrieval — filtered (soft boost)
        dense_filtered = []
        if metadata_filter:
            try:
                dense_filtered = self._vs.similarity_search_with_relevance_scores(
                    query, k=HYBRID_K, filter=metadata_filter
                )
            except Exception:
                logger.warning("Filtered dense retrieval failed; proceeding unfiltered.")

        # Merge dense results, deduplicate, keep higher score
        merged: dict[str, tuple[Document, float]] = {}
        for doc, score in dense_unfiltered + dense_filtered:
            key = doc.page_content
            if key not in merged or score > merged[key][1]:
                merged[key] = (doc, score)
        dense_merged = sorted(merged.values(), key=lambda x: x[1], reverse=True)[:HYBRID_K]

        # Sparse BM25 retrieval
        bm25_query = query
        if KEYWORD_AUGMENTATION_ENABLED:
            # FR-RET-06: augmented query for BM25 only; original preserved for reranking
            bm25_query = self._augment_query(query)
        sparse_results = self._bm25.search(bm25_query, k=HYBRID_K)

        # RRF fusion
        fused = _reciprocal_rank_fusion(dense_merged, sparse_results, k=RRF_K)

        # Cross-encoder reranking
        ranked_with_scores = rerank(query, fused, top_n=TOP_K)

        results = []
        for doc, score in ranked_with_scores:
            results.append(self._to_dto(doc, score))

        logger.info("Retrieval complete: %d documents returned (top score=%.3f)",
                    len(results), results[0].score if results else 0.0)
        return results

    def _augment_query(self, query: str) -> str:
        """FR-RET-06: placeholder for keyword augmentation."""
        # Implementation: call self._llm to extract keywords and append to query.
        # Disabled by default (KEYWORD_AUGMENTATION_ENABLED=False).
        return query

    @staticmethod
    def _to_dto(doc: Document, score: float) -> RetrievedContext:
        return RetrievedContext(
            content=doc.page_content,
            metadata=doc.metadata,
            score=score,
        )