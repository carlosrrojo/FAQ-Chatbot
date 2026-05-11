from typing import Optional
import logging
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# Hybrid search
HYBRID_K          = 20    # candidates fetched from each index before RRF merge
RRF_K             = 60    # RRF smoothing constant (standard default)
TOP_K             = 4     # final chunks returned after fusion

# ---------------------------------------------------------------------------
# BM25Index — sparse keyword index kept in sync with Chroma
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Lightweight BM25 index built over the same documents stored in Chroma.

    Lifecycle
    ---------
    - Call build(docs) once after loading/ingesting documents.
    - Call add(docs) incrementally when new documents are ingested.
    - Held in RAM; rebuilt from Chroma on RAGCore init.

    Tokenisation
    ------------
    Lowercased whitespace split — fast and sufficient for English customer
    service text. Swap for a proper tokeniser (nltk, spacy) if needed.
    """

    def __init__(self):
        self._docs:   list[Document] = []
        self._bm25:   Optional[BM25Okapi] = None

    def build(self, docs: list[Document]) -> None:
        """(Re)build the index from a list of LangChain Documents."""
        self._docs  = docs
        corpus      = [self._tokenise(d.page_content) for d in docs]
        self._bm25  = BM25Okapi(corpus)

    def add(self, docs: list[Document]) -> None:
        """Incrementally add documents and rebuild (cheap for <100k chunks)."""
        self._docs.extend(docs)
        self.build(self._docs)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        Return top-k (Document, normalised_bm25_score) pairs.
        Scores are normalised to [0, 1] by dividing by the max score in the
        result set so they sit on the same scale as cosine similarity.
        """
        if self._bm25 is None or not self._docs:
            return []

        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each doc with its score and sort descending
        ranked = sorted(
            zip(self._docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        # Normalise scores to [0, 1]
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        return [(doc, round(score / max_score, 4)) for doc, score in ranked]

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.lower().split()



# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_hits:  list[tuple[Document, float]],
    sparse_hits: list[tuple[Document, float]],
    k:           int = RRF_K,
    top_n:       int = TOP_K,
) -> list[tuple[Document, float]]:
    """
    Merge dense (Chroma) and sparse (BM25) result lists using RRF.

    RRF score for document d:
        rrf(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the 1-based position of d in list i.

    Documents are deduplicated by page_content. The fused score is
    normalised to [0, 1] for consistency with the rest of the pipeline.

    Parameters
    ----------
    dense_hits  : (Document, cosine_score) from Chroma
    sparse_hits : (Document, bm25_score)   from BM25Index
    k           : RRF smoothing constant   (default 60)
    top_n       : number of results to return after fusion
    """
    # ── DEBUG: Dense retriever results ──
    logger.debug("\n" + "=" * 70)
    logger.debug("🔍  DENSE RETRIEVER (Chroma) — %d hits", len(dense_hits))
    logger.debug("-" * 70)
    for rank, (doc, score) in enumerate(dense_hits, start=1):
        snippet = doc.page_content[:120].replace("\n", " ")
        logger.debug("  [%2d] score=%.4f  | %s…", rank, score, snippet)
    logger.debug("")

    # ── DEBUG: Sparse retriever results ──
    logger.debug("📝  SPARSE RETRIEVER (BM25) — %d hits", len(sparse_hits))
    logger.debug("-" * 70)
    for rank, (doc, score) in enumerate(sparse_hits, start=1):
        snippet = doc.page_content[:120].replace("\n", " ")
        logger.debug("  [%2d] score=%.4f  | %s…", rank, score, snippet)
    logger.debug("")

    scores: dict[str, float]   = {}
    docs:   dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_hits, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        docs[key]   = doc

    for rank, (doc, _) in enumerate(sparse_hits, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        docs[key]   = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Normalise fused scores to [0, 1]
    max_score = ranked[0][1] if ranked else 1.0
    fused = [(docs[key], round(score / max_score, 4)) for key, score in ranked]

    # ── DEBUG: Final fused results ──
    logger.debug("🏆  RRF FUSED RESULTS — top %d", top_n)
    logger.debug("-" * 70)
    for rank, (doc, score) in enumerate(fused, start=1):
        snippet = doc.page_content[:120].replace("\n", " ")
        logger.debug("  [%2d] rrf_score=%.4f  | %s…", rank, score, snippet)
    logger.debug("=" * 70 + "\n")

    return fused