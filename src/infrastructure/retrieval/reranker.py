from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_encoder: CrossEncoder | None = None

def get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(RERANK_MODEL)
    return _encoder

def rerank(
    query: str,
    docs: list[Document],
    top_n: int,
) -> list[Document]:
    """Return top_n docs reranked by cross-encoder score."""
    if not docs:
        return docs
    encoder = get_encoder()
    pairs   = [(query, doc.page_content) for doc in docs]
    scores  = encoder.predict(pairs)          # shape: (len(docs),)
    ranked  = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    from src.config import RELEVANCE_THRESHOLD
    filtered_ranked = [(doc, score) for doc, score in ranked if score >= RELEVANCE_THRESHOLD]

    logger.debug("🏆  RERANK RESULTS — top %d (Threshold: %.4f)", top_n, RELEVANCE_THRESHOLD)
    logger.debug("-" * 70)
    for rank, (doc, score) in enumerate(filtered_ranked, start=1):
        snippet = doc.page_content[:120].replace("\n", " ")
        logger.debug("  [%2d] rerank_score=%.4f  | %s…", rank, score, snippet)
    logger.debug("=" * 70 + "\n")
    
    return [doc for doc, _ in filtered_ranked[:top_n]]
