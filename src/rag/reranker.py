from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

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
    return [doc for doc, _ in ranked[:top_n]]
