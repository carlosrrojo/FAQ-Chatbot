from rank_bm25 import BM25Okapi
from src.config import HYBRID_K
from langchain_chroma import Chroma
import regex
import threading
import logging

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, vectorstore: Chroma) -> None:
        self.vectorstore = vectorstore
        self._lock = threading.RLock()
        self._docs : list = []
        self._index: BM25Okapi | None = None
        self._build()

    
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """
        Unicode-aware tokeniser for Spanish/Galician text.
        Strips leading/trailing punctuation so accented words like
        '¿habitación?' are correctly reduced to 'habitación'.
        """
        tokens = []
        for tok in text.lower().split():
            tok = regex.sub(r'^\p{P}+|\p{P}+$', '', tok)
            if tok:
                tokens.append(tok)
        return tokens

    def _build(self) -> None:
        """Internal: construct the index from the current ChromaDB state."""
        collection = self._vectorstore.get()
        self._docs = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])
        if not self._docs:
            logger.warning("BM25Retriever: ChromaDB collection is empty.")
            return
        tokenised = [self._tokenise(doc) for doc in self._docs]
        self._index = BM25Okapi(tokenised)
        self._metadatas = metadatas
        logger.info("BM25 index built: %d documents.", len(self._docs))
    
    def rebuild(self) -> None:
        """
        Public signal: called by the file watcher after re-ingestion.
        Thread-safe: acquires the lock before rebuilding.
        """
        with self._lock:
            logger.info("BM25Retriever: rebuilding index after re-ingestion.")
            self._build()
    
    def search(self, query: str, k: int = HYBRID_K) -> list[tuple]:
        """
        Returns top-k (doc_text, normalised_score) tuples.
        Augmented query support: if keyword augmentation is enabled in config,
        the caller should pass the augmented query string here (FR-RET-06).
        """
        with self._lock:
            if self._index is None or not self._docs:
                logger.warning("BM25Retriever: index not available, returning empty.")
                return []
            tokens = self._tokenise(query)
            scores = self._index.get_scores(tokens)
            max_score = max(scores) if max(scores) > 0 else 1.0
            normalised = scores / max_score
            top_indices = sorted(
                range(len(normalised)), key=lambda i: normalised[i], reverse=True
            )[:k]
            results = []
            for idx in top_indices:
                if normalised[idx] > 0:
                    results.append((self._docs[idx], float(normalised[idx]),
                                    self._metadatas[idx] if self._metadatas else {}))
            logger.debug("BM25 search: %d results for query '%s'", len(results), query[:60])
            return results
