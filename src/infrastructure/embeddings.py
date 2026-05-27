from functools import lru_cache
from langchain_core.embeddings import Embeddings
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from src import config

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Lazy, cached embedding-model factory.

    The lru_cache ensures the model is loaded into memory exactly once
    per process, which matters because sentence-transformers model load
    takes 5–15 s and ~2 GB RAM for bge-m3.
    """
    if config.EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": config.EMBEDDING_NORMALIZE,
                "batch_size": config.EMBEDDING_BATCH_SIZE,
            },
        )
    elif config.EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {config.EMBEDDING_PROVIDER}")