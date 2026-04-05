import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BENCHMARK_DIR = "benchmarks"


def load_benchmark(benchmark_name: str) -> list[str]:
    """Load a benchmark file and return its non-empty lines."""
    benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_name)
    with open(benchmark_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_sections(
    embeddings: OllamaEmbeddings,
    vectorstore: Chroma,
) -> set[str]:
    """Return the set of unique section names stored in the vector store."""
    data = vectorstore.get()
    sections: set[str] = set()
    for meta in data.get("metadatas", []):
        if meta and "section" in meta:
            sections.add(meta["section"])
    return sections