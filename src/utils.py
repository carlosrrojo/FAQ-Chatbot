import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BENCHMARK_DIR = "benchmarks"

def load_benchmark(benchmark_name: str) -> list[str]:
    benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_name)
    with open(benchmark_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


"""
Retrieves all sections from the vectorstore.
"""
def get_sections(embeddings, vectorstore):
    data = vectorstore.get()
    sections = set()
    for meta in data.get("metadatas", []):
        if meta and "section" in meta:
            sections.add(meta["section"])
    return set(sections)