import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import hashlib
import json
import os
from datetime import datetime, timezone
from src.config import MANIFEST_PATH, COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
import logging

BENCHMARK_DIR = "benchmarks"

logger = logging.getLogger(__name__)

def load_benchmark(benchmark_name: str) -> list[str]:
    """Load a benchmark file and return its non-empty lines."""
    benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_name)
    with open(benchmark_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_sections(
    vectorstore: Chroma,
) -> set[str]:
    """Return the set of unique section names stored in the vector store."""
    data = vectorstore.get()
    sections: set[str] = set()
    for meta in data.get("metadatas", []):
        if meta and "section" in meta:
            sections.add(meta["section"])
    return sections


def stable_id(content: str, metadata: dict) -> str:
    """SHA-256 of content + section metadata = stable, collision-resistant ID."""
    key = content + metadata.get("section", "") + metadata.get("parent_section", "")
    return hashlib.sha256(key.encode()).hexdigest()[:32]


# Add to end of src/rag/ingest.py, inside the main ingestion function
def write_manifest(source_files: list[str], chunks_per_file: dict[str, int]) -> None:
    """
    Writes a JSON manifest capturing this ingestion run's provenance.
    FR-ING-07: Enables evaluation result traceability.
    """
    manifest = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_chunks": sum(chunks_per_file.values()),
        "documents": [
            {
                "filename": os.path.basename(f),
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(f), tz=timezone.utc
                ).isoformat(),
                "chunk_count": chunks_per_file.get(f, 0),
            }
            for f in source_files
        ],
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Ingestion manifest written to %s", MANIFEST_PATH)