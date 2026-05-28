from sqlalchemy import true
import os
import json
from dotenv import load_dotenv

load_dotenv()

"""
Centralised configuration for the RAG agent.

All tunable constants live here so they never drift between modules.
"""

# ── LLM / Embeddings ─────────────────────────────────────────────────────────
MODEL_NAME = "llama3.1"
EMBEDDING_MODEL: str = "BAAI/bge-m3"

EMBEDDING_PROVIDER = "huggingface"          # "ollama" | "huggingface"
EMBEDDING_DEVICE = "cpu"                    # "cpu" | "cuda" | "mps"
EMBEDDING_NORMALIZE = True                  # bge-m3 expects L2-normalised vectors
EMBEDDING_BATCH_SIZE = 12

# ── Storage paths ──────────────────────────────────────────────────────────────
DB_PATH    = "data/chroma_db"
DATA_PATH  = "data/documents"

MEMORY_DB_PATH: str = "data/memory.sqlite"        # FR-MEM-02
MANIFEST_PATH: str = "data/chroma_db/ingest_manifest.json"   # FR-ING-07
METADATA_CACHE_DB_PATH: str = "data/metadata_cache.sqlite"   # FR-ING-06: LLM extraction cache

# ── ChromaDB ───────────────────────────────────────────────────────
CHUNK_SIZE: int = 1024
CHUNK_OVERLAP: int = 256
ACTIVE_COLLECTION_PATH = os.path.join(DB_PATH, "active_collection.json")

def get_active_collection_name() -> str:
    base_name = EMBEDDING_MODEL.replace("/", "-") + "_espazo_nature_" + str(CHUNK_SIZE)
    if os.path.exists(ACTIVE_COLLECTION_PATH):
        try:
            with open(ACTIVE_COLLECTION_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data.get("active_collection", base_name)
        except Exception:
            pass
    return base_name

COLLECTION = get_active_collection_name()

# ── Retrieval pipeline hyperparameters ────────────────────────────────────────────────────────
HYBRID_K = 20  # candidates fetched from each index (dense + sparse) before RRF
RRF_K    = 60  # RRF smoothing constant (standard default)
TOP_K    = 3   # documents returned by the cross-encoder reranker (final stage) to the LLM
RELEVANCE_THRESHOLD: float = -5.0    # FR-AGT-04: cross-encoder OOS threshold

# ── Keyword augmentation ──────────────────────────────────────────────────────
KEYWORD_AUGMENTATION_ENABLED: bool = False   # FR-RET-06: explicit off by default

# ── Metadata filtering ────────────────────────────────────────────────────────
FILTER_CONFIDENCE_THRESHOLD: float = 0.6     # Min fuzzy match score to apply section filter

# ── Language handling ─────────────────────────────────────────────────────────
TRANSLATE_CONTEXT_ENABLED: bool = False       # FR-AGT-02: translate retrieved docs to user language

# ── Memory / session ─────────────────────────────────────────────────────────
MAX_HISTORY_TURNS: int = 20    # FR-MEM-03: sliding window (turns = user+assistant pairs)
CONVERSATION_SUMMARIZATION_ENABLED: bool = True
SESSION_TTL_DAYS: int = 30     # FR-PRV-01: data retention policy
RETENTION_CHECK_INTERVAL_HOURS: int = 24  # FR-PRV-01: purge cycle (hours)


# ── Platform constraints ──────────────────────────────────────────────────────
WHATSAPP_MAX_CHARS: int = 4096     # FR-CHN-06
INSTAGRAM_MAX_CHARS: int = 1000    # Instagram DM practical limit

# ── Telemetry ─────────────────────────────────────────────────────────────────
TELEMETRY_ENABLED: bool = True

# ── Deduplication ─────────────────────────────────────────────────────────────
DEDUP_TTL_SECONDS: int = 300       # 5 min — covers Meta's retry window
DEDUP_MAX_SIZE: int = 10_000       # Max tracked message IDs before eviction

# ── Backups ───────────────────────────────────────────────────────────────────
BACKUP_DIR = os.getenv("BACKUP_DIR", "data/backups")
BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "7"))