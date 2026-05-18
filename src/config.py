import os
"""
Centralised configuration for the RAG agent.

All tunable constants live here so they never drift between modules.
"""

# ── LLM / Embeddings ─────────────────────────────────────────────────────────
MODEL_NAME = "llama3.1"
EMBEDDING_MODEL: str = "llama3.1"

# ── Storage paths ──────────────────────────────────────────────────────────────
DB_PATH    = "data/chroma_db"
DATA_PATH  = "data/documents"
COLLECTION = "metadata_espazo_nature_1024"

MEMORY_DB_PATH: str = "data/memory.sqlite"        # FR-MEM-02
MANIFEST_PATH: str = "data/chroma_db/ingest_manifest.json"   # FR-ING-07

# ── ChromaDB collection ───────────────────────────────────────────────────────
COLLECTION: str = "metadata_espazo_nature_1024"
CHUNK_SIZE: int = 1024
CHUNK_OVERLAP: int = 256

# ── Retrieval pipeline hyperparameters ────────────────────────────────────────────────────────
HYBRID_K = 20  # candidates fetched from each index (dense + sparse) before RRF
RRF_K    = 60  # RRF smoothing constant (standard default)
TOP_K    = 3   # documents returned by the cross-encoder reranker (final stage) to the LLM
RELEVANCE_THRESHOLD: float = -5.0    # FR-AGT-04: cross-encoder OOS threshold

# ── Keyword augmentation ──────────────────────────────────────────────────────
KEYWORD_AUGMENTATION_ENABLED: bool = False   # FR-RET-06: explicit off by default

# ── Metadata filtering ────────────────────────────────────────────────────────
FILTER_CONFIDENCE_THRESHOLD: float = 0.6     # Min fuzzy match score to apply section filter

# ── Memory / session ─────────────────────────────────────────────────────────
MAX_HISTORY_TURNS: int = 20    # FR-MEM-03: sliding window (turns = user+assistant pairs)
SESSION_TTL_DAYS: int = 30     # FR-PRV-01: data retention policy

# ── Platform constraints ──────────────────────────────────────────────────────
WHATSAPP_MAX_CHARS: int = 4096     # FR-CHN-06
INSTAGRAM_MAX_CHARS: int = 1000    # Instagram DM practical limit

# ── Telemetry ─────────────────────────────────────────────────────────────────
TELEMETRY_ENABLED: bool = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"