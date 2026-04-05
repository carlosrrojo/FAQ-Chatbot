"""
Centralised configuration for the RAG agent.

All tunable constants live here so they never drift between modules.
"""

# ── LLM / Embeddings ─────────────────────────────────────────────────────────
MODEL_NAME = "llama3.1"

# ── Vector store ──────────────────────────────────────────────────────────────
DB_PATH    = "data/chroma_db"
COLLECTION = "recursive_espazo_nature"

# ── Retrieval pipeline ────────────────────────────────────────────────────────
HYBRID_K = 20  # candidates fetched from each index (dense + sparse) before RRF
RRF_K    = 60  # RRF smoothing constant (standard default)
RERANK_K = 8   # pool size passed to the cross-encoder reranker
TOP_K    = 3   # final chunks returned to the LLM after reranking
