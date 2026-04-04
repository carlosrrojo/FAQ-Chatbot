from pathlib import Path

# Project root = FAQ-Chatbot/ (two levels up from src/rag/config.py)
_ROOT = Path(__file__).resolve().parents[2]

# Shared configuration for the RAG pipeline
MODEL_NAME = "llama3.1"
DATA_PATH  = str(_ROOT / "data" / "documents")
DB_PATH    = str(_ROOT / "data" / "chroma_db")
COLLECTION = "espazo_nature_keybert"
