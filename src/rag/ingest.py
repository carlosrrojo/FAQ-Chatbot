from src.rag.metadata_extractor import MetadataExtractor, enrich_document
import glob as glob_module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.rag.processor import parse_doc, get_children
from src.rag.config import MODEL_NAME, DATA_PATH, DB_PATH, COLLECTION
from src.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)

CHUNK_SIZE      = 1024
CHUNK_OVERLAP   = 256

# 1. Load each PDF and split into sections using headings
docs: list[Document] = []
for pdf_path in glob_module.glob(f"{DATA_PATH}/*.pdf"):
    docs.extend(parse_doc(pdf_path))
logger.info("Loaded %d documents.", len(docs))

# 2. Split into chunks (tune chunk_size to your doc type)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, # size of each chunk. llama3.1 context window is 2048
    chunk_overlap=CHUNK_OVERLAP,   # overlap keeps context across chunks
    add_start_index=True # track index in original document
)

chunks = splitter.split_documents(docs)
logger.info("Splited into %d chunks.", len(chunks))

# 3. Extract metadata with LLM
extractor = MetadataExtractor(MODEL_NAME)
for i, chunk in enumerate(chunks):
    extras = get_children(chunk.metadata["section"], chunks)
    # Seed base metadata preserved from your existing schema
    chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
    chunk.metadata["index_start"] = i * (CHUNK_SIZE - CHUNK_OVERLAP)
    chunk.metadata["entity_id"]   = None
    enrich_document(chunk, extractor, None, extras)
logger.info("Added extra metadata to documents. Current chunks: %d", len(chunks))

# 4. Embed and persist to vector store
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

logger.info("ingested to: %s in %s", COLLECTION, DB_PATH)
document_ids = vectorstore.add_documents(chunks)
logger.info("Stored %d documents using %d/%d split.", len(document_ids), CHUNK_SIZE, CHUNK_OVERLAP)
