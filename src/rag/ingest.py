from src.utils import write_manifest
from src.rag.metadata_extractor import MetadataExtractor, enrich_document
import glob as glob_module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.rag.processor import parse_doc, get_children
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, EMBEDDING_MODEL,
    DATA_PATH, DB_PATH, COLLECTION, METADATA_CACHE_DB_PATH,
)
from src.infrastructure.cache.metadata_cache import MetadataCache
from src.logging_config import configure_logging
from src.utils import stable_id
import logging
import chromadb
from src.infrastructure.embeddings import get_embeddings
import time
import json
import os
from src.config import ACTIVE_COLLECTION_PATH

configure_logging()
logger = logging.getLogger(__name__)

def run_ingest() -> None:
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

    # 3. Extract metadata with LLM (backed by persistent cache)
    cache = MetadataCache(METADATA_CACHE_DB_PATH)
    try:
        extractor = MetadataExtractor(MODEL_NAME, cache=cache)
        for i, chunk in enumerate(chunks):
            extras = get_children(chunk.metadata["section"], chunks)
            # Seed base metadata preserved from your existing schema
            chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
            chunk.metadata["index_start"] = i * (CHUNK_SIZE - CHUNK_OVERLAP)
            chunk.metadata["entity_id"]   = None
            enrich_document(chunk, extractor, None, extras)
        logger.info("Added extra metadata to documents. Current chunks: %d", len(chunks))
    finally:
        cache.close()

    # 4. Embed and persist to vector store


    embeddings = get_embeddings()

    client = chromadb.PersistentClient(path=DB_PATH)

    # Determine new collection name with timestamp
    base_collection = EMBEDDING_MODEL.replace("/", "-") + "_espazo_nature_" + str(CHUNK_SIZE)
    new_collection = f"{base_collection}_{int(time.time())}"
    logger.info("Creating new collection for ingestion: %s", new_collection)

    vectorstore = Chroma(
        collection_name=new_collection,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    # Assigning stable content-derived IDs for idempotent ingestion.
    # Determinism is guaranteed by the metadata cache: LLM-derived fields in
    # chunk.metadata are frozen after the first extraction, so stable_id()
    # will always produce the same hash for identical source content.
    ids = [stable_id(chunk.page_content, chunk.metadata) for chunk in chunks]
    logger.info("Ingesting into temporary/new collection: %s", new_collection)
    document_ids = vectorstore.add_documents(chunks, ids=ids)
    logger.info("Stored %d documents in new collection.", len(document_ids))

    # --- VALIDATION ---
    logger.info("Validating new collection %s...", new_collection)
    inserted_count = vectorstore._collection.count()
    if inserted_count == 0:
        raise ValueError(f"Validation failed: Collection {new_collection} is empty after ingestion.")

    if inserted_count < len(chunks):
        raise ValueError(f"Validation failed: Document count mismatch. Expected at least {len(chunks)}, got {inserted_count}.")

    # Run a quick test query to ensure retrieval is functional
    test_query_results = vectorstore.similarity_search("glamping", k=1)
    if not test_query_results:
        raise ValueError(f"Validation failed: Test query returned no results from {new_collection}.")

    logger.info("Validation successful for collection %s.", new_collection)

    # --- ATOMIC SWAP ---
    # Write the new collection name to the active collection pointer file
    os.makedirs(os.path.dirname(ACTIVE_COLLECTION_PATH), exist_ok=True)
    with open(ACTIVE_COLLECTION_PATH, "w", encoding="utf-8") as fh:
        json.dump({"active_collection": new_collection}, fh, indent=2)
    logger.info("Atomic swap completed: active collection is now %s.", new_collection)

    # --- CLEANUP ---
    # Delete obsolete collections (anything starting with base_collection that is not the active one)
    try:
        cols = client.list_collections()
        for col in cols:
            name = col.name
            if name.startswith(base_collection) and name != new_collection:
                logger.info("Deleting obsolete collection: %s", name)
                client.delete_collection(name)
    except Exception as e:
        logger.warning("Error cleaning up obsolete collections: %s", e)

    # Build per-file chunk counts
    source_files = glob_module.glob(f"{DATA_PATH}/*.pdf")
    chunks_per_file = {f: sum(1 for c in chunks if c.metadata.get("source") == f) for f in source_files}
    write_manifest(source_files=source_files, chunks_per_file=chunks_per_file, collection_name=new_collection)

if __name__ == "__main__":
    run_ingest()