from src.rag.metadata_extractor import MetadataExtractor, enrich_document
import glob as glob_module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.rag.processor import parse_doc, get_siblings
from src.rag.config import MODEL_NAME, DATA_PATH, DB_PATH, COLLECTION
# from processor import add_metadata_keyBERT

CHUNK_SIZE      = 1024
CHUNK_OVERLAP   = 256

# 1. Load each PDF and split into sections using headings
docs: list[Document] = []
for pdf_path in glob_module.glob(f"{DATA_PATH}/*.pdf"):
    docs.extend(parse_doc(pdf_path))
print(f"Loaded {len(docs)} documents.")

# 2. Split into chunks (tune chunk_size to your doc type)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, # size of each chunk. llama3.1 context window is 2048
    chunk_overlap=CHUNK_OVERLAP,   # overlap keeps context across chunks
    add_start_index=True # track index in original document
)

chunks = splitter.split_documents(docs)
print(f"Splited into {len(chunks)} chunks.")

# 3. Extract metadata with LLM
extractor = MetadataExtractor(MODEL_NAME)
for i, chunk in enumerate(chunks):
    extras = get_siblings(chunk.metadata["section"], chunks)
    # Seed base metadata preserved from your existing schema
    chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
    chunk.metadata["index_start"] = i * (CHUNK_SIZE - CHUNK_OVERLAP)
    chunk.metadata["entity_id"]   = None
    enrich_document(chunk, extractor, None, extras)
#chunks = add_metadata_keyBERT(chunks)
print(f"Added extra metadata to documents. Current chunks: {len(chunks)}")

# 4. Embed and persist to vector store
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

print(f"ingested to: {COLLECTION} in {DB_PATH}")
document_ids = vectorstore.add_documents(chunks)
print(f"Stored {len(document_ids)} documents using {CHUNK_SIZE}/{CHUNK_OVERLAP} split.")
