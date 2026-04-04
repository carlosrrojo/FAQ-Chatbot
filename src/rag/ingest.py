from processor import add_metadata_keyBERT
import glob as glob_module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from processor import docs_into_sections, add_metadata
from config import MODEL_NAME, DATA_PATH, DB_PATH, COLLECTION

# 1. Load each PDF and split into sections using headings
docs: list[Document] = []
for pdf_path in glob_module.glob(f"{DATA_PATH}/*.pdf"):
    docs.extend(docs_into_sections(pdf_path))
print(f"Loaded {len(docs)} documents.")

# 2. Split into chunks (tune chunk_size to your doc type)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, # size of each chunk. llama3.1 context window is 2048
    chunk_overlap=100,   # overlap keeps context across chunks
    add_start_index=True # track index in original document
)

chunks = splitter.split_documents(docs)
print(f"Splited into {len(chunks)} chunks.")


# 3. Extract metadata with LLM
chunks = add_metadata_keyBERT(chunks)
print(f"Added extra metadata to documents. Current chunks: {len(chunks)}")

# 4. Embed and persist to vector store
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

document_ids = vectorstore.add_documents(chunks)
print(f"Stored {len(document_ids)} documents using 800/100 split.")