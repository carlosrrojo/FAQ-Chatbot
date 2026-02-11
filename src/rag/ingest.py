import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import chromadb
import time


# Configuration
DATA_PATH = "data/documents"
DB_PATH = "data/chroma_db"
MODEL_NAME = "llama3"
BATCH_SIZE = 100

def reset_db():
    print(f"Resetting database at {DB_PATH} (clearing collection 'langchain')...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("langchain")
        print("Deleted existing collection 'langchain'")
    except Exception as e:
        print(f"Collection 'langchain' could not be deleted (might not exist): {e}")

def load_file(file_path):
    """Load a single file based on its extension."""
    try:
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            return []
        
        return loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def ingest_docs(clear_db=False):
    if clear_db:
        reset_db()

    print(f"Scanning documents in {DATA_PATH}...")
    
    all_documents = []
    
    if not os.path.exists(DATA_PATH):
        print(f"Directory {DATA_PATH} does not exist.")
        return

    # Iterate over files in the directory
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith((".txt", ".pdf")):
                print(f"Loading {file}...")
                docs = load_file(file_path)
                if docs:
                    all_documents.extend(docs)
                    print(f"  Loaded {len(docs)} document(s) from {file}")
                else:
                    print(f"  Skipped {file} (empty or error)")

    if not all_documents:
        print("No documents found/loaded.")
        return

    print(f"Total loaded documents: {len(all_documents)}")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks.")

    # Create Embeddings & Store in Chroma
    print("Initializing ChromaDB and embeddings...")
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    
    # Initialize Chroma client
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function,
        collection_name="langchain"
    )

    print(f"Inserting chunks into ChromaDB in batches of {BATCH_SIZE}...")
    
    total_chunks = len(chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"  Processing batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} chunks)...")
        try:
            db.add_documents(batch)
        except Exception as e:
            print(f"  Error inserting batch {i}: {e}")
            
    print(f"Successfully ingested {len(all_documents)} documents ({len(chunks)} chunks) into {DB_PATH}.")

if __name__ == "__main__":
    ingest_docs(clear_db=True)
