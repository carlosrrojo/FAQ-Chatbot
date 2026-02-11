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


def reset_db():
    print(f"Resetting database at {DB_PATH} (clearing collection 'langchain')...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("langchain")
        print("Deleted existing collection 'langchain'")
    except Exception as e:
        print(f"Collection 'langchain' could not be deleted (might not exist): {e}")

def ingest_docs(clear_db=False):
    if clear_db:
        reset_db()

    print(f"Loading documents from {DATA_PATH}...")
    # Load all .txt files
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    txt_documents = txt_loader.load()

    # Load all .pdf files
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    documents = txt_documents + pdf_documents
    
    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents:")
    for d in documents:
        print(d.metadata.get('source', 'Unknown'))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Create Embeddings & Store in Chroma
    print("Creating embeddings and storing in ChromaDB...")
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    
    # Initialize and persist ChromaDB
    # We use the same 'langchain' collection name implicitly
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function, 
        persist_directory=DB_PATH,
        collection_name="langchain" 
    )
    
    print(f"Successfully ingested {len(documents)} documents into {DB_PATH}.")

if __name__ == "__main__":
    ingest_docs(clear_db=True)
