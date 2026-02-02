import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Configuration
DATA_PATH = "data/documents"
DB_PATH = "data/chroma_db"
MODEL_NAME = "llama3"

def ingest_docs():
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
        print(d.source)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Create Embeddings & Store in Chroma
    print("Creating embeddings and storing in ChromaDB...")
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    
    # Initialize and persist ChromaDB
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function, 
        persist_directory=DB_PATH
    )
    
    print(f"Successfully ingested {len(documents)} documents into {DB_PATH}.")

if __name__ == "__main__":
    ingest_docs()
