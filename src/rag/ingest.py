import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import chromadb

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

def extract_metadata(content: str, llm: ChatOllama) -> dict:
    pass

def ingest_docs(clear_db=False):
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vector_store = Chroma(
        collection_name="espazo_nature",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    # Loading docs
    loader = PyPDFLoader(os.path.join(DATA_PATH, "INFORMACIÓN PARA EL BOT.pdf")) # Hardcoded for now

    docs = loader.load()

    assert len(docs) > 0, "No documents loaded"
    
    print(f"Total characters: {len(docs[0].page_content)}")
    print(docs[0].page_content[:500])

    # Splitting docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    # Storing documents
    document_ids = vector_store.add_documents(documents=all_splits)
    print(document_ids[:3])

    

if __name__ == "__main__":
    ingest_docs(clear_db=True)
