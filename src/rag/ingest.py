import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import chromadb
import sys

# Configuration
DATA_PATH = "data/documents"
DB_PATH = "data/chroma_db"
MODEL_NAME = "llama3.1"
COLLECTION_NAME = "espazo_nature"


# Reset database - Erases the collection
def reset_db():
    print(f"Resetting database at {DB_PATH} (clearing collection '{COLLECTION_NAME}')...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' could not be deleted (might not exist): {e}")

def extract_metadata(content: str, llm: ChatOllama) -> dict:
    pass

def load_documents(format: str = "txt") -> list[Document]:
    print(f"Loading documents from {DATA_PATH}...")
    # Loading docs
    docs = []
    loader = None
    for file in os.listdir(DATA_PATH):
        if format=="txt" and file.endswith("txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file))
        elif format=="pdf" and file.endswith("pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        elif format=="md" and file.endswith("md"):
            loader = TextLoader(os.path.join(DATA_PATH, file)) # Using TextLoader instead of undefined MarkdownLoader
        if loader == None:
            continue
        docs.extend(loader.load())
    return docs

# Process image, tables and other types of data (not neccessary for now)
def process_documents(docs: list[Document]) -> list[Document]:
    return docs

#======================
#= CHUNKING STRATEGIES=
#======================
def md_chunking_strategie(docs):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    md_docs = []
    for doc in docs:
        # MarkdownHeaderTextSplitter works on raw text and returns Document objects with header metadata
        splits = md_splitter.split_text(doc.page_content)
        for split in splits:
            # Merge original document metadata (like 'source') with the new header metadata
            split.metadata.update(doc.metadata)
            md_docs.append(split)
    return md_docs

def naive_chunking_strategie(docs):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # chunk size (characters)
            chunk_overlap=256,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def semantic_chunking_strategie(docs, embeddings):
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def ingest_docs(clear_db=False, strategy="recursive"):
    if clear_db:
        reset_db()
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vector_store = Chroma(
        collection_name=strategy+"_"+COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    if strategy == "md":
        docs = load_documents("md")
    else:
        docs = load_documents("pdf")
    assert len(docs) > 0, "No documents loaded"
    print(f"Loaded {len(docs)} documents.")

    characters = 0
    for doc in docs:
        characters += len(doc.page_content)
        print(doc.metadata)
    
    print(f"Total characters: {characters}")

    docs_processed = process_documents(docs)

    # Chunking process
    if strategy == "md":
        all_splits = md_chunking_strategie(docs_processed)
    elif strategy == "recursive":
        all_splits = naive_chunking_strategie(docs_processed)
    elif strategy == "semantic":
        all_splits = semantic_chunking_strategie(docs_processed, embeddings)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


    print(f"Split document into {len(all_splits)} sub-documents using {strategy}.")

    # Storing documents
    document_ids = vector_store.add_documents(documents=all_splits)
    print(document_ids[:3])

if __name__ == "__main__":
    from langchain_core.globals import set_debug
    from dotenv import load_dotenv
    set_debug(False)
    load_dotenv()
    for i in ["recursive","md","semantic"]:
        ingest_docs(clear_db=False, strategy=i)
