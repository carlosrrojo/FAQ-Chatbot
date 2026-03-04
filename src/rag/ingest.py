import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from extract_processor import ExtractProcessor
import json

# Configuration
DATA_PATH = "data/documents"
DB_PATH = "data/chroma_db"
MODEL_NAME = "llama3.1"
COLLECTION_NAME = "espazo_nature"


# Reset database - Erases the collection
def reset_db(strategy: str):
    print(f"Resetting database at {DB_PATH} (clearing collection '{strategy}_{COLLECTION_NAME}')...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(strategy+"_"+COLLECTION_NAME)
        print(f"Deleted existing collection '{strategy}_{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' could not be deleted (might not exist): {e}")

def load_documents(format: str = "txt") -> list[Document]:
    print(f"Loading documents from {DATA_PATH}...")
    # Loading docs
    docs = []
    for file in os.listdir(DATA_PATH):
        if format=="txt" and file.endswith("txt"):
            docs.append(os.path.join(DATA_PATH, file))
        elif format=="pdf" and file.endswith("pdf"):
            docs.append(os.path.join(DATA_PATH, file))
        elif format=="md" and file.endswith("md"):
            docs.append(os.path.join(DATA_PATH, file)) # Using TextLoader instead of undefined MarkdownLoader
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

def recursive_chunking_strategie(docs):
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
        reset_db(strategy)

    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vector_store = Chroma(
        collection_name=strategy+"_"+COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    llm = ChatOllama(model=MODEL_NAME)
    extract_processor = ExtractProcessor(llm)

    if strategy == "md":
        docs = load_documents("md")
    else:
        docs = load_documents("pdf")
    assert len(docs) > 0, "No documents loaded"
    print(f"Loaded {len(docs)} documents.")

    # Divide documents into sections by headings
    print(f"Dividing documents into sections...")
    docs_processed = []
    for doc in docs:
        docs_processed.extend(extract_processor.process_document(doc))
    print(f"Processed documents into {len(docs_processed)} sections.")
    for doc in docs_processed:
        print(doc.metadata['section'])

    # Chunking process
    if strategy == "md":
        all_splits = md_chunking_strategie(docs_processed)
    elif strategy == "recursive":
        all_splits = recursive_chunking_strategie(docs_processed)
    elif strategy == "semantic":
        all_splits = semantic_chunking_strategie(docs_processed, embeddings)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    print(f"Split document into {len(all_splits)} sub-documents using {strategy}.")

    # Metadata extraction with LLM
    print("Extracting keywords for metadata...")
    docs_with_metadata = []
    for doc in all_splits:
        keywords = extract_processor.extract_metadata(doc.page_content)
        doc.page_content = f"""
        [KEYWORDS]
        {" , ".join(keywords)}
        [CONTENT]
        {doc.page_content}
        """
        doc.metadata.update({"keywords": json.dumps(keywords)})
        docs_with_metadata.append(doc)
    
    print("Added metadata to documents.")

    # Storing documents
    print("Storing documents in vector store...")
    document_ids = vector_store.add_documents(documents=docs_with_metadata)
    print(f"Stored {len(document_ids)} documents.")

if __name__ == "__main__":
    from langchain_core.globals import set_debug
    from dotenv import load_dotenv
    set_debug(False)
    load_dotenv()
    ingest_docs(clear_db=True, strategy="recursive")
    #for i in ["md","semantic"]:
    #    ingest_docs(clear_db=True, strategy=i)