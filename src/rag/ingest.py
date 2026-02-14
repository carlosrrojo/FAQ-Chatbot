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
    """
    Extracts metadata from a chunk of text using an LLM.
    Returns a dictionary of metadata.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are an expert at extracting metadata from text for retrieval purposes.
    Analyze the following text and extract:
    1. A short, descriptive summary (max 1 sentence).
    2. 3-5 keywords that capture the main topics.

    Return the result in the following format:
    Summary: <summary>
    Keywords: <keyword1>, <keyword2>, ...

    Text:
    {text}
    """)
    
    chain = prompt | llm
    try:
        response = chain.invoke({"text": content})
        content_lines = response.content.strip().split('\n')
        metadata = {}
        for line in content_lines:
            if line.startswith("Summary:"):
                metadata['summary'] = line.replace("Summary:", "").strip()
            elif line.startswith("Keywords:"):
                metadata['keywords'] = line.replace("Keywords:", "").strip()
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}

def ingest_docs(clear_db=False):
    if clear_db:
        reset_db()

    print(f"Loading documents from {DATA_PATH}...")
    
    # Load all .txt files
    # We assume .txt files are Markdown formatted for structural splitting
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    txt_documents = txt_loader.load()

    # Load all .pdf files
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    all_documents = txt_documents + pdf_documents
    
    if not all_documents:
        print("No documents found.")
        return

    print(f"Loaded {len(all_documents)} documents.")

    # Initialize Embeddings and LLM
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    llm = ChatOllama(model=MODEL_NAME)

    final_chunks = []

    # Process .txt documents (Structural Splitting)
    if txt_documents:
        print("Processing .txt documents with Structural Splitting...")
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        for doc in txt_documents:
            # Split by headers
            md_splits = markdown_splitter.split_text(doc.page_content)
            for split in md_splits:
                # Add back original metadata (source, etc.)
                split.metadata.update(doc.metadata)
                final_chunks.append(split)

    # Process PDF documents (Semantic Splitting)
    if pdf_documents:
        print("Processing .pdf documents with Semantic Splitting...")
        semantic_chunker = SemanticChunker(embedding_function)
        
        # SemanticChunker can process a list of documents directly
        semantic_pdf_chunks = semantic_chunker.split_documents(pdf_documents)
        final_chunks.extend(semantic_pdf_chunks)
        
    print(f"Total chunks after splitting: {len(final_chunks)}")

    # Metadata Extraction
    print("Extracting metadata for each chunk (this may take a while)...")
    for i, chunk in enumerate(final_chunks):
        print(f"Extracting metadata for chunk {i+1}/{len(final_chunks)}...")
        metadata = extract_metadata(chunk.page_content, llm)
        chunk.metadata.update(metadata)
        
    # Store in Chroma
    print("Creating embeddings and storing in ChromaDB...")
    if final_chunks:
        db = Chroma.from_documents(
            documents=final_chunks, 
            embedding=embedding_function, 
            persist_directory=DB_PATH,
            collection_name="langchain" 
        )
        print(f"Successfully ingested {len(final_chunks)} chunks into {DB_PATH}.")
    else:
        print("No chunks to ingest.")

if __name__ == "__main__":
    ingest_docs(clear_db=True)
