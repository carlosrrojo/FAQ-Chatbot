import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from src.rag.extract_processor import ExtractProcessor
import json

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

def recursive_chunking_strategie(docs):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # chunk size (characters)
            chunk_overlap=256,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
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
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    print(f"Split document into {len(all_splits)} sub-documents using {strategy}.")

    # Metadata extraction with LLM
    print("Extracting keywords for metadata...")
    docs_with_metadata = []
    for doc in all_splits:
        keywords = extract_processor.extract_metadata(doc.page_content)
        siblings = extract_processor.get_siblings(all_splits, doc.metadata["parent_section"])
        doc.page_content = f"""
        [KEYWORDS]
        {" , ".join(keywords)}
        [CONTENT]
        {doc.page_content}
        [REFERENCES]
        {siblings}
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
    ingest_docs(clear_db=False, strategy="recursive")
    #for i in ["md","semantic"]:
    #    ingest_docs(clear_db=True, strategy=i)
