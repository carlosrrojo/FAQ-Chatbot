import os
import json
from unittest.mock import patch, MagicMock, mock_open
import pytest
from langchain_core.documents import Document

from src.config import get_active_collection_name, ACTIVE_COLLECTION_PATH
from src.rag.ingest import run_ingest

def test_get_active_collection_name_default():
    # If the active collection file does not exist, return default base name
    with patch("os.path.exists", return_value=False):
        name = get_active_collection_name()
        assert "espazo_nature" in name

def test_get_active_collection_name_custom():
    # If the active collection file exists, return the name stored inside it
    mock_data = '{"active_collection": "test_collection_12345"}'
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_data)):
        name = get_active_collection_name()
        assert name == "test_collection_12345"

@patch("src.rag.ingest.glob_module.glob")
@patch("src.rag.ingest.parse_doc")
@patch("src.rag.ingest.MetadataExtractor")
@patch("src.rag.ingest.enrich_document")
@patch("src.rag.ingest.get_embeddings")
@patch("src.rag.ingest.chromadb.PersistentClient")
@patch("src.rag.ingest.Chroma")
@patch("src.rag.ingest.write_manifest")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_run_ingest_success(
    mock_makedirs,
    mock_open_func,
    mock_write_manifest,
    mock_chroma_cls,
    mock_chromadb_client,
    mock_get_embeddings,
    mock_enrich_doc,
    mock_metadata_extractor,
    mock_parse_doc,
    mock_glob
):
    # Setup mocks
    mock_glob.return_value = ["/dummy/path/doc.pdf"]
    mock_parse_doc.return_value = [Document(page_content="Mock content", metadata={"section": "general", "parent_section": "none"})]
    
    # Mock Chroma vector store
    mock_vectorstore = MagicMock()
    mock_vectorstore._collection.count.return_value = 1
    mock_vectorstore.similarity_search.return_value = [Document(page_content="glamping results")]
    mock_chroma_cls.return_value = mock_vectorstore
    
    # Mock chromadb client
    mock_client = MagicMock()
    mock_client.list_collections.return_value = [MagicMock(name="old_col")]
    mock_chromadb_client.return_value = mock_client
    
    # Run ingestion
    run_ingest()
    
    # Verify that documents were added
    mock_vectorstore.add_documents.assert_called_once()
    
    # Verify validation query was run
    mock_vectorstore.similarity_search.assert_called_once_with("glamping", k=1)
    
    # Verify pointer file was updated
    mock_open_func.assert_any_call(ACTIVE_COLLECTION_PATH, "w", encoding="utf-8")
    
    # Verify obsolete collection cleanup was attempted
    mock_client.list_collections.assert_called_once()
    
    # Verify manifest was written
    mock_write_manifest.assert_called_once()

@patch("src.rag.ingest.glob_module.glob")
@patch("src.rag.ingest.parse_doc")
@patch("src.rag.ingest.MetadataExtractor")
@patch("src.rag.ingest.enrich_document")
@patch("src.rag.ingest.get_embeddings")
@patch("src.rag.ingest.chromadb.PersistentClient")
@patch("src.rag.ingest.Chroma")
@patch("src.rag.ingest.write_manifest")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_run_ingest_validation_failure_empty(
    mock_makedirs,
    mock_open_func,
    mock_write_manifest,
    mock_chroma_cls,
    mock_chromadb_client,
    mock_get_embeddings,
    mock_enrich_doc,
    mock_metadata_extractor,
    mock_parse_doc,
    mock_glob
):
    # Setup mocks
    mock_glob.return_value = ["/dummy/path/doc.pdf"]
    mock_parse_doc.return_value = [Document(page_content="Mock content", metadata={"section": "general", "parent_section": "none"})]
    
    # Mock Chroma vector store to return count = 0
    mock_vectorstore = MagicMock()
    mock_vectorstore._collection.count.return_value = 0
    mock_chroma_cls.return_value = mock_vectorstore
    
    # Run ingestion and verify validation fails with ValueError
    with pytest.raises(ValueError, match="Validation failed: Collection .* is empty after ingestion."):
        run_ingest()
        
    # Verify pointer file was NEVER written/updated
    for call in mock_open_func.call_args_list:
        assert call[0][0] != ACTIVE_COLLECTION_PATH

@patch("src.rag.ingest.glob_module.glob")
@patch("src.rag.ingest.parse_doc")
@patch("src.rag.ingest.MetadataExtractor")
@patch("src.rag.ingest.enrich_document")
@patch("src.rag.ingest.get_embeddings")
@patch("src.rag.ingest.chromadb.PersistentClient")
@patch("src.rag.ingest.Chroma")
@patch("src.rag.ingest.write_manifest")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_run_ingest_validation_failure_query(
    mock_makedirs,
    mock_open_func,
    mock_write_manifest,
    mock_chroma_cls,
    mock_chromadb_client,
    mock_get_embeddings,
    mock_enrich_doc,
    mock_metadata_extractor,
    mock_parse_doc,
    mock_glob
):
    # Setup mocks
    mock_glob.return_value = ["/dummy/path/doc.pdf"]
    mock_parse_doc.return_value = [Document(page_content="Mock content", metadata={"section": "general", "parent_section": "none"})]
    
    # Mock Chroma vector store to return valid count but empty query results
    mock_vectorstore = MagicMock()
    mock_vectorstore._collection.count.return_value = 1
    mock_vectorstore.similarity_search.return_value = []
    mock_chroma_cls.return_value = mock_vectorstore
    
    # Run ingestion and verify validation fails with ValueError
    with pytest.raises(ValueError, match="Validation failed: Test query returned no results from .*"):
        run_ingest()
        
    # Verify pointer file was NEVER written/updated
    for call in mock_open_func.call_args_list:
        assert call[0][0] != ACTIVE_COLLECTION_PATH
