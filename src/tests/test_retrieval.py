"""
Retrieval smoke tests.
Runs retrieval pipeline verification.
"""
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document
import src.utils as utils
from src.rag.agent import retrieve_documents

def run_smoke_test_interactive():
    benchmark = utils.load_benchmark("entorno.txt")
    for question in benchmark:
        try:
            # invoke using original function
            serialized, docs = retrieve_documents.func(query=question)
            print(f"Q: {question} -> retrieved {len(docs)} documents.")
        except Exception as e:
            print(f"Failed to retrieve for {question}: {e}")

@patch("src.rag.agent._vectorstore")
@patch("src.rag.agent._bm25_index")
@patch("src.rag.agent._metadata_extractor")
@patch("src.rag.agent.rerank")
def test_retrieval_smoke_pipeline(mock_rerank, mock_meta_extractor, mock_bm25, mock_vectorstore):
    # Mock metadata extractor output
    from src.rag.agent import QueryMetadata
    mock_meta = QueryMetadata(finding="none", keywords=["cabana"])
    mock_meta_extractor.invoke.return_value = mock_meta
    
    # Mock vectorstore similarity search
    mock_doc = Document(page_content="Cabaña de ejemplo", metadata={"section": "cabanas"})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.8)]
    
    # Mock BM25 search to return the expected 3-tuple (content, score, metadata)
    mock_bm25.search.return_value = [("Cabaña de ejemplo", 0.9, {"section": "cabanas"})]
    
    # Mock rerank output
    mock_rerank.return_value = ([mock_doc], 0.9)
    
    serialized, artifact = retrieve_documents.func(query="¿Qué es la cabaña?")
    docs = artifact["docs"]
    
    assert len(docs) == 1
    assert docs[0].page_content == "Cabaña de ejemplo"
    assert "Cabaña de ejemplo" in serialized

if __name__ == "__main__":
    run_smoke_test_interactive()