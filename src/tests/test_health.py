# src/tests/test_health.py
import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment variables for testing before importing
os.environ["WEBHOOK_VERIFY_TOKEN"] = "MySecretToken"
os.environ["WA_PHONE_NUMBER_ID"] = "mock_id"
os.environ["META_ACCESS_TOKEN"] = "mock_token"
os.environ["INSTAGRAM_ACCESS_TOKEN"] = "mock_token"
os.environ["GOOGLE_API_KEY"] = "mock_key"
os.environ["AGENT_LLM_PROVIDER"] = "ollama"

from src.transport.app import create_app

@pytest.fixture
def app():
    # Patch adapters and background systems so they do not run real logic or start background threads
    with patch("src.transport.app.SqliteMemoryAdapter"), \
         patch("src.transport.app.WhatsAppClient"), \
         patch("src.transport.app.InstagramClient"), \
         patch("src.transport.app.RAGOrchestrator"), \
         patch("src.transport.app.InMemoryDeduplicationStore"), \
         patch("src.transport.app.RetentionScheduler"), \
         patch("src.transport.app.start_watcher"):
        
        app = create_app()
        app.config.update({
            "TESTING": True,
            "SHUTDOWN_MANAGER": MagicMock(),
        })
        # Set is_shutting_down to False by default
        app.config["SHUTDOWN_MANAGER"].is_shutting_down = False
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

def test_healthz_endpoint(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json == {"status": "healthy"}

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_all_healthy_ollama(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    # Mock SQLite success
    mock_conn = MagicMock()
    mock_sqlite_connect.return_value.__enter__.return_value = mock_conn

    # Mock ChromaDB success
    mock_vectorstore.get.return_value = {"ids": ["chunk_1"], "metadatas": [{"section": "cabanas"}]}

    # Mock Ollama success (requests.get for tags, requests.head for meta)
    mock_resp_ollama = MagicMock()
    mock_resp_ollama.status_code = 200
    mock_resp_ollama.json.return_value = {
        "models": [{"name": "llama3.1:latest"}]
    }
    
    mock_resp_meta = MagicMock()
    mock_resp_meta.status_code = 200

    def mock_requests_side_effect(method, url, *args, **kwargs):
        if "tags" in url:
            return mock_resp_ollama
        elif "facebook.com" in url:
            return mock_resp_meta
        return MagicMock()

    mock_requests.get.side_effect = lambda url, **kwargs: mock_requests_side_effect("GET", url, **kwargs)
    mock_requests.head.side_effect = lambda url, **kwargs: mock_requests_side_effect("HEAD", url, **kwargs)

    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json
    assert data["status"] == "ready"
    assert data["dependencies"]["sqlite"]["status"] == "healthy"
    assert data["dependencies"]["chromadb"]["status"] == "healthy"
    assert data["dependencies"]["llm"]["status"] == "healthy"
    assert data["dependencies"]["meta_api"]["status"] == "healthy"

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_sqlite_unhealthy(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    # SQLite fails
    mock_sqlite_connect.side_effect = Exception("SQLite locked")

    # Mock other dependencies as healthy
    mock_vectorstore.get.return_value = {}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.1"}]}
    mock_requests.get.return_value = mock_resp
    mock_requests.head.return_value = MagicMock(status_code=200)

    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "unhealthy"
    assert data["dependencies"]["sqlite"]["status"] == "unhealthy"
    assert "SQLite locked" in data["dependencies"]["sqlite"]["message"]
    assert data["dependencies"]["chromadb"]["status"] == "healthy"

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_chromadb_unhealthy(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    # SQLite healthy
    mock_sqlite_connect.return_value.__enter__.return_value = MagicMock()
    # ChromaDB fails
    mock_vectorstore.get.side_effect = Exception("Chroma index corrupted")

    # Mock other dependencies as healthy
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.1"}]}
    mock_requests.get.return_value = mock_resp
    mock_requests.head.return_value = MagicMock(status_code=200)

    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "unhealthy"
    assert data["dependencies"]["chromadb"]["status"] == "unhealthy"
    assert "Chroma index corrupted" in data["dependencies"]["chromadb"]["message"]

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_ollama_unhealthy_missing_model(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    mock_sqlite_connect.return_value.__enter__.return_value = MagicMock()
    mock_vectorstore.get.return_value = {}
    mock_requests.head.return_value = MagicMock(status_code=200)

    # Ollama is reachable, but returning a model list without llama3.1
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "mistral"}]}
    mock_requests.get.return_value = mock_resp

    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "unhealthy"
    assert data["dependencies"]["llm"]["status"] == "degraded"
    assert "missing" in data["dependencies"]["llm"]["message"]

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_ollama_unhealthy_api_error(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    mock_sqlite_connect.return_value.__enter__.return_value = MagicMock()
    mock_vectorstore.get.return_value = {}
    mock_requests.head.return_value = MagicMock(status_code=200)

    # Ollama API is down / times out
    mock_requests.get.side_effect = Exception("Connection timed out")

    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "unhealthy"
    assert data["dependencies"]["llm"]["status"] == "unhealthy"
    assert "Connection timed out" in data["dependencies"]["llm"]["message"]

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_meta_unhealthy(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    mock_sqlite_connect.return_value.__enter__.return_value = MagicMock()
    mock_vectorstore.get.return_value = {}
    
    # Ollama is healthy
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.1"}]}
    mock_requests.get.return_value = mock_resp
    
    # Meta is unreachable
    mock_requests.head.side_effect = Exception("Meta server down")

    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "unhealthy"
    assert data["dependencies"]["meta_api"]["status"] == "unhealthy"
    assert "Meta server down" in data["dependencies"]["meta_api"]["message"]

def test_readyz_during_shutdown(client, app):
    # Configure ShutdownManager to report shutting down
    app.config["SHUTDOWN_MANAGER"].is_shutting_down = True

    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json["status"] == "shutting_down"

@patch("src.transport.health_controller.sqlite3.connect")
@patch("src.transport.health_controller.requests")
@patch("src.rag.agent._vectorstore")
def test_readyz_gemini_healthy(mock_vectorstore, mock_requests, mock_sqlite_connect, client, app):
    # Set provider to gemini
    with patch.dict(os.environ, {"AGENT_LLM_PROVIDER": "gemini", "GOOGLE_API_KEY": "somekey"}):
        mock_sqlite_connect.return_value.__enter__.return_value = MagicMock()
        mock_vectorstore.get.return_value = {}
        mock_requests.head.return_value = MagicMock(status_code=200)

        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json
        assert data["dependencies"]["llm"]["status"] == "healthy"
        assert "Gemini" in data["dependencies"]["llm"]["message"]
