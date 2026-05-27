# src/transport/health_controller.py
import os
import sqlite3
import requests
import datetime
import logging
from flask import Blueprint, jsonify, current_app
from src.config import MEMORY_DB_PATH, MODEL_NAME

logger = logging.getLogger(__name__)
health_bp = Blueprint("health", __name__)

@health_bp.route("/healthz", methods=["GET"])
def healthz():
    """
    Liveness endpoint. Simple and fast check to verify that the
    application process is running and responsive.
    """
    return jsonify({"status": "healthy"}), 200

@health_bp.route("/readyz", methods=["GET"])
def readyz():
    """
    Readiness endpoint. Verifies that all dependencies (SQLite, ChromaDB,
    Ollama/Gemini, and the Meta API) are healthy and reachable.
    """
    shutdown_mgr = current_app.config.get("SHUTDOWN_MANAGER")
    if shutdown_mgr and shutdown_mgr.is_shutting_down:
        logger.warning("Readiness probe failed: Application is shutting down.")
        return jsonify({
            "status": "shutting_down",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }), 503

    dependencies = {}
    is_ready = True

    # 1. SQLite Check
    try:
        with sqlite3.connect(MEMORY_DB_PATH, timeout=2.0) as conn:
            conn.execute("SELECT 1;")
        dependencies["sqlite"] = {
            "status": "healthy",
            "message": "Connection and verification query successful"
        }
    except Exception as e:
        is_ready = False
        logger.error("Readiness check - SQLite unhealthy: %s", e)
        dependencies["sqlite"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # 2. ChromaDB Check
    try:
        from src.rag.agent import _vectorstore
        # Fetching a single item metadata is extremely fast and verifies read access
        _vectorstore.get(limit=1)
        dependencies["chromadb"] = {
            "status": "healthy",
            "message": "ChromaDB client initialized and query successful"
        }
    except Exception as e:
        is_ready = False
        logger.error("Readiness check - ChromaDB unhealthy: %s", e)
        dependencies["chromadb"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # 3. LLM API Check
    llm_provider = os.getenv("AGENT_LLM_PROVIDER", "ollama")
    if llm_provider == "gemini":
        if os.getenv("GOOGLE_API_KEY"):
            try:
                # Perform a basic check to ensure Gemini API is reachable
                resp = requests.head("https://generativelanguage.googleapis.com", timeout=2.0)
                # Any response indicates the service is reachable and responsive to DNS/HTTP requests
                dependencies["llm"] = {
                    "status": "healthy",
                    "message": "Gemini API reachable"
                }
            except Exception as e:
                is_ready = False
                logger.error("Readiness check - Gemini API unreachable: %s", e)
                dependencies["llm"] = {
                    "status": "unhealthy",
                    "message": f"Gemini API connection error: {e}"
                }
        else:
            is_ready = False
            logger.error("Readiness check - Gemini API unhealthy: GOOGLE_API_KEY is not configured")
            dependencies["llm"] = {
                "status": "unhealthy",
                "message": "GOOGLE_API_KEY is not configured"
            }
    else:
        # Ollama check
        ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
        if not ollama_host.startswith(("http://", "https://")):
            ollama_url = f"http://{ollama_host}"
        else:
            ollama_url = ollama_host
        try:
            resp = requests.get(f"{ollama_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model_exists = any(MODEL_NAME in m or m.startswith(MODEL_NAME) for m in models)
                if model_exists:
                    dependencies["llm"] = {
                        "status": "healthy",
                        "message": f"Ollama service reachable, model '{MODEL_NAME}' available"
                    }
                else:
                    is_ready = False
                    logger.error("Readiness check - Ollama degraded: Model '%s' is not loaded", MODEL_NAME)
                    dependencies["llm"] = {
                        "status": "degraded",
                        "message": f"Ollama service reachable, but model '{MODEL_NAME}' is missing"
                    }
            else:
                is_ready = False
                logger.error("Readiness check - Ollama unhealthy: Service returned status %d", resp.status_code)
                dependencies["llm"] = {
                    "status": "unhealthy",
                    "message": f"Ollama returned status {resp.status_code}"
                }
        except Exception as e:
            is_ready = False
            logger.error("Readiness check - Ollama unhealthy: %s", e)
            dependencies["llm"] = {
                "status": "unhealthy",
                "message": f"Ollama connection error: {e}"
            }

    # 4. Meta API Check
    try:
        # Check connectivity to Meta's servers
        requests.head("https://graph.facebook.com", timeout=2.0)
        dependencies["meta_api"] = {
            "status": "healthy",
            "message": "Meta Graph API reachable"
        }
    except Exception as e:
        is_ready = False
        logger.error("Readiness check - Meta API unhealthy: %s", e)
        dependencies["meta_api"] = {
            "status": "unhealthy",
            "message": f"Meta Graph API connection error: {e}"
        }

    status = "ready" if is_ready else "unhealthy"
    status_code = 200 if is_ready else 503

    return jsonify({
        "status": status,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "dependencies": dependencies
    }), status_code
