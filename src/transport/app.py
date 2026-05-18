# src/transport/app.py
import logging
import os
from flask import Flask
from src.config import DATA_PATH
from src.infrastructure.memory.sqlite_memory import SqliteMemoryAdapter
from src.infrastructure.channels.whatsapp_client import WhatsAppClient
from src.infrastructure.channels.instagram_client import InstagramClient
from src.domain.orchestrator import RAGOrchestrator
from src.transport.webhook_controller import webhook_bp
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.rag.watcher import start_watcher

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Flask application factory.
    Wires all infrastructure adapters and domain services,
    then registers the webhook blueprint.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = Flask(__name__)

    # Infrastructure
    memory = SqliteMemoryAdapter()
    wa_client = WhatsAppClient()
    ig_client = InstagramClient()

    # Domain
    orchestrator = RAGOrchestrator(memory_store=memory)

    # Register blueprint, passing dependencies via app config
    app.config["ORCHESTRATOR"] = orchestrator
    app.config["WA_CLIENT"] = wa_client
    app.config["IG_CLIENT"] = ig_client

    
    app.register_blueprint(webhook_bp)

    # Start file watcher (FR-ING-05)
    # The retriever singleton is accessible via the orchestrator's internal agent
    # Watcher callback will call retriever.rebuild_index()
    abs_data_path = os.path.abspath(DATA_PATH)
    os.makedirs(abs_data_path, exist_ok=True)
    start_watcher(
        on_reingest_callback=orchestrator._retriever.rebuild_index if hasattr(orchestrator, '_retriever') else None,
        path=abs_data_path
    )

    logger.info("FAQ-Chatbot application started.")
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False)