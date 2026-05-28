# src/transport/app.py
import logging
import os
from flask import Flask
from src.config import DATA_PATH, CONCURRENT_WORKERS, MAX_QUEUE_DEPTH, SYSTEM_PROMPT_PATH
from src.infrastructure.memory.sqlite_memory import SqliteMemoryAdapter
from src.infrastructure.channels.whatsapp_client import WhatsAppClient
from src.infrastructure.channels.instagram_client import InstagramClient
from src.infrastructure.deduplication import InMemoryDeduplicationStore
from src.domain.orchestrator import RAGOrchestrator
from src.transport.webhook_controller import webhook_bp
from src.transport.health_controller import health_bp
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.rag.watcher import start_watcher
from src.rag.agent import rebuild_bm25
from src.infrastructure.retention_scheduler import RetentionScheduler
from src.transport.shutdown import ShutdownManager
from src.utils import load_prompt

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
    shutdown_mgr = ShutdownManager(
        max_workers=CONCURRENT_WORKERS,
        max_queue_depth=MAX_QUEUE_DEPTH,
    )
    memory = SqliteMemoryAdapter()
    wa_client = WhatsAppClient()
    ig_client = InstagramClient()
    dedup_store = InMemoryDeduplicationStore()

    # Load system prompt template
    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)

    # Domain
    orchestrator = RAGOrchestrator(
        memory_store=memory,
        system_prompt_template=system_prompt,
    )

    # GDPR retention enforcement (FR-PRV-01)
    retention = RetentionScheduler(memory)
    retention.start()

    # Register blueprint, passing dependencies via app config
    app.config["ORCHESTRATOR"] = orchestrator
    app.config["WA_CLIENT"] = wa_client
    app.config["IG_CLIENT"] = ig_client
    app.config["DEDUP_STORE"] = dedup_store
    app.config["RETENTION_SCHEDULER"] = retention
    app.config["SHUTDOWN_MANAGER"] = shutdown_mgr

    app.register_blueprint(webhook_bp)
    app.register_blueprint(health_bp)

    # Start file watcher (FR-ING-05)
    # After re-ingestion, rebuild the BM25 sparse index to stay in sync
    # with the updated ChromaDB collection (FR-RET-02 criterion 3).
    abs_data_path = os.path.abspath(DATA_PATH)
    os.makedirs(abs_data_path, exist_ok=True)
    observer = start_watcher(
        on_reingest_callback=rebuild_bm25,
        path=abs_data_path
    )

    # Register background resources to enable graceful teardown
    shutdown_mgr.register_resources(observer=observer, retention=retention)

    # Try registering signal handlers for graceful shutdown (signal only works in main thread)
    try:
        import signal
        signal.signal(signal.SIGTERM, shutdown_mgr.handle_signal)
        signal.signal(signal.SIGINT, shutdown_mgr.handle_signal)
        logger.info("Signal handlers registered for SIGTERM and SIGINT.")
    except ValueError:
        logger.warning("Could not register signal handlers (must run in main thread).")

    logger.info("FAQ-Chatbot application started.")
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False)