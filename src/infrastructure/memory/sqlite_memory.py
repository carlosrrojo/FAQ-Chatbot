import logging
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.domain.ports import IMemoryStore
from src.config import MEMORY_DB_PATH

logger = logging.getLogger(__name__)


class SqliteMemoryAdapter(IMemoryStore):
    """
    Durable conversation memory backed by SQLite.
    Supports per-user erasure for GDPR compliance
    """

    def __init__(self, db_path: str = MEMORY_DB_PATH) -> None:
        self._db_path = db_path
        # check_same_thread=False is required if requests handle concurrently
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._checkpointer.setup()  # Ensures the checkpoints tables are created
        logger.info("SqliteMemoryAdapter initialised at %s", db_path)

    def get_checkpointer(self) -> SqliteSaver:
        return self._checkpointer

    def delete_session(self, sender_id: str) -> None:
        """
        Permanently delete all LangGraph checkpoint rows for the given thread_id.
        FR-PRV-02: Right to erasure. Logs the operation for audit purposes
        without logging the deleted content.
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            # LangGraph SqliteSaver stores checkpoints in 'checkpoints' table
            # with a 'thread_id' column.
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (sender_id,)
            )
            deleted = cursor.rowcount
            conn.commit()
        logger.info(
            "GDPR erasure: deleted %d checkpoint row(s) for sender_id hash=%s",
            deleted, hash(sender_id)   # log the hash, not the ID itself
        )