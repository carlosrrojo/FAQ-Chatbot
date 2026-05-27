import datetime
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
        # Setup tables
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS session_activity (
                    thread_id TEXT PRIMARY KEY,
                    last_active_at TEXT NOT NULL
                )"""
            )
            conn.commit()

        # check_same_thread=False is required if requests handle concurrently
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._checkpointer.setup()  # Ensures the checkpoints tables are created
        logger.info("SqliteMemoryAdapter initialised at %s", db_path)

    def get_checkpointer(self) -> SqliteSaver:
        return self._checkpointer

    def delete_session(self, sender_id: str) -> None:
        """
        Permanently delete all LangGraph checkpoint, write, and activity rows for the given thread_id.
        FR-PRV-02: Right to erasure. Logs the operation for audit purposes
        without logging the deleted content.
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (sender_id,))
            deleted_checkpoints = cursor.rowcount
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (sender_id,))
            deleted_writes = cursor.rowcount
            cursor.execute("DELETE FROM session_activity WHERE thread_id = ?", (sender_id,))
            deleted_activity = cursor.rowcount
            conn.commit()
        logger.info(
            "GDPR erasure: deleted %d checkpoint, %d write, and %d activity row(s) for sender_id hash=%s",
            deleted_checkpoints, deleted_writes, deleted_activity, hash(sender_id)   # log the hash, not the ID itself
        )

    def touch_session(self, sender_id: str) -> None:
        """Record current timestamp as last activity for the session. (FR-PRV-01)"""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO session_activity (thread_id, last_active_at)
                   VALUES (?, ?)
                   ON CONFLICT(thread_id)
                   DO UPDATE SET last_active_at = excluded.last_active_at""",
                (sender_id, now),
            )
            conn.commit()

    def purge_expired_sessions(self, ttl_days: int) -> int:
        """Delete all sessions inactive for longer than ttl_days. Returns count deleted. (FR-PRV-01)"""
        cutoff = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=ttl_days)
        ).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            # Find expired thread_ids
            cursor.execute(
                "SELECT thread_id FROM session_activity WHERE last_active_at < ?",
                (cutoff,),
            )
            expired = [row[0] for row in cursor.fetchall()]

            if not expired:
                return 0

            placeholders = ",".join("?" * len(expired))
            cursor.execute(f"DELETE FROM writes WHERE thread_id IN ({placeholders})", expired)
            cursor.execute(f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})", expired)
            cursor.execute(f"DELETE FROM session_activity WHERE thread_id IN ({placeholders})", expired)
            conn.commit()

        logger.info(
            "FR-PRV-01 retention purge: deleted %d expired session(s) (TTL=%d days)",
            len(expired), ttl_days,
        )
        return len(expired)