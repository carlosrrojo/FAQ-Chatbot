import datetime
import logging
import sqlite3
import hmac
import hashlib
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from src.domain.ports import IMemoryStore
from src.config import MEMORY_DB_PATH

logger = logging.getLogger(__name__)


def _hash_thread_id(thread_id: str) -> str:
    secret = os.getenv("SESSION_HMAC_KEY") or os.getenv("META_APP_SECRET") or "default_fallback_secret_key_for_dev"
    return hmac.new(secret.encode("utf-8"), thread_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _hash_config(config: dict) -> dict:
    if not config or "configurable" not in config:
        return config
    # Avoid mutating the original config dictionary. Make a shallow copy.
    config_copy = dict(config)
    config_copy["configurable"] = dict(config["configurable"])
    thread_id = config_copy["configurable"].get("thread_id")
    if thread_id is not None:
        config_copy["configurable"]["thread_id"] = _hash_thread_id(thread_id)
    return config_copy


class HashingSqliteSaver(SqliteSaver):
    """
    A LangGraph-compatible checkpointer that wraps SqliteSaver to hash the
    thread_id in the config before any DB operations, ensuring minimisation at rest.
    """
    def get(self, config):
        return super().get(_hash_config(config))

    def get_tuple(self, config):
        return super().get_tuple(_hash_config(config))

    def list(self, config, *, filter = None, before = None, limit = None):
        return super().list(
            _hash_config(config) if config is not None else None,
            filter=filter,
            before=_hash_config(before) if before is not None else None,
            limit=limit
        )

    def put(self, config, checkpoint, metadata, new_versions):
        hashed_config = _hash_config(config)
        res_config = super().put(hashed_config, checkpoint, metadata, new_versions)
        # Restore the original thread_id in the returned config so that callers
        # continue to see the plaintext thread_id in memory.
        if config and "configurable" in config and "thread_id" in config["configurable"]:
            res_config = dict(res_config)
            res_config["configurable"] = dict(res_config["configurable"])
            res_config["configurable"]["thread_id"] = config["configurable"]["thread_id"]
        return res_config

    def put_writes(self, config, writes, task_id, task_path = ''):
        return super().put_writes(_hash_config(config), writes, task_id, task_path=task_path)

    def delete_thread(self, thread_id: str) -> None:
        return super().delete_thread(_hash_thread_id(thread_id))

    # Async methods
    async def aget(self, config):
        return await super().aget(_hash_config(config))

    async def aget_tuple(self, config):
        return await super().aget_tuple(_hash_config(config))

    def alist(self, config, *, filter = None, before = None, limit = None):
        return super().alist(
            _hash_config(config) if config is not None else None,
            filter=filter,
            before=_hash_config(before) if before is not None else None,
            limit=limit
        )

    async def aput(self, config, checkpoint, metadata, new_versions):
        hashed_config = _hash_config(config)
        res_config = await super().aput(hashed_config, checkpoint, metadata, new_versions)
        if config and "configurable" in config and "thread_id" in config["configurable"]:
            res_config = dict(res_config)
            res_config["configurable"] = dict(res_config["configurable"])
            res_config["configurable"]["thread_id"] = config["configurable"]["thread_id"]
        return res_config

    async def aput_writes(self, config, writes, task_id, task_path = ''):
        return await super().aput_writes(_hash_config(config), writes, task_id, task_path=task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return await super().adelete_thread(_hash_thread_id(thread_id))


class SqliteMemoryAdapter(IMemoryStore):
    """
    Durable conversation memory backed by SQLite.
    Supports per-user erasure for GDPR compliance.
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
        self._checkpointer = HashingSqliteSaver(self._conn)
        self._checkpointer.setup()  # Ensures the checkpoints tables are created
        logger.info("SqliteMemoryAdapter initialised at %s", db_path)

    def get_checkpointer(self) -> HashingSqliteSaver:
        return self._checkpointer

    def delete_session(self, sender_id: str) -> None:
        """
        Permanently delete all LangGraph checkpoint, write, and activity rows for the given thread_id.
        FR-PRV-02: Right to erasure. Logs the operation for audit purposes
        without logging the deleted content.
        """
        hashed_id = _hash_thread_id(sender_id)
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (hashed_id,))
            deleted_checkpoints = cursor.rowcount
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (hashed_id,))
            deleted_writes = cursor.rowcount
            cursor.execute("DELETE FROM session_activity WHERE thread_id = ?", (hashed_id,))
            deleted_activity = cursor.rowcount
            conn.commit()
        logger.info(
            "GDPR erasure: deleted %d checkpoint, %d write, and %d activity row(s) for sender_id hash=%s",
            deleted_checkpoints, deleted_writes, deleted_activity, hash(sender_id)   # log the process hash, not the ID itself
        )

    def touch_session(self, sender_id: str) -> None:
        """Record current timestamp as last activity for the session. (FR-PRV-01)"""
        hashed_id = _hash_thread_id(sender_id)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO session_activity (thread_id, last_active_at)
                   VALUES (?, ?)
                   ON CONFLICT(thread_id)
                   DO UPDATE SET last_active_at = excluded.last_active_at""",
                (hashed_id, now),
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
            "GDPR retention purge: deleted %d expired session(s) (TTL=%d days)",
            len(expired), ttl_days,
        )
        return len(expired)