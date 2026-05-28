"""
metadata_cache.py
-----------------
Persistent SQLite-backed memoisation cache for LLM metadata extraction.

Ensures that each unique (content, section, parent_section, extraction_version)
tuple is sent to the LLM at most once.  Subsequent ingestion runs read the
frozen result from the cache, making chunk ids and metadata values fully
deterministic.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class MetadataCache:
    """
    SQLite-backed metadata cache with WAL mode.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    _CREATE_TABLE_SQL = """\
        CREATE TABLE IF NOT EXISTS metadata_cache (
            cache_key     TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL,
            model_name    TEXT NOT NULL,
            created_at    TEXT NOT NULL
        );
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._CREATE_TABLE_SQL)
        self._conn.commit()
        logger.debug("MetadataCache opened at %s", db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, cache_key: str) -> Optional[dict]:
        """Return the cached metadata dict on hit, ``None`` on miss."""
        cursor = self._conn.execute(
            "SELECT metadata_json FROM metadata_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            logger.debug("Cache MISS for key %.12s…", cache_key)
            return None
        logger.debug("Cache HIT for key %.12s…", cache_key)
        return json.loads(row[0])

    def set(
        self, cache_key: str, metadata: dict, model_name: str
    ) -> None:
        """Upsert a metadata entry and commit immediately."""
        self._conn.execute(
            """INSERT OR REPLACE INTO metadata_cache
               (cache_key, metadata_json, model_name, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                cache_key,
                json.dumps(metadata, ensure_ascii=False),
                model_name,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        logger.debug("Cache SET for key %.12s…", cache_key)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
        logger.debug("MetadataCache closed (%s)", self._db_path)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MetadataCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
