# src/infrastructure/deduplication.py
"""
In-memory message deduplication store with automatic TTL expiry.

Thread-safe implementation using a threading.Lock around an
OrderedDict for O(1) insert/lookup and FIFO eviction.  FR-CHN-07.
"""
import logging
import threading
import time
from collections import OrderedDict

from src.config import DEDUP_TTL_SECONDS, DEDUP_MAX_SIZE
from src.domain.ports import IDeduplicationStore

logger = logging.getLogger(__name__)


class InMemoryDeduplicationStore(IDeduplicationStore):
    """
    Tracks recently-seen message IDs in an OrderedDict with timestamps.
    Expired entries are lazily purged on each call to ``is_duplicate()``.
    """

    def __init__(
        self,
        ttl_seconds: int = DEDUP_TTL_SECONDS,
        max_size: int = DEDUP_MAX_SIZE,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def is_duplicate(self, message_id: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)

            if message_id in self._seen:
                logger.info("Duplicate message suppressed: %s", message_id)
                return True

            # Record as seen
            self._seen[message_id] = now

            # Enforce size cap (FIFO eviction)
            while len(self._seen) > self._max_size:
                self._seen.popitem(last=False)

            return False

    def _purge_expired(self, now: float) -> None:
        """Remove entries older than TTL from the front of the OrderedDict."""
        while self._seen:
            oldest_key, oldest_time = next(iter(self._seen.items()))
            if now - oldest_time > self._ttl:
                self._seen.popitem(last=False)
            else:
                break
