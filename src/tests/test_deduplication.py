# src/tests/test_deduplication.py
"""Unit tests for InMemoryDeduplicationStore (FR-CHN-07)."""
import threading
import time
from unittest.mock import patch

import pytest

from src.infrastructure.deduplication import InMemoryDeduplicationStore


class TestInMemoryDeduplicationStore:
    """Core behaviour of the TTL-based deduplication cache."""

    def test_first_message_is_not_duplicate(self):
        store = InMemoryDeduplicationStore(ttl_seconds=60, max_size=100)
        assert store.is_duplicate("msg_001") is False

    def test_same_id_is_duplicate(self):
        store = InMemoryDeduplicationStore(ttl_seconds=60, max_size=100)
        store.is_duplicate("msg_001")
        assert store.is_duplicate("msg_001") is True

    def test_different_ids_are_independent(self):
        store = InMemoryDeduplicationStore(ttl_seconds=60, max_size=100)
        store.is_duplicate("msg_001")
        assert store.is_duplicate("msg_002") is False

    def test_expired_entry_is_not_duplicate(self):
        store = InMemoryDeduplicationStore(ttl_seconds=1, max_size=100)
        store.is_duplicate("msg_001")

        # Patch time.monotonic to simulate TTL expiry
        original_now = time.monotonic()
        with patch("src.infrastructure.deduplication.time.monotonic",
                   return_value=original_now + 2):
            assert store.is_duplicate("msg_001") is False

    def test_max_size_evicts_oldest(self):
        store = InMemoryDeduplicationStore(ttl_seconds=60, max_size=3)
        store.is_duplicate("msg_001")
        store.is_duplicate("msg_002")
        store.is_duplicate("msg_003")
        # This should evict msg_001 (FIFO)
        store.is_duplicate("msg_004")

        # msg_001 was evicted, so it should no longer be duplicate
        assert store.is_duplicate("msg_001") is False
        # msg_003 should still be tracked (msg_002 was evicted when
        # msg_004 pushed size to 4, then msg_001 re-insert pushed to 4 again)
        assert store.is_duplicate("msg_004") is True

    def test_thread_safety(self):
        """Concurrent access from multiple threads must not raise or corrupt state."""
        store = InMemoryDeduplicationStore(ttl_seconds=60, max_size=10_000)
        errors: list[Exception] = []

        def worker(start_id: int, count: int):
            try:
                for i in range(count):
                    store.is_duplicate(f"msg_{start_id + i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i * 1000, 500))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # Spot-check: IDs that were actually inserted should be duplicates
        assert store.is_duplicate("msg_0") is True
        assert store.is_duplicate("msg_1000") is True
        assert store.is_duplicate("msg_3499") is True
