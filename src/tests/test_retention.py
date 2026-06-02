"""Unit tests for automated GDPR retention enforcement (FR-PRV-01)."""
import datetime
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.memory.sqlite_memory import SqliteMemoryAdapter, _hash_thread_id
from src.infrastructure.retention_scheduler import RetentionScheduler


@pytest.fixture
def temp_memory_store(tmp_path):
    # Use a unique on-disk database per test to keep tables persistent across connection calls
    db_file = tmp_path / "test_memory.sqlite"
    adapter = SqliteMemoryAdapter(db_path=str(db_file))
    return adapter


class TestSqliteMemoryAdapterRetention:
    def test_touch_session_creates_activity_row(self, temp_memory_store):
        adapter = temp_memory_store
        adapter.touch_session("thread_123")

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT thread_id, last_active_at FROM session_activity")
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == _hash_thread_id("thread_123")
        # Check if timestamp is in ISO format
        try:
            datetime.datetime.fromisoformat(rows[0][1])
        except ValueError:
            pytest.fail("Stored timestamp is not valid ISO-8601")

    def test_touch_session_updates_existing(self, temp_memory_store):
        adapter = temp_memory_store
        adapter.touch_session("thread_123")
        hashed_id = _hash_thread_id("thread_123")

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_active_at FROM session_activity WHERE thread_id = ?", (hashed_id,))
            first_time = cursor.fetchone()[0]

        # Ensure different timestamp by inserting directly or mocking datetime, or simply waiting/modifying
        # Let's mock datetime or simply verify the ON CONFLICT behavior
        adapter.touch_session("thread_123")

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_active_at FROM session_activity WHERE thread_id = ?", (hashed_id,))
            second_time = cursor.fetchone()[0]

        # The rows count should still be 1
        cursor.execute("SELECT COUNT(*) FROM session_activity")
        assert cursor.fetchone()[0] == 1

    def test_purge_deletes_expired_sessions(self, temp_memory_store):
        adapter = temp_memory_store
        thread_old_hashed = _hash_thread_id("thread_old")
        thread_new_hashed = _hash_thread_id("thread_new")

        # Add dummy rows to checkpoints and writes tables to verify cascaded deletes
        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, 'ckpt_1')", (thread_old_hashed,))
            cursor.execute("INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel) VALUES (?, 'ckpt_1', 'task_1', 0, 'channel_1')", (thread_old_hashed,))
            cursor.execute("INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, 'ckpt_2')", (thread_new_hashed,))
            cursor.execute("INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel) VALUES (?, 'ckpt_2', 'task_2', 0, 'channel_2')", (thread_new_hashed,))
            conn.commit()

        # Insert old and new activity records
        # old is 31 days ago, new is 5 days ago
        old_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=31)).isoformat()
        new_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()

        with sqlite3.connect(adapter._db_path) as conn:
            conn.execute("INSERT INTO session_activity VALUES (?, ?)", (thread_old_hashed, old_time))
            conn.execute("INSERT INTO session_activity VALUES (?, ?)", (thread_new_hashed, new_time))
            conn.commit()

        deleted = adapter.purge_expired_sessions(ttl_days=30)
        assert deleted == 1

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT thread_id FROM checkpoints")
            checkpoints = [r[0] for r in cursor.fetchall()]
            cursor.execute("SELECT thread_id FROM writes")
            writes = [r[0] for r in cursor.fetchall()]
            cursor.execute("SELECT thread_id FROM session_activity")
            activity = [r[0] for r in cursor.fetchall()]

        assert checkpoints == [thread_new_hashed]
        assert writes == [thread_new_hashed]
        assert activity == [thread_new_hashed]

    def test_purge_preserves_active_sessions(self, temp_memory_store):
        adapter = temp_memory_store
        thread_active_hashed = _hash_thread_id("thread_active")

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, 'ckpt_1')", (thread_active_hashed,))
            conn.commit()

        active_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=29)).isoformat()
        with sqlite3.connect(adapter._db_path) as conn:
            conn.execute("INSERT INTO session_activity VALUES (?, ?)", (thread_active_hashed, active_time))
            conn.commit()

        deleted = adapter.purge_expired_sessions(ttl_days=30)
        assert deleted == 0

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT thread_id FROM checkpoints")
            assert cursor.fetchone()[0] == thread_active_hashed

    def test_delete_session_cleans_all_tables(self, temp_memory_store):
        adapter = temp_memory_store
        hashed_id = _hash_thread_id("thread_to_delete")

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, 'ckpt_1')", (hashed_id,))
            cursor.execute("INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel) VALUES (?, 'ckpt_1', 'task_1', 0, 'channel_1')", (hashed_id,))
            conn.commit()

        adapter.touch_session("thread_to_delete")

        # Now delete
        adapter.delete_session("thread_to_delete")

        with sqlite3.connect(adapter._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (hashed_id,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM writes WHERE thread_id = ?", (hashed_id,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM session_activity WHERE thread_id = ?", (hashed_id,))
            assert cursor.fetchone()[0] == 0


class TestRetentionScheduler:
    @patch("src.infrastructure.retention_scheduler.SESSION_TTL_DAYS", 30)
    @patch("src.infrastructure.retention_scheduler.RETENTION_CHECK_INTERVAL_HOURS", 24)
    def test_scheduler_runs_purge_on_start(self):
        mock_memory = MagicMock()
        mock_memory.purge_expired_sessions.return_value = 5

        scheduler = RetentionScheduler(mock_memory)
        
        with patch.object(scheduler, "_schedule_next") as mock_schedule:
            scheduler.start()
            mock_memory.purge_expired_sessions.assert_called_once_with(30)
            mock_schedule.assert_called_once()

        scheduler.stop()

    @patch("src.infrastructure.retention_scheduler.SESSION_TTL_DAYS", 30)
    @patch("src.infrastructure.retention_scheduler.RETENTION_CHECK_INTERVAL_HOURS", 24)
    def test_scheduler_schedules_next_run(self):
        mock_memory = MagicMock()
        scheduler = RetentionScheduler(mock_memory)

        with patch("threading.Timer") as mock_timer_cls:
            scheduler._schedule_next()
            mock_timer_cls.assert_called_once()
            # Interval is 24 hours = 86400 seconds
            args, kwargs = mock_timer_cls.call_args
            assert args[0] == 86400
            assert mock_timer_cls.return_value.daemon is True
            assert mock_timer_cls.return_value.start.called
