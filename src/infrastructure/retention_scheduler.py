"""
Periodic GDPR retention enforcement scheduler. FR-PRV-01.

Runs purge_expired_sessions() once at startup and then every
RETENTION_CHECK_INTERVAL_HOURS hours on a daemon thread.
"""
import logging
import threading

from src.config import SESSION_TTL_DAYS, RETENTION_CHECK_INTERVAL_HOURS
from src.domain.ports import IMemoryStore

logger = logging.getLogger(__name__)


class RetentionScheduler:
    def __init__(self, memory_store: IMemoryStore) -> None:
        self._memory = memory_store
        self._interval = RETENTION_CHECK_INTERVAL_HOURS * 3600
        self._timer: threading.Timer | None = None

    def start(self) -> None:
        """Run an immediate purge, then schedule recurring runs."""
        self._run_purge()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _run_purge(self) -> None:
        try:
            deleted = self._memory.purge_expired_sessions(SESSION_TTL_DAYS)
            logger.info("Retention purge completed: %d session(s) removed.", deleted)
        except Exception:
            logger.exception("Retention purge failed.")
        finally:
            self._schedule_next()

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._interval, self._run_purge)
        self._timer.daemon = True
        self._timer.start()
