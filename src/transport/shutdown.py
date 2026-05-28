# src/transport/shutdown.py
import logging
import signal
import sys
import threading
import concurrent.futures
from typing import Callable, Any

logger = logging.getLogger(__name__)


class ShutdownManager:
    """
    Coordinates the graceful shutdown sequence for the Flask application.
    Manages background execution via a thread pool, stops schedulers/watchers,
    drains in-flight webhook tasks, and handles termination signals.
    """

    def __init__(self, max_workers: int = 10, max_queue_depth: int = 20, exit_fn: Callable[[int], None] = sys.exit) -> None:
        self.is_shutting_down = False
        self.max_workers = max_workers
        self.max_queue_depth = max_queue_depth
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="webhook_worker",
        )
        self._futures: set[concurrent.futures.Future] = set()
        self._lock = threading.RLock()

        # Resources registered for graceful shutdown
        self._observer: Any = None
        self._retention: Any = None
        self._exit_fn = exit_fn

    def register_resources(self, observer: Any = None, retention: Any = None) -> None:
        """Register watchers and schedulers so they can be stopped during shutdown."""
        self._observer = observer
        self._retention = retention

    def submit_task(self, fn: Callable[..., None], *args: Any, **kwargs: Any) -> concurrent.futures.Future | None:
        """
        Submit a task for background execution. If shutdown is in progress,
        or the executor is full (capacity limit reached), the task is rejected
        and None is returned.
        """
        with self._lock:
            if self.is_shutting_down:
                logger.warning("Rejecting background task: shutdown in progress.")
                return None

            if len(self._futures) >= self.max_workers + self.max_queue_depth:
                logger.warning(
                    "Rejecting background task: thread pool is at capacity (max_workers=%d, max_queue_depth=%d).",
                    self.max_workers,
                    self.max_queue_depth,
                )
                return None

            future = self._executor.submit(fn, *args, **kwargs)
            self._futures.add(future)

        # Define cleanup closure to discard the future once done
        def _cleanup(f: concurrent.futures.Future) -> None:
            with self._lock:
                self._futures.discard(f)

        future.add_done_callback(_cleanup)
        return future

    def initiate_shutdown(self, signum: int | None = None) -> None:
        """
        Coordinated graceful shutdown sequence. Stops watchers/schedulers,
        drains in-flight request tasks within a 10-second window,
        and shuts down the executor.
        """
        with self._lock:
            if self.is_shutting_down:
                return
            self.is_shutting_down = True

        logger.info("Graceful shutdown sequence initiated (Signal: %s).", signum)

        # 1. Stop background activity immediately to prevent new work or re-ingestion
        if self._observer:
            logger.info("Stopping watchdog file observer...")
            try:
                self._observer.stop()
            except Exception:
                logger.exception("Error stopping watchdog observer.")

        if self._retention:
            logger.info("Stopping GDPR retention scheduler...")
            try:
                self._retention.stop()
            except Exception:
                logger.exception("Error stopping retention scheduler.")

        # 2. Wait for watchdog threads to join
        if self._observer:
            logger.info("Joining watchdog file observer...")
            try:
                self._observer.join(timeout=3.0)
            except Exception:
                logger.exception("Error joining watchdog observer.")

        # 3. Drain in-flight webhook processing tasks
        with self._lock:
            active_futures = list(self._futures)

        if active_futures:
            logger.info("Draining %d in-flight webhook task(s)...", len(active_futures))
            done, not_done = concurrent.futures.wait(active_futures, timeout=10.0)
            if not_done:
                logger.warning(
                    "%d task(s) did not complete within the 10s grace period and will be cut off.",
                    len(not_done),
                )
            else:
                logger.info("All in-flight webhook tasks completed successfully.")

        # 4. Final shutdown of thread executor
        self._executor.shutdown(wait=True)
        logger.info("Graceful shutdown sequence completed.")

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Signal handler callback for SIGTERM and SIGINT."""
        logger.info("Received signal %d. Coordinating graceful shutdown...", signum)
        self.initiate_shutdown(signum=signum)
        self._exit_fn(0)
