"""
Lightweight latency instrumentation for the FAQ-Chatbot pipeline.

Usage
-----
    from src.telemetry import timed, Timer

    @timed()                          # decorator
    def my_function():
        ...

    with Timer("my_block"):           # inline context manager
        ...

Environment
-----------
    TELEMETRY_ENABLED=false   — disables all instrumentation (zero overhead).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Callable

_enabled: bool = os.getenv("TELEMETRY_ENABLED", "true").lower() != "false"
_log = logging.getLogger("telemetry")


def timed(label: str | None = None) -> Callable:
    """
    Decorator that logs the wall-clock duration of the wrapped function.

    Args:
        label: Human-readable name for the timing entry.
               Defaults to the function's ``__qualname__``.

    Example::

        @timed("my_stage")
        def heavy_function():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        _label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                _log.info("[LATENCY] %-45s %8.1f ms", _label, elapsed_ms)

        return wrapper
    return decorator


@contextmanager
def Timer(label: str):
    """
    Context manager that logs the wall-clock duration of a code block.

    Args:
        label: Human-readable name for the timing entry.

    Example::

        with Timer("dense_search"):
            results = vectorstore.similarity_search(...)
    """
    if not _enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        _log.info("[LATENCY] %-45s %8.1f ms", label, elapsed_ms)
