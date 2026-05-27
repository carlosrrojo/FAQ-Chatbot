"""
Unit tests for src.telemetry — no heavy RAG dependencies required.
"""

import logging
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_telemetry(enabled: bool = True):
    """Re-import telemetry with the env-var set to the desired value."""
    import importlib
    import src.telemetry as tel_mod
    os.environ["TELEMETRY_ENABLED"] = "true" if enabled else "false"
    # Force re-evaluation of the module-level _enabled flag
    importlib.reload(tel_mod)
    return tel_mod


# ---------------------------------------------------------------------------
# @timed decorator
# ---------------------------------------------------------------------------

class TestTimedDecorator:
    def test_return_value_preserved(self):
        from src.telemetry import timed

        @timed()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_custom_label_preserved(self):
        from src.telemetry import timed

        @timed("my_stage")
        def identity(x):
            return x

        assert identity(42) == 42

    def test_logs_latency_line(self, caplog):
        tel = _import_telemetry(enabled=True)

        @tel.timed("test_fn")
        def noop():
            return "ok"

        with caplog.at_level(logging.INFO, logger="telemetry"):
            result = noop()

        assert result == "ok"
        assert any("[LATENCY]" in r.message for r in caplog.records)

    def test_disabled_emits_no_log(self, caplog):
        tel = _import_telemetry(enabled=False)

        @tel.timed("silent")
        def noop():
            return "ok"

        with caplog.at_level(logging.INFO, logger="telemetry"):
            result = noop()

        assert result == "ok"
        assert not any("[LATENCY]" in r.message for r in caplog.records)

    def test_exception_still_propagates(self):
        from src.telemetry import timed

        @timed()
        def boom():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            boom()


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class TestTimer:
    def test_no_exception_on_clean_block(self):
        from src.telemetry import Timer

        with Timer("clean_block"):
            x = 1 + 1
        assert x == 2

    def test_logs_latency_line(self, caplog):
        tel = _import_telemetry(enabled=True)

        with caplog.at_level(logging.INFO, logger="telemetry"):
            with tel.Timer("ctx_block"):
                pass

        assert any("[LATENCY]" in r.message for r in caplog.records)

    def test_disabled_emits_no_log(self, caplog):
        tel = _import_telemetry(enabled=False)

        with caplog.at_level(logging.INFO, logger="telemetry"):
            with tel.Timer("silent_ctx"):
                pass

        assert not any("[LATENCY]" in r.message for r in caplog.records)

    def test_exception_inside_block_propagates(self):
        from src.telemetry import Timer

        with pytest.raises(RuntimeError, match="ctx error"):
            with Timer("failing_block"):
                raise RuntimeError("ctx error")
