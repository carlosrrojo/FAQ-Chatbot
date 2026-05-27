# src/tests/test_shutdown.py
import time
from unittest.mock import MagicMock, patch
import pytest
from flask import Flask
from src.transport.shutdown import ShutdownManager
from src.transport.webhook_controller import webhook_bp


def test_shutdown_manager_lifecycle():
    """Verify ShutdownManager registers resources and stops them on shutdown."""
    print("\n[DEBUG] test_shutdown_manager_lifecycle starting")
    mock_observer = MagicMock()
    mock_retention = MagicMock()
    
    manager = ShutdownManager(exit_fn=MagicMock())
    try:
        manager.register_resources(observer=mock_observer, retention=mock_retention)
        assert manager.is_shutting_down is False
        
        manager.initiate_shutdown()
        assert manager.is_shutting_down is True
        
        mock_observer.stop.assert_called_once()
        mock_retention.stop.assert_called_once()
        mock_observer.join.assert_called_once_with(timeout=3.0)
    finally:
        print("[DEBUG] test_shutdown_manager_lifecycle cleanup")
        manager.initiate_shutdown()
    print("[DEBUG] test_shutdown_manager_lifecycle finished")


def test_shutdown_manager_submit_task():
    """Verify that submit_task submits tasks and discards completed futures."""
    print("\n[DEBUG] test_shutdown_manager_submit_task starting")
    manager = ShutdownManager(exit_fn=MagicMock())
    try:
        task_run = False
        def sample_task():
            nonlocal task_run
            print("[DEBUG] sample_task running")
            task_run = True

        print("[DEBUG] Submitting task...")
        future = manager.submit_task(sample_task)
        assert future is not None
        
        # Wait for the future to finish
        print("[DEBUG] Waiting for future result...")
        future.result(timeout=2.0)
        print("[DEBUG] Future result resolved")
        assert task_run is True
        
        # Wait for done callback to clean up the future from active set
        print("[DEBUG] Waiting for done callback cleanup...")
        for i in range(20):
            if future not in manager._futures:
                print(f"[DEBUG] Future removed after {i*0.05}s")
                break
            time.sleep(0.05)
            
        assert future not in manager._futures
        
        # Stop/Shutdown the manager
        print("[DEBUG] Initiating manager shutdown...")
        manager.initiate_shutdown()
        print("[DEBUG] Manager shutdown complete")
        
        # New tasks should be rejected
        print("[DEBUG] Submitting task during shutdown...")
        rejected_future = manager.submit_task(sample_task)
        assert rejected_future is None
    finally:
        print("[DEBUG] test_shutdown_manager_submit_task cleanup")
        manager.initiate_shutdown()
    print("[DEBUG] test_shutdown_manager_submit_task finished")


def test_shutdown_signal_handler():
    """Verify that signal handler triggers shutdown and calls the exit function."""
    print("\n[DEBUG] test_shutdown_signal_handler starting")
    mock_exit = MagicMock()
    manager = ShutdownManager(exit_fn=mock_exit)
    try:
        manager.handle_signal(signum=15, frame=None)
        assert manager.is_shutting_down is True
        mock_exit.assert_called_once_with(0)
    finally:
        print("[DEBUG] test_shutdown_signal_handler cleanup")
        manager.initiate_shutdown()
    print("[DEBUG] test_shutdown_signal_handler finished")


def test_webhook_rejection_on_shutdown():
    """Verify that webhook controller returns 503 during graceful shutdown."""
    print("\n[DEBUG] test_webhook_rejection_on_shutdown starting")
    app = Flask(__name__)
    app.register_blueprint(webhook_bp)
    
    manager = ShutdownManager(exit_fn=MagicMock())
    manager.is_shutting_down = True
    app.config["SHUTDOWN_MANAGER"] = manager
    
    try:
        client = app.test_client()
        with patch("src.transport.webhook_controller._verify_signature", return_value=True):
            response = client.post("/webhook", json={"object": "whatsapp_business_account"})
            assert response.status_code == 503
            assert response.json == {"error": "Service Temporarily Unavailable"}
    finally:
        print("[DEBUG] test_webhook_rejection_on_shutdown cleanup")
        manager.initiate_shutdown()
    print("[DEBUG] test_webhook_rejection_on_shutdown finished")


def test_webhook_drain_on_shutdown():
    """Verify that initiate_shutdown blocks to drain in-flight tasks."""
    print("\n[DEBUG] test_webhook_drain_on_shutdown starting")
    manager = ShutdownManager(exit_fn=MagicMock())
    try:
        task_completed = False
        def long_task():
            nonlocal task_completed
            print("[DEBUG] long_task running")
            time.sleep(0.3)
            task_completed = True
            print("[DEBUG] long_task completed")
            
        print("[DEBUG] Submitting long task...")
        future = manager.submit_task(long_task)
        assert future is not None
        assert future in manager._futures
        
        print("[DEBUG] Initiating shutdown (draining)...")
        start_time = time.monotonic()
        manager.initiate_shutdown()
        duration = time.monotonic() - start_time
        print(f"[DEBUG] Shutdown complete. Duration: {duration}s")
        
        assert task_completed is True
        assert duration >= 0.25
    finally:
        print("[DEBUG] test_webhook_drain_on_shutdown cleanup")
        manager.initiate_shutdown()
    print("[DEBUG] test_webhook_drain_on_shutdown finished")
