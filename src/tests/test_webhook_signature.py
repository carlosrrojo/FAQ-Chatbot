"""
Unit tests for webhook HMAC-SHA256 signature verification (FR-CHN-04).

Verifies that the POST /webhook endpoint correctly validates the
X-Hub-Signature-256 header using the META_APP_SECRET env var.
"""

import hashlib
import hmac
import json
import os
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask

from src.transport.webhook_controller import webhook_bp


APP_SECRET = "test_secret_key_for_unit_tests"

SAMPLE_PAYLOAD = json.dumps({
    "object": "whatsapp_business_account",
    "entry": [],
}).encode()


def _compute_signature(payload: bytes, secret: str = APP_SECRET) -> str:
    """Compute a valid X-Hub-Signature-256 header value."""
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.fixture()
def app():
    """Minimal Flask app wired with the webhook blueprint."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["ORCHESTRATOR"] = MagicMock()
    flask_app.config["WA_CLIENT"] = MagicMock()
    flask_app.config["IG_CLIENT"] = MagicMock()
    flask_app.config["DEDUP_STORE"] = MagicMock()
    flask_app.register_blueprint(webhook_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# ── Test cases ────────────────────────────────────────────────────────────────


@patch.dict(os.environ, {"META_APP_SECRET": APP_SECRET})
def test_valid_signature_accepted(client):
    """A correctly signed request should be accepted (200)."""
    sig = _compute_signature(SAMPLE_PAYLOAD)
    resp = client.post(
        "/webhook",
        data=SAMPLE_PAYLOAD,
        content_type="application/json",
        headers={"X-Hub-Signature-256": sig},
    )
    assert resp.status_code == 200


@patch.dict(os.environ, {"META_APP_SECRET": APP_SECRET})
def test_invalid_signature_rejected(client):
    """A request with a tampered signature should be rejected (401)."""
    bad_sig = "sha256=" + "a" * 64  # wrong digest
    resp = client.post(
        "/webhook",
        data=SAMPLE_PAYLOAD,
        content_type="application/json",
        headers={"X-Hub-Signature-256": bad_sig},
    )
    assert resp.status_code == 401


@patch.dict(os.environ, {"META_APP_SECRET": APP_SECRET})
def test_missing_signature_rejected(client):
    """A request with no X-Hub-Signature-256 header should be rejected (401)."""
    resp = client.post(
        "/webhook",
        data=SAMPLE_PAYLOAD,
        content_type="application/json",
    )
    assert resp.status_code == 401


@patch.dict(os.environ, {}, clear=False)
def test_missing_app_secret_rejected(client):
    """If META_APP_SECRET is not set, all requests should be rejected (fail-closed)."""
    # Ensure the env var is absent
    os.environ.pop("META_APP_SECRET", None)

    sig = _compute_signature(SAMPLE_PAYLOAD)
    resp = client.post(
        "/webhook",
        data=SAMPLE_PAYLOAD,
        content_type="application/json",
        headers={"X-Hub-Signature-256": sig},
    )
    assert resp.status_code == 401
