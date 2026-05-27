from unittest.mock import patch, MagicMock
import os
import pytest

# Set environment variables for testing BEFORE importing create_app
os.environ["WEBHOOK_VERIFY_TOKEN"] = "MySecretToken"
os.environ["WA_PHONE_NUMBER_ID"] = "mock_id"
os.environ["META_ACCESS_TOKEN"] = "mock_token"
os.environ["INSTAGRAM_ACCESS_TOKEN"] = "mock_token"

from src.transport.app import create_app


@pytest.fixture
def app():
    # Patch adapters and background systems so they do not run real logic or start background threads
    with patch("src.transport.app.SqliteMemoryAdapter"), \
         patch("src.transport.app.WhatsAppClient") as mock_wa_cls, \
         patch("src.transport.app.InstagramClient") as mock_ig_cls, \
         patch("src.transport.app.RAGOrchestrator") as mock_orch_cls, \
         patch("src.transport.app.InMemoryDeduplicationStore") as mock_dedup_cls, \
         patch("src.transport.app.start_watcher"):
        
        mock_wa = mock_wa_cls.return_value
        mock_ig = mock_ig_cls.return_value
        mock_orch = mock_orch_cls.return_value
        mock_dedup = mock_dedup_cls.return_value
        mock_dedup.is_duplicate.return_value = False  # default: no dupes

        app = create_app()
        app.config.update({
            "TESTING": True,
            "WA_CLIENT": mock_wa,
            "IG_CLIENT": mock_ig,
            "ORCHESTRATOR": mock_orch,
            "DEDUP_STORE": mock_dedup,
        })
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _bypass_signature_verification():
    """Bypass HMAC signature check for integration tests.

    Signature verification correctness is covered by
    src/tests/test_webhook_signature.py.
    """
    with patch(
        "src.transport.webhook_controller._verify_signature",
        return_value=True,
    ):
        yield


def test_webhook_verification(client):
    # Test correct token
    response = client.get("/webhook", query_string={
        "hub.mode": "subscribe",
        "hub.verify_token": "MySecretToken",
        "hub.challenge": "12345"
    })
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "12345"

    # Test incorrect token
    response = client.get("/webhook", query_string={
        "hub.mode": "subscribe",
        "hub.verify_token": "WrongToken",
        "hub.challenge": "12345"
    })
    assert response.status_code == 403


# SyncThread helper to run the background thread logic synchronously in the test
class SyncThread:
    def __init__(self, target, args=(), **kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def start(self):
        self.target(*self.args)



@patch("src.transport.webhook_controller.threading.Thread", new=SyncThread)
def test_whatsapp_message_handling(client, app):
    mock_orch = app.config["ORCHESTRATOR"]
    mock_wa = app.config["WA_CLIENT"]

    from src.domain.models import ChatResponse, Platform
    mock_orch.generate_reply.return_value = ChatResponse(
        text="This is a mock answer.",
        sender_id="123456789",
        platform=Platform.WHATSAPP
    )

    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "msg_123",
                        "from": "123456789",
                        "type": "text",
                        "text": {"body": "Hello bot"}
                    }]
                }
            }]
        }]
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json == {"status": "accepted"}

    # Assertions
    mock_wa.mark_as_read.assert_called_once_with("msg_123")
    mock_orch.generate_reply.assert_called_once()
    mock_wa.send_reply.assert_called_once()


@patch("src.transport.webhook_controller.threading.Thread", new=SyncThread)
def test_instagram_message_handling(client, app):
    mock_orch = app.config["ORCHESTRATOR"]
    mock_ig = app.config["IG_CLIENT"]

    from src.domain.models import ChatResponse, Platform
    mock_orch.generate_reply.return_value = ChatResponse(
        text="This is a mock answer.",
        sender_id="987654321",
        platform=Platform.INSTAGRAM
    )

    payload = {
        "object": "instagram",
        "entry": [{
            "messaging": [{
                "sender": {"id": "987654321"},
                "message": {"text": "Hello insta"}
            }]
        }]
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json == {"status": "accepted"}

    # Assertions
    mock_orch.generate_reply.assert_called_once()
    mock_ig.send_reply.assert_called_once()


@patch("src.transport.webhook_controller.threading.Thread", new=SyncThread)
@pytest.mark.parametrize("media_type, media_payload", [
    ("audio", {"audio": {"id": "123", "mime_type": "audio/ogg"}}),
    ("image", {"image": {"id": "123", "mime_type": "image/jpeg"}}),
    ("video", {"video": {"id": "123", "mime_type": "video/mp4"}}),
    ("sticker", {"sticker": {"id": "123", "mime_type": "image/webp"}}),
    ("location", {"location": {"latitude": 0.0, "longitude": 0.0}}),
    ("document", {"document": {"id": "123", "mime_type": "application/pdf"}}),
    ("contacts", {"contacts": [{"name": {"formatted_name": "John Doe"}}]}),
    ("interactive", {"interactive": {"type": "list_reply"}}),
    ("button", {"button": {"text": "Click"}}),
    ("reaction", {"reaction": {"message_id": "msg_123", "emoji": "👍"}}),
    ("unknown", {}),
])
def test_whatsapp_non_text_message_handling(client, app, media_type, media_payload):
    mock_orch = app.config["ORCHESTRATOR"]
    mock_wa = app.config["WA_CLIENT"]

    mock_wa.send_reply.reset_mock()
    mock_orch.generate_reply.reset_mock()

    message_data = {
        "id": "msg_123",
        "from": "123456789",
        "type": media_type,
    }
    message_data.update(media_payload)

    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [message_data]
                }
            }]
        }]
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json == {"status": "accepted"}

    # The orchestrator should NOT be called for non-text messages
    mock_orch.generate_reply.assert_not_called()

    # WhatsApp client send_reply should be called with the predefined message
    mock_wa.send_reply.assert_called_once()
    call_args = mock_wa.send_reply.call_args[0][0]
    from src.domain.models import ChatResponse
    assert isinstance(call_args, ChatResponse)
    assert call_args.sender_id == "123456789"
    from src.transport.webhook_controller import _UNSUPPORTED_TYPE_RESPONSE_ES
    assert call_args.text == _UNSUPPORTED_TYPE_RESPONSE_ES


@patch("src.transport.webhook_controller.threading.Thread", new=SyncThread)
def test_duplicate_whatsapp_message_is_suppressed(client, app):
    """FR-CHN-07: duplicate message_id must not trigger a second reply."""
    dedup = app.config["DEDUP_STORE"]
    dedup.is_duplicate.return_value = True  # simulate duplicate

    payload = {
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"value": {"messages": [{
            "id": "msg_dup",
            "from": "123456789",
            "type": "text",
            "text": {"body": "Hello bot"},
        }]}}]}],
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200

    mock_orch = app.config["ORCHESTRATOR"]
    mock_wa = app.config["WA_CLIENT"]
    mock_orch.generate_reply.assert_not_called()
    mock_wa.send_reply.assert_not_called()
    mock_wa.mark_as_read.assert_not_called()


@patch("src.transport.webhook_controller.threading.Thread", new=SyncThread)
def test_duplicate_instagram_message_is_suppressed(client, app):
    """FR-CHN-07: duplicate Instagram mid must not trigger a second reply."""
    dedup = app.config["DEDUP_STORE"]
    dedup.is_duplicate.return_value = True  # simulate duplicate

    payload = {
        "object": "instagram",
        "entry": [{"messaging": [{"sender": {"id": "987654321"}, "message": {
            "mid": "mid_dup",
            "text": "Hello insta",
        }}]}],
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200

    mock_orch = app.config["ORCHESTRATOR"]
    mock_ig = app.config["IG_CLIENT"]
    mock_orch.generate_reply.assert_not_called()
    mock_ig.send_reply.assert_not_called()


