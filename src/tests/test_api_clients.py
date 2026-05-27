"""
Unit tests for WhatsAppClient and InstagramClient under the new architecture.

Mocks ``requests.post`` to verify each client builds the correct URL,
headers, and JSON payload without making real HTTP calls.
"""

from unittest.mock import patch, MagicMock
import pytest
import os
import requests

from src.infrastructure.channels.whatsapp_client import WhatsAppClient
from src.infrastructure.channels.instagram_client import InstagramClient
from src.domain.models import ChatResponse, Platform


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------

class TestWhatsAppClient:
    """Tests for :class:`WhatsAppClient`."""

    PHONE_ID = "123456"
    TOKEN = "test-token"

    @pytest.fixture(autouse=True)
    def setup_env(self):
        os.environ["WA_PHONE_NUMBER_ID"] = self.PHONE_ID
        os.environ["META_ACCESS_TOKEN"] = self.TOKEN
        yield
        # Clean up
        os.environ.pop("WA_PHONE_NUMBER_ID", None)
        os.environ.pop("META_ACCESS_TOKEN", None)

    @patch("src.infrastructure.channels.whatsapp_client.requests.post")
    def test_send_reply_payload(self, mock_post: MagicMock):
        """send_reply should POST the correct WhatsApp Cloud API payload."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = WhatsAppClient()
        response = ChatResponse(
            text="Hola mundo",
            sender_id="34600000000",
            platform=Platform.WHATSAPP
        )
        client.send_reply(response)

        mock_post.assert_called_once_with(
            f"https://graph.facebook.com/v22.0/{self.PHONE_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": "34600000000",
                "type": "text",
                "text": {"body": "Hola mundo"},
            },
            headers={
                "Authorization": f"Bearer {self.TOKEN}",
            },
            timeout=10,
        )

    @patch("src.infrastructure.channels.whatsapp_client.requests.post")
    def test_send_reply_splitting(self, mock_post: MagicMock):
        """send_reply should split long responses into multiple messages."""
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        client = WhatsAppClient()
        # Create a message longer than WHATSAPP_MAX_CHARS (4096)
        part1 = "A" * 4000 + "."
        part2 = "B" * 500 + "."
        long_text = part1 + " " + part2
        
        response = ChatResponse(
            text=long_text,
            sender_id="34600000000",
            platform=Platform.WHATSAPP
        )
        client.send_reply(response)

        assert mock_post.call_count == 2
        # Check first call
        mock_post.assert_any_call(
            f"https://graph.facebook.com/v22.0/{self.PHONE_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": "34600000000",
                "type": "text",
                "text": {"body": part1},
            },
            headers={"Authorization": f"Bearer {self.TOKEN}"},
            timeout=10,
        )
        # Check second call
        mock_post.assert_any_call(
            f"https://graph.facebook.com/v22.0/{self.PHONE_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": "34600000000",
                "type": "text",
                "text": {"body": part2},
            },
            headers={"Authorization": f"Bearer {self.TOKEN}"},
            timeout=10,
        )

    @patch("src.infrastructure.channels.whatsapp_client.requests.post")
    def test_mark_as_read_payload(self, mock_post: MagicMock):
        """mark_as_read should POST a status update for the given message ID."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = WhatsAppClient()
        client.mark_as_read("wamid.abc123")

        mock_post.assert_called_once_with(
            f"https://graph.facebook.com/v22.0/{self.PHONE_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": "wamid.abc123",
            },
            headers={
                "Authorization": f"Bearer {self.TOKEN}",
            },
            timeout=10,
        )

    @patch("src.infrastructure.channels.whatsapp_client.requests.post")
    def test_send_reply_raises_on_http_error(self, mock_post: MagicMock):
        """send_reply should propagate HTTPError from the API."""
        mock_response = MagicMock()
        error = requests.HTTPError("400 Bad Request")
        error.response = MagicMock(text="invalid payload")
        mock_response.raise_for_status.side_effect = error
        mock_post.return_value = mock_response

        client = WhatsAppClient()
        response = ChatResponse(
            text="fail",
            sender_id="34600000000",
            platform=Platform.WHATSAPP
        )
        with pytest.raises(requests.HTTPError):
            client.send_reply(response)

    @patch("src.infrastructure.channels.whatsapp_client.requests.post")
    def test_send_text_direct(self, mock_post: MagicMock):
        """send_text should directly enforce limits and split text at sentence boundaries."""
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        client = WhatsAppClient()
        part1 = "A" * 4000 + "."
        part2 = "B" * 500 + "."
        long_text = part1 + " " + part2

        client._send_text("34600000000", long_text)

        assert mock_post.call_count == 2
        mock_post.assert_any_call(
            f"https://graph.facebook.com/v22.0/{self.PHONE_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": "34600000000",
                "type": "text",
                "text": {"body": part1},
            },
            headers={"Authorization": f"Bearer {self.TOKEN}"},
            timeout=10,
        )
        mock_post.assert_any_call(
            f"https://graph.facebook.com/v22.0/{self.PHONE_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": "34600000000",
                "type": "text",
                "text": {"body": part2},
            },
            headers={"Authorization": f"Bearer {self.TOKEN}"},
            timeout=10,
        )


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------

class TestInstagramClient:
    """Tests for :class:`InstagramClient`."""

    TOKEN = "ig-test-token"

    @pytest.fixture(autouse=True)
    def setup_env(self):
        os.environ["INSTAGRAM_ACCESS_TOKEN"] = self.TOKEN
        yield
        os.environ.pop("INSTAGRAM_ACCESS_TOKEN", None)

    @patch("src.infrastructure.channels.instagram_client.requests.post")
    def test_send_reply_payload(self, mock_post: MagicMock):
        """send_reply should POST the correct Instagram Send API payload."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = InstagramClient()
        response = ChatResponse(
            text="¡Bienvenido!",
            sender_id="user-456",
            platform=Platform.INSTAGRAM
        )
        client.send_reply(response)

        mock_post.assert_called_once_with(
            "https://graph.facebook.com/v22.0/me/messages",
            json={
                "recipient": {"id": "user-456"},
                "message": {"text": "¡Bienvenido!"},
                "messaging_type": "RESPONSE",
            },
            headers={
                "Authorization": f"Bearer {self.TOKEN}",
            },
            timeout=10,
        )

    @patch("src.infrastructure.channels.instagram_client.requests.post")
    def test_send_reply_truncation(self, mock_post: MagicMock):
        """send_reply should truncate message if it exceeds limit."""
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        client = InstagramClient()
        from src.config import INSTAGRAM_MAX_CHARS
        long_text = "A" * (INSTAGRAM_MAX_CHARS + 100)
        
        response = ChatResponse(
            text=long_text,
            sender_id="user-456",
            platform=Platform.INSTAGRAM
        )
        client.send_reply(response)

        mock_post.assert_called_once_with(
            "https://graph.facebook.com/v22.0/me/messages",
            json={
                "recipient": {"id": "user-456"},
                "message": {"text": "A" * INSTAGRAM_MAX_CHARS},
                "messaging_type": "RESPONSE",
            },
            headers={
                "Authorization": f"Bearer {self.TOKEN}",
            },
            timeout=10,
        )

    @patch("src.infrastructure.channels.instagram_client.requests.post")
    def test_send_reply_raises_on_http_error(self, mock_post: MagicMock):
        """send_reply should propagate HTTPError from the API."""
        mock_response = MagicMock()
        error = requests.HTTPError("400 Bad Request")
        error.response = MagicMock(text="invalid payload")
        mock_response.raise_for_status.side_effect = error
        mock_post.return_value = mock_response

        client = InstagramClient()
        response = ChatResponse(
            text="fail",
            sender_id="user-456",
            platform=Platform.INSTAGRAM
        )
        with pytest.raises(requests.HTTPError):
            client.send_reply(response)
