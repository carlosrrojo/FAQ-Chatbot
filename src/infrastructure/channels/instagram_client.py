# src/infrastructure/channels/instagram_client.py
import logging
import os
import requests
from src.domain.ports import IMessageChannel
from src.domain.models import ChatResponse
from src.config import INSTAGRAM_MAX_CHARS

logger = logging.getLogger(__name__)
_GRAPH_API_VERSION = "v22.0"


class InstagramClient(IMessageChannel):
    """
    Outbound Instagram DM delivery via Messenger Send API.
    Satisfies FR-CHN-03 (permanent fix — replaces the Step 0.3 patch).
    """

    def __init__(self) -> None:
        self._token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self._base = f"https://graph.facebook.com/{_GRAPH_API_VERSION}/me/messages"

    def send_reply(self, response: ChatResponse) -> None:
        text = response.text[:INSTAGRAM_MAX_CHARS]
        if len(response.text) > INSTAGRAM_MAX_CHARS:
            logger.warning(
                "Instagram response truncated from %d to %d chars for sender %s***",
                len(response.text), INSTAGRAM_MAX_CHARS, response.sender_id[:6]
            )
        payload = {
            "recipient": {"id": response.sender_id},
            "message": {"text": text},
            "messaging_type": "RESPONSE",
        }
        resp = requests.post(
            self._base,
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10,
        )
        resp.raise_for_status()
        logger.debug("Instagram DM delivered to %s***", response.sender_id[:6])

    def mark_as_read(self, message_id: str) -> None:
        # Instagram Messenger API does not expose read receipts in the same way.
        # This is a no-op to satisfy the interface contract.
        pass