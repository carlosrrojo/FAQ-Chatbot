# src/infrastructure/channels/whatsapp_client.py
import logging
import os
import re
import requests
from src.domain.ports import IMessageChannel
from src.domain.models import ChatResponse
from src.config import WHATSAPP_MAX_CHARS

logger = logging.getLogger(__name__)
_GRAPH_API_VERSION = "v22.0"


def _split_at_sentence_boundary(text: str, limit: int) -> list[str]:
    """
    Splits text into fragments ≤ limit characters, breaking at sentence
    boundaries ('. ', '! ', '? ') where possible.
    """
    if len(text) <= limit:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    fragments, current = [], ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                fragments.append(current)
            # If a single sentence exceeds the limit, hard-split it.
            while len(sentence) > limit:
                fragments.append(sentence[:limit])
                sentence = sentence[limit:]
            current = sentence
    if current:
        fragments.append(current)
    return fragments


class WhatsAppClient(IMessageChannel):

    def __init__(self) -> None:
        self._token = os.getenv("META_ACCESS_TOKEN")
        self._phone_id = os.getenv("WA_PHONE_NUMBER_ID")
        self._base = f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{self._phone_id}"

    def send_reply(self, response: ChatResponse) -> None:
        """
        Delivers the reply, splitting into multiple messages if the text
        exceeds WHATSAPP_MAX_CHARS (4096). FR-CHN-06.
        """
        fragments = _split_at_sentence_boundary(response.text, WHATSAPP_MAX_CHARS)
        if len(fragments) > 1:
            logger.info("Response split into %d fragments for sender %s",
                        len(fragments), response.sender_id[:6] + "***")
        for fragment in fragments:
            self._send_text(response.sender_id, fragment)

    def _send_text(self, to: str, text: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }
        resp = requests.post(
            f"{self._base}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10,
        )
        resp.raise_for_status()
        logger.debug("WhatsApp message delivered to %s***", to[:6])

    def mark_as_read(self, message_id: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        resp = requests.post(
            f"{self._base}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10,
        )
        resp.raise_for_status()