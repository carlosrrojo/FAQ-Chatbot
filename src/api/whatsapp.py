"""
WhatsApp Cloud API client.
Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
"""

import logging
import requests

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v22.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/"


class WhatsAppClient:
    def __init__(self, phone_number_id: str, access_token: str):
        self.phone_number_id = phone_number_id
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    # ── Send plain text ───────────────────────────────────────────────────────
    def send_text(self, to: str, text: str) -> dict:
        """Send a plain text message to a WhatsApp number."""
        print(f"Sending message to: {to}")
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
        return self._post("messages", payload)

    # ── Mark message as read ──────────────────────────────────────────────────
    def mark_as_read(self, message_id: str) -> dict:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        return self._post("messages", payload)

    # ── Internal HTTP helper ──────────────────────────────────────────────────
    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{BASE_URL}/{self.phone_number_id}/{endpoint}"
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info("[WA] API response: %s", data)
            return data
        except requests.HTTPError as e:
            logger.error("[WA] HTTP error: %s — %s", e, e.response.text)
            raise
        except requests.RequestException as e:
            logger.error("[WA] Request failed: %s", e)
            raise
