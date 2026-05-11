"""
Instagram Messaging API client.
Docs: https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/messaging
"""

import logging
import requests

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v22.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class InstagramClient:
    def __init__(self, page_id: str, access_token: str):
        self.page_id = page_id
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    # ── Send plain text ───────────────────────────────────────────────────────
    def send_text(self, to: str, text: str) -> dict:
        """Send a text message to an Instagram user via the Page inbox."""
        logger.info("Sending IG message to: %s", to)
        payload = {
            "recipient": {"id": to},
            "message": {"text": text},
        }
        return self._post("messages", payload)

    # ── Internal HTTP helper ──────────────────────────────────────────────────
    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{BASE_URL}/{self.page_id}/{endpoint}"
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info("[IG] API response: %s", data)
            return data
        except requests.HTTPError as e:
            logger.error("[IG] HTTP error: %s — %s", e, e.response.text)
            raise
        except requests.RequestException as e:
            logger.error("[IG] Request failed: %s", e)
            raise
