"""
Flask webhook server for the Espazo Nature chatbot.

Handles incoming events from Meta (WhatsApp + Instagram) and dispatches
them to the RAG agent via :func:`src.rag.agent.generate_reply`.
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, request

from src.rag.agent import generate_reply
from src.rag.config import DATA_PATH
from src.rag.watcher import start_watcher
from src.api.whatsapp import WhatsAppClient
from src.api.instagram import InstagramClient

from src.logging_config import configure_logging

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────
whatsapp = WhatsAppClient(
    phone_number_id=os.environ["WA_PHONE_NUMBER_ID"],
    access_token=os.environ["META_ACCESS_TOKEN"],
)

instagram = InstagramClient(
    page_id=os.environ["IG_PAGE_ID"],
    access_token=os.environ["META_ACCESS_TOKEN"],
)

VERIFY_TOKEN = os.environ["WEBHOOK_VERIFY_TOKEN"]

# ── Document watcher (started once at import time) ────────────────────────────
_observer = None
try:
    abs_data_path = os.path.abspath(DATA_PATH)
    os.makedirs(abs_data_path, exist_ok=True)
    _observer = start_watcher(abs_data_path)
    logger.info("Document watcher started on %s", abs_data_path)
except Exception:
    logger.exception("Failed to start document watcher")


# ── Webhook verification (GET) ────────────────────────────────────────────────
@app.get("/webhook")
def verify_webhook():
    """
    Meta sends a GET to verify the webhook endpoint.
    Must echo back ``hub.challenge`` when the token matches.
    """
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return challenge, 200

    logger.warning("Webhook verification failed. Check WEBHOOK_VERIFY_TOKEN.")
    return "Forbidden", 403


# ── Incoming events (POST) ────────────────────────────────────────────────────
@app.post("/webhook")
def handle_webhook():
    """
    Receive all incoming events from Meta (WhatsApp + Instagram).
    Always responds 200 quickly; heavy processing happens synchronously
    (move to a task queue for production scale).
    """
    data = request.get_json(silent=True)
    if not data:
        return "Bad Request", 400

    object_type = data.get("object")

    import threading

    def process_event():
        try:
            if object_type == "whatsapp_business_account":
                _handle_whatsapp(data)
            elif object_type == "instagram":
                _handle_instagram(data)
            else:
                logger.warning("Unknown webhook object type: %s", object_type)
        except Exception:
            # Always return 200 so Meta does not retry; log the error internally.
            logger.exception("Unhandled error processing webhook")

    # Run processing asynchronously to avoid Meta webhook timeouts and retries
    threading.Thread(target=process_event).start()

    return "EVENT_RECEIVED", 200


# ── WhatsApp dispatcher ───────────────────────────────────────────────────────
def _handle_whatsapp(data: dict) -> None:
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value    = change.get("value", {})
            messages = value.get("messages", [])

            for msg in messages:
                msg_type = msg.get("type")
                sender   = msg.get("from")  # E.164 phone number

                if msg_type == "text":
                    user_text = msg["text"]["body"]
                    logger.info("[WA] Message from %s: %s", sender, user_text)
                    reply = generate_reply("whatsapp", user_text, sender)
                    whatsapp.send_text(to=sender, text=reply)
                else:
                    logger.info(
                        "[WA] Unsupported message type '%s' from %s", msg_type, sender
                    )
                    whatsapp.send_text(
                        to=sender,
                        text="Sorry, I can only process text messages at the moment.",
                    )


# ── Instagram dispatcher ──────────────────────────────────────────────────────
def _handle_instagram(data: dict) -> None:
    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            msg       = event.get("message", {})

            if not msg or msg.get("is_echo"):
                continue  # ignore echo events (messages sent BY the page)

            if "text" in msg:
                user_text = msg["text"]
                logger.info("[IG] DM from %s: %s", sender_id, user_text)
                reply = generate_reply("instagram", user_text, sender_id)
                instagram.send_text(to=sender_id, text=reply)
            else:
                logger.info(
                    "[IG] Unsupported message type from %s", sender_id
                )
                instagram.send_text(
                    to=sender_id,
                    text="Sorry, I can only process text messages at the moment.",
                )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="localhost", port=port, debug=False)