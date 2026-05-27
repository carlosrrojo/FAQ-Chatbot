# src/transport/webhook_controller.py
import hashlib
import hmac
import logging
import os
import threading
from flask import Blueprint, request, jsonify, current_app
from src.domain.models import ChatRequest, Platform
from src.transport.whatsapp_adapter import parse_whatsapp_payload
from src.transport.instagram_adapter import parse_instagram_payload

logger = logging.getLogger(__name__)
webhook_bp = Blueprint("webhook", __name__)

_UNSUPPORTED_TYPE_RESPONSE_ES = (
    "Lo siento, actualmente solo puedo procesar mensajes de texto. "
    "Por favor, escríbeme tu consulta y estaré encantado de ayudarte. 😊"
)
_UNSUPPORTED_TYPE_RESPONSE_EN = (
    "I'm sorry, I can only process text messages at the moment. "
    "Please type your question and I'll be happy to help! 😊"
)


@webhook_bp.route("/webhook", methods=["GET"])
def verify():
    """Meta webhook verification endpoint. FR-CHN-04."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == os.getenv("WEBHOOK_VERIFY_TOKEN"):
        logger.info("Webhook verification successful.")
        return challenge, 200
    logger.warning("Webhook verification failed: token mismatch.")
    return "Forbidden", 403


def _verify_signature(payload: bytes, signature_header: str | None) -> bool:
    """
    Validate the X-Hub-Signature-256 header against the raw request body
    using the Meta App Secret (HMAC-SHA256).

    Returns True if the signature is valid, False otherwise.
    FR-CHN-04 (enhanced).
    """
    app_secret = os.getenv("META_APP_SECRET")
    if not app_secret:
        logger.error("META_APP_SECRET not configured — rejecting request.")
        return False

    if not signature_header:
        logger.warning("Missing X-Hub-Signature-256 header.")
        return False

    # Header format: "sha256=<hex digest>"
    if not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(
        app_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature_header[7:])


@webhook_bp.route("/webhook", methods=["POST"])
def receive():
    """
    Main inbound webhook endpoint. Dispatches to platform-specific handlers
    in a background thread to satisfy Meta's 20-second deadline. FR-CHN-01.
    """
    # ── Signature verification (FR-CHN-04) ──────────────────────────
    raw_body = request.get_data()
    signature = request.headers.get("X-Hub-Signature-256")
    if not _verify_signature(raw_body, signature):
        logger.warning("Webhook signature verification failed.")
        return "Unauthorized", 401

    data = request.get_json(silent=True) or {}
    platform_object = data.get("object", "")

    if platform_object == "whatsapp_business_account":
        thread = threading.Thread(
            target=_handle_whatsapp,
            args=(data, current_app._get_current_object()),
            daemon=True,
        )
        thread.start()
    elif platform_object == "instagram":
        thread = threading.Thread(
            target=_handle_instagram,
            args=(data, current_app._get_current_object()),
            daemon=True,
        )
        thread.start()
    else:
        logger.warning("Unknown webhook object type: %s", platform_object)

    return jsonify({"status": "accepted"}), 200


def _handle_whatsapp(data: dict, app) -> None:
    with app.app_context():
        orchestrator = app.config["ORCHESTRATOR"]
        wa_client = app.config["WA_CLIENT"]

        try:
            result = parse_whatsapp_payload(data)
        except (KeyError, IndexError):
            logger.warning("Could not parse WhatsApp payload.", exc_info=True)
            return

        if result is None:
            return

        chat_request, message_id, msg_type = result

        if msg_type != "text":
            # FR-CHN-05: graceful handling of non-text message types
            logger.info("Non-text message type received: %s", msg_type)
            from src.domain.models import ChatResponse
            wa_client.send_reply(ChatResponse(
                text=_UNSUPPORTED_TYPE_RESPONSE_ES,
                sender_id=chat_request.sender_id,
                platform=Platform.WHATSAPP,
            ))
            return

        if message_id:
            wa_client.mark_as_read(message_id)

        response = orchestrator.generate_reply(chat_request)
        wa_client.send_reply(response)


def _handle_instagram(data: dict, app) -> None:
    with app.app_context():
        orchestrator = app.config["ORCHESTRATOR"]
        ig_client = app.config["IG_CLIENT"]

        try:
            chat_request, msg_type = parse_instagram_payload(data)
        except (KeyError, IndexError, ValueError):
            logger.warning("Could not parse Instagram payload.", exc_info=True)
            return

        if msg_type != "text":
            logger.info("Non-text Instagram message type received: %s", msg_type)
            return  # Instagram: silently ignore non-text (no read receipts to send)

        response = orchestrator.generate_reply(chat_request)
        ig_client.send_reply(response)   # FR-CHN-03: bug permanently fixed