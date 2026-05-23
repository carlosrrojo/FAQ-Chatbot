# src/transport/whatsapp_adapter.py
import logging
from src.domain.models import ChatRequest, Platform

logger = logging.getLogger(__name__)


def parse_whatsapp_payload(data: dict) -> tuple[ChatRequest, str | None, str] | None:
    """
    Parses a WhatsApp Cloud API webhook payload.
    Returns (ChatRequest, message_id, message_type), or None for
    non-message webhooks (e.g. delivery/read status updates).
    """
    entry = data["entry"][0]
    change = entry["changes"][0]["value"]

    # Status-update webhooks (sent/delivered/read receipts) carry
    # "statuses" instead of "messages".  Skip silently.
    if "messages" not in change:
        logger.info("Status-update webhook received: %s", change.get("statuses"))
        return None

    message = change["messages"][0]

    sender_id = message["from"]
    message_id = message.get("id")
    msg_type = message.get("type", "unknown")

    text = ""
    if msg_type == "text":
        text = message["text"]["body"]

    return (
        ChatRequest(
            platform=Platform.WHATSAPP,
            sender_id=sender_id,
            text=text,
            message_id=message_id,
        ),
        message_id,
        msg_type,
    )