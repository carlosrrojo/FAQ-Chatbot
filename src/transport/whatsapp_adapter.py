# src/transport/whatsapp_adapter.py
from src.domain.models import ChatRequest, Platform


def parse_whatsapp_payload(data: dict) -> tuple[ChatRequest, str | None, str]:
    """
    Parses a WhatsApp Cloud API webhook payload.
    Returns (ChatRequest, message_id, message_type).
    Raises ValueError if the payload does not contain a user message.
    """
    entry = data["entry"][0]
    change = entry["changes"][0]["value"]
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