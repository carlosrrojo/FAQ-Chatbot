# src/transport/instagram_adapter.py
from src.domain.models import ChatRequest, Platform


def parse_instagram_payload(data: dict) -> tuple[ChatRequest, str]:
    """
    Parses an Instagram Graph API webhook payload.
    Returns (ChatRequest, message_type).
    """
    entry = data["entry"][0]
    messaging = entry["messaging"][0]

    sender_id = messaging["sender"]["id"]
    message = messaging.get("message", {})
    msg_type = "text" if "text" in message else "unsupported"
    text = message.get("text", "")

    return (
        ChatRequest(
            platform=Platform.INSTAGRAM,
            sender_id=sender_id,
            text=text,
        ),
        msg_type,
    )