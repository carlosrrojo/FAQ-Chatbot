"""
Domain models (DTOs) for the FAQ-Chatbot project.

These dataclasses and enums provide a typed, validated representation of the
data that flows through the system — from incoming webhook events through the
RAG pipeline and back out as platform replies.

Layers
------
    Webhook event  →  IncomingMessage  →  agent / RAG  →  BotReply  →  API client
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Platform(str, Enum):
    """Messaging platforms supported by the chatbot."""
    WHATSAPP  = "whatsapp"
    INSTAGRAM = "instagram"


class MessageType(str, Enum):
    """Content types the webhook may receive."""
    TEXT       = "text"
    IMAGE      = "image"
    AUDIO      = "audio"
    VIDEO      = "video"
    DOCUMENT   = "document"
    STICKER    = "sticker"
    LOCATION   = "location"
    ATTACHMENT = "attachment"   # Instagram catch-all
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Incoming message
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatRequest:
    """
    Platform-agnostic representation of a message received from a user.

    Built by the webhook dispatchers in ``main.py`` after parsing the
    raw Meta payload.
    """
    platform:   Platform
    sender_id:  str
    text:       Optional[str] = None
    msg_type:   MessageType   = MessageType.TEXT
    message_id: Optional[str] = None     # WA message ID (for mark-as-read)
    raw:        Optional[dict] = None    # original payload for debugging

    @property
    def is_text(self) -> bool:
        return self.msg_type == MessageType.TEXT and self.text is not None


# ---------------------------------------------------------------------------
# Bot reply
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatResponse:
    """
    The chatbot's response to an incoming message.

    Produced by ``generate_reply`` and consumed by the platform clients
    (``WhatsAppClient`` / ``InstagramClient``) to send the answer back.
    """
    platform:  Platform
    sender_id: str
    text:      str


# ---------------------------------------------------------------------------
# Retrieved context (RAG pipeline)
# ---------------------------------------------------------------------------

@dataclass
class RetrievedContext:
    """
    A single chunk returned by the retrieval pipeline.

    Mirrors the LangChain ``Document`` but as a plain DTO so domain logic
    stays decoupled from the framework.
    """
    content:  str
    metadata: dict = field(default_factory=dict)
    score:    Optional[float] = None    # relevance / rerank score


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

class Category(str, Enum):
    """Routing categories. (str, Enum) mixin keeps it 3.10-safe and
    serialisation-friendly; finalise the membership against the classifier."""
    FAQ             = "faq"
    GREETING        = "greeting"
    BOOKING_PAYMENT = "booking_payment"
    PII_DISCLOSURE  = "pii_disclosure"
    UNSUPPORTED     = "unsupported"
    INJECTION       = "injection"
    PRIVACY_RIGHTS  = "privacy_rights"

@dataclass(frozen=True)
class Classification:
    """Result of routing an inbound message. Domain DTO; both the classifier
    adapter and the orchestrator depend on *this*, not on each other."""
    category: Category
    sub_category: str | None
    language: str                 # ISO 639-1: "es", "en", "fr", "de"
    reasoning: str | None = None