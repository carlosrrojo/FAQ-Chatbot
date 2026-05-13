# src/domain/ports.py
from abc import ABC, abstractmethod
from typing import Optional
from .models import ChatRequest, ChatResponse, RetrievedContext


class IRetriever(ABC):
    """Port: document retrieval. Implemented by HybridRetriever."""

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedContext]:
        """Return top-K reranked documents for the given query."""
        ...

    @abstractmethod
    def rebuild_index(self) -> None:
        """Signal the retriever to rebuild its sparse index (post-ingestion)."""
        ...


class IMemoryStore(ABC):
    """Port: conversation checkpointing. Implemented by SqliteMemory."""

    @abstractmethod
    def get_checkpointer(self):
        """Return the LangGraph-compatible checkpointer object."""
        ...

    @abstractmethod
    def delete_session(self, sender_id: str) -> None:
        """Permanently erase all data for a given user. (FR-PRV-02)"""
        ...


class IMessageChannel(ABC):
    """Port: outbound message delivery. One implementation per platform."""

    @abstractmethod
    def send_reply(self, response: ChatResponse) -> None:
        """Deliver the response to the user on this channel."""
        ...

    @abstractmethod
    def mark_as_read(self, message_id: str) -> None:
        """Signal read receipt where supported by the platform."""
        ...