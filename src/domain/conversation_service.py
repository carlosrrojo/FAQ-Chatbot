# src/domain/conversation_service.py - FR-MEM-03
import logging
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from src.config import MAX_HISTORY_TURNS

logger = logging.getLogger(__name__)


def truncate_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Enforces a sliding window of MAX_HISTORY_TURNS conversation turns.
    One turn = one HumanMessage + one AIMessage.
    The SystemMessage (always first) is always preserved.
    
    FR-MEM-03: Prevents context window overflow on long conversations.
    """
    if not messages:
        return messages

    # Separate the system message from the conversation
    system_msgs = [m for m in messages if m.__class__.__name__ == "SystemMessage"]
    convo_msgs = [m for m in messages if m.__class__.__name__ != "SystemMessage"]

    max_messages = MAX_HISTORY_TURNS * 2   # pairs of (human, AI)
    if len(convo_msgs) > max_messages:
        logger.debug(
            "History truncated: %d → %d messages (window=%d turns)",
            len(convo_msgs), max_messages, MAX_HISTORY_TURNS
        )
        convo_msgs = convo_msgs[-max_messages:]

    return system_msgs + convo_msgs