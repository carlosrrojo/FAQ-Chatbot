"""
Unit tests for the RAG Agent (agent unit behavior).
Tests memory sliding window truncation, language detection, and graph compilation.
"""

from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from src.rag.agent import _manage_memory, _detect_user_language, build_graph
from src.config import MAX_HISTORY_TURNS


def test_manage_memory_no_truncation():
    # History within limit should not request message removal
    messages = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there", id="2")]
    result = _manage_memory({"messages": messages})
    assert result == {"messages": []}


@patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", False)
def test_manage_memory_truncation():
    # Create messages exceeding MAX_HISTORY_TURNS * 2 limit
    limit = MAX_HISTORY_TURNS * 2
    messages = []
    for i in range(limit + 4):
        if i % 2 == 0:
            messages.append(HumanMessage(content=f"User {i}", id=str(i)))
        else:
            messages.append(AIMessage(content=f"AI {i}", id=str(i)))

    result = _manage_memory({"messages": messages})
    
    # It should request removal of the first 4 messages
    removed = result["messages"]
    assert len(removed) == 4
    assert all(isinstance(msg, RemoveMessage) for msg in removed)
    assert [msg.id for msg in removed] == ["0", "1", "2", "3"]



def test_detect_user_language_spanish():
    messages = [
        HumanMessage(content="Hola, ¿cómo estás?", id="1"),
        AIMessage(content="Estoy bien, gracias.", id="2")
    ]
    lang = _detect_user_language(messages)
    assert lang == "es"


def test_detect_user_language_english():
    messages = [
        HumanMessage(content="How are you doing today?", id="1")
    ]
    lang = _detect_user_language(messages)
    assert lang == "en"


def test_detect_user_language_short_fallback():
    # If text is too short, fallback to Spanish ("es")
    messages = [
        HumanMessage(content="Hi", id="1")
    ]
    lang = _detect_user_language(messages)
    assert lang == "es"


def test_build_graph():
    # Verifies build_graph compiles a valid Graph object
    with patch("src.rag.agent._llm"), patch("src.rag.agent.embeddings"):
        graph = build_graph(skip_memory=True)
        assert graph is not None
        assert hasattr(graph, "invoke")
