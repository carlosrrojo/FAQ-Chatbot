# src/tests/test_summarization.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from src.rag.agent import _manage_memory

@pytest.fixture(autouse=True)
def mock_config():
    # Patch MAX_HISTORY_TURNS to a low number (e.g. 2 turns = 4 messages) for simple test invocation
    with patch("src.rag.agent.MAX_HISTORY_TURNS", 2), \
         patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        yield

def test_under_turn_limit():
    state = {
        "messages": [
            HumanMessage(content="Hola", id="h1"),
            AIMessage(content="¡Hola! ¿En qué puedo ayudarte?", id="ai1"),
        ]
    }
    result = _manage_memory(state)
    assert result == {"messages": []}

@patch("src.rag.agent._llm")
def test_over_turn_limit_summarization(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="Summary of first two turns.")

    state = {
        "messages": [
            HumanMessage(content="Turn 1", id="h1"),
            AIMessage(content="Response 1", id="ai1"),
            HumanMessage(content="Turn 2", id="h2"),
            AIMessage(content="Response 2", id="ai2"),
            HumanMessage(content="Turn 3", id="h3"),
            AIMessage(content="Response 3", id="ai3"),
        ]
    }

    result = _manage_memory(state)
    messages = result["messages"]

    # We expect h1 and ai1 to be removed, and a new summary system message to be added
    # Total messages: 6, max: 4. So 2 oldest (h1, ai1) are removed.
    assert len(messages) == 3
    remove_ids = [m.id for m in messages if isinstance(m, RemoveMessage)]
    assert "h1" in remove_ids
    assert "ai1" in remove_ids

    summary_msg = [m for m in messages if isinstance(m, SystemMessage)][0]
    assert summary_msg.id == "conversation_summary"
    assert "Summary of first two turns" in summary_msg.content

    # Verify LLM prompt contents
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert "Turn 1" in prompt_sent
    assert "Response 1" in prompt_sent
    assert "Turn 2" not in prompt_sent  # only to_remove are sent to summarize

@patch("src.rag.agent._llm")
def test_recursive_summarization(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="Combined summary of old and new.")

    existing_summary = SystemMessage(content="Resumen de la conversación anterior: Old summary.", id="conversation_summary")
    state = {
        "messages": [
            existing_summary,
            HumanMessage(content="Turn 1", id="h1"),
            AIMessage(content="Response 1", id="ai1"),
            HumanMessage(content="Turn 2", id="h2"),
            AIMessage(content="Response 2", id="ai2"),
            HumanMessage(content="Turn 3", id="h3"),
            AIMessage(content="Response 3", id="ai3"),
        ]
    }

    result = _manage_memory(state)
    messages = result["messages"]

    # Expect: remove h1, ai1, AND the existing summary, and append new summary
    assert len(messages) == 4
    remove_ids = [m.id for m in messages if isinstance(m, RemoveMessage)]
    assert "h1" in remove_ids
    assert "ai1" in remove_ids
    assert "conversation_summary" in remove_ids

    new_summary = [m for m in messages if isinstance(m, SystemMessage)][0]
    assert new_summary.id == "conversation_summary"
    assert new_summary.content == "Resumen de la conversación anterior: Combined summary of old and new."

    # Verify LLM prompt contains the old summary
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert "Old summary." in prompt_sent
    assert "Turn 1" in prompt_sent

@patch("src.rag.agent._llm")
def test_summarization_disabled(mock_llm):
    # Set disabled
    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", False):
        existing_summary = SystemMessage(content="Resumen de la conversación anterior: Old summary.", id="conversation_summary")
        state = {
            "messages": [
                existing_summary,
                HumanMessage(content="Turn 1", id="h1"),
                AIMessage(content="Response 1", id="ai1"),
                HumanMessage(content="Turn 2", id="h2"),
                AIMessage(content="Response 2", id="ai2"),
                HumanMessage(content="Turn 3", id="h3"),
                AIMessage(content="Response 3", id="ai3"),
            ]
        }

        result = _manage_memory(state)
        messages = result["messages"]

        # Expect: remove h1, ai1, and old summary, but NO new summary is added
        assert len(messages) == 3
        for m in messages:
            assert isinstance(m, RemoveMessage)
        
        remove_ids = [m.id for m in messages]
        assert "h1" in remove_ids
        assert "ai1" in remove_ids
        assert "conversation_summary" in remove_ids

        # LLM should not be called
        mock_llm.invoke.assert_not_called()

@patch("src.rag.agent._llm")
def test_summarization_fallback_on_error(mock_llm):
    # Force LLM error
    mock_llm.invoke.side_effect = Exception("LLM connection timed out")

    existing_summary = SystemMessage(content="Resumen de la conversación anterior: Old summary.", id="conversation_summary")
    state = {
        "messages": [
            existing_summary,
            HumanMessage(content="Turn 1", id="h1"),
            AIMessage(content="Response 1", id="ai1"),
            HumanMessage(content="Turn 2", id="h2"),
            AIMessage(content="Response 2", id="ai2"),
            HumanMessage(content="Turn 3", id="h3"),
            AIMessage(content="Response 3", id="ai3"),
        ]
    }

    result = _manage_memory(state)
    messages = result["messages"]

    # Verify fallback: hard truncation was executed, removing old summary and old turns without crashing
    assert len(messages) == 3
    for m in messages:
        assert isinstance(m, RemoveMessage)
    
    remove_ids = [m.id for m in messages]
    assert "h1" in remove_ids
    assert "ai1" in remove_ids
    assert "conversation_summary" in remove_ids
