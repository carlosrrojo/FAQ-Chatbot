# src/tests/test_summarization.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from src.rag.agent import _manage_memory, summarize_history

@pytest.fixture(autouse=True)
def mock_config():
    # Patch MAX_HISTORY_TURNS to a low number (e.g. 2 turns = 4 messages) for simple test invocation
    with patch("src.rag.agent.MAX_HISTORY_TURNS", 2):
        yield

# --- Active Path (Truncation Safety Net) Tests ---

def test_active_path_under_safety_limit():
    # With summarization enabled, limit is MAX_HISTORY_TURNS * 4 = 8.
    # 6 messages is under safety limit, so nothing is returned (no hard truncation).
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
    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        result = _manage_memory(state)
    assert result == {"messages": []}

def test_active_path_over_safety_limit_with_summarization():
    # With summarization enabled, limit is MAX_HISTORY_TURNS * 4 = 8.
    # 10 messages exceeds 8. Truncates down to MAX_HISTORY_TURNS * 2 = 4 messages (retaining h4, ai4, h5, ai5).
    # Removed messages: h1, ai1, h2, ai2, h3, ai3.
    state = {
        "messages": [
            HumanMessage(content="Turn 1", id="h1"),
            AIMessage(content="Response 1", id="ai1"),
            HumanMessage(content="Turn 2", id="h2"),
            AIMessage(content="Response 2", id="ai2"),
            HumanMessage(content="Turn 3", id="h3"),
            AIMessage(content="Response 3", id="ai3"),
            HumanMessage(content="Turn 4", id="h4"),
            AIMessage(content="Response 4", id="ai4"),
            HumanMessage(content="Turn 5", id="h5"),
            AIMessage(content="Response 5", id="ai5"),
        ]
    }
    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        result = _manage_memory(state)
    
    messages = result["messages"]
    assert len(messages) == 6
    remove_ids = [m.id for m in messages if isinstance(m, RemoveMessage)]
    assert len(remove_ids) == 6
    assert set(remove_ids) == {"h1", "ai1", "h2", "ai2", "h3", "ai3"}

def test_active_path_over_safety_limit_without_summarization():
    # With summarization disabled, limit is MAX_HISTORY_TURNS * 2 = 4.
    # 6 messages exceeds 4. Truncates down to MAX_HISTORY_TURNS * 2 = 4 messages (retaining h2, ai2, h3, ai3).
    # Removed messages: h1, ai1.
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
    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", False):
        result = _manage_memory(state)
    
    messages = result["messages"]
    assert len(messages) == 2
    remove_ids = [m.id for m in messages if isinstance(m, RemoveMessage)]
    assert len(remove_ids) == 2
    assert set(remove_ids) == {"h1", "ai1"}


# --- Background Path (Summarization Worker) Tests ---

def test_background_under_threshold():
    # Convo threshold is MAX_HISTORY_TURNS * 2 = 4.
    # 4 messages is <= 4, so no summarization should happen.
    messages = [
        HumanMessage(content="Turn 1", id="h1"),
        AIMessage(content="Response 1", id="ai1"),
        HumanMessage(content="Turn 2", id="h2"),
        AIMessage(content="Response 2", id="ai2"),
    ]
    mock_agent = MagicMock()
    mock_agent.get_state.return_value = MagicMock(values={"messages": messages})

    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        summarize_history(mock_agent, "thread-1")

    mock_agent.get_state.assert_called_once_with({"configurable": {"thread_id": "thread-1"}})
    mock_agent.update_state.assert_not_called()

@patch("src.rag.agent._llm")
def test_background_over_threshold_summarization(mock_llm):
    # Convo threshold is MAX_HISTORY_TURNS * 2 = 4.
    # 6 messages > 4, so we summarize the older ones (h1, ai1) and keep 4 (h2, ai2, h3, ai3).
    mock_llm.invoke.return_value = MagicMock(content="Summary of first turn.")

    messages = [
        HumanMessage(content="Turn 1", id="h1"),
        AIMessage(content="Response 1", id="ai1"),
        HumanMessage(content="Turn 2", id="h2"),
        AIMessage(content="Response 2", id="ai2"),
        HumanMessage(content="Turn 3", id="h3"),
        AIMessage(content="Response 3", id="ai3"),
    ]
    mock_agent = MagicMock()
    mock_agent.get_state.return_value = MagicMock(values={"messages": messages})

    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        summarize_history(mock_agent, "thread-1")

    mock_agent.get_state.assert_called_once_with({"configurable": {"thread_id": "thread-1"}})
    mock_agent.update_state.assert_called_once()

    update_args = mock_agent.update_state.call_args
    assert update_args[0][0] == {"configurable": {"thread_id": "thread-1"}}
    assert update_args[1]["as_node"] == "manage_memory"

    returned_messages = update_args[0][1]["messages"]
    assert len(returned_messages) == 3  # 2 RemoveMessages + 1 SystemMessage (summary)
    remove_ids = [m.id for m in returned_messages if isinstance(m, RemoveMessage)]
    assert "h1" in remove_ids
    assert "ai1" in remove_ids

    summary_msg = [m for m in returned_messages if isinstance(m, SystemMessage)][0]
    assert summary_msg.id == "conversation_summary"
    assert "Summary of first turn" in summary_msg.content

    # Verify LLM prompt contains the messages to remove but not the kept ones
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert "Turn 1" in prompt_sent
    assert "Response 1" in prompt_sent
    assert "Turn 2" not in prompt_sent

@patch("src.rag.agent._llm")
def test_background_recursive_summarization(mock_llm):
    # Convo threshold is 4.
    # Existing summary + 6 messages -> we remove existing summary + h1 + ai1, keeping h2, ai2, h3, ai3.
    mock_llm.invoke.return_value = MagicMock(content="Combined summary of old and new.")

    existing_summary = SystemMessage(content="Resumen de la conversación anterior: Old summary.", id="conversation_summary")
    messages = [
        existing_summary,
        HumanMessage(content="Turn 1", id="h1"),
        AIMessage(content="Response 1", id="ai1"),
        HumanMessage(content="Turn 2", id="h2"),
        AIMessage(content="Response 2", id="ai2"),
        HumanMessage(content="Turn 3", id="h3"),
        AIMessage(content="Response 3", id="ai3"),
    ]
    mock_agent = MagicMock()
    mock_agent.get_state.return_value = MagicMock(values={"messages": messages})

    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        summarize_history(mock_agent, "thread-1")

    mock_agent.update_state.assert_called_once()
    returned_messages = mock_agent.update_state.call_args[0][1]["messages"]

    assert len(returned_messages) == 4  # 3 RemoveMessages + 1 SystemMessage (summary)
    remove_ids = [m.id for m in returned_messages if isinstance(m, RemoveMessage)]
    assert "h1" in remove_ids
    assert "ai1" in remove_ids
    assert "conversation_summary" in remove_ids

    new_summary = [m for m in returned_messages if isinstance(m, SystemMessage)][0]
    assert new_summary.id == "conversation_summary"
    assert new_summary.content == "Resumen de la conversación anterior: Combined summary of old and new."

    # Verify LLM prompt contains the old summary
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert "Old summary." in prompt_sent
    assert "Turn 1" in prompt_sent

@patch("src.rag.agent._llm")
def test_background_disabled(mock_llm):
    # When disabled, summarize_history should return immediately without doing anything.
    messages = [
        HumanMessage(content="Turn 1", id="h1"),
        AIMessage(content="Response 1", id="ai1"),
        HumanMessage(content="Turn 2", id="h2"),
        AIMessage(content="Response 2", id="ai2"),
        HumanMessage(content="Turn 3", id="h3"),
        AIMessage(content="Response 3", id="ai3"),
    ]
    mock_agent = MagicMock()

    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", False):
        summarize_history(mock_agent, "thread-1")

    mock_agent.get_state.assert_not_called()
    mock_agent.update_state.assert_not_called()
    mock_llm.invoke.assert_not_called()

@patch("src.rag.agent._llm")
def test_background_fallback_on_error(mock_llm):
    # Verify that summarize_history catches exceptions from LLM invocation gracefully,
    # without propagating them or calling update_state.
    mock_llm.invoke.side_effect = Exception("LLM connection timed out")

    messages = [
        HumanMessage(content="Turn 1", id="h1"),
        AIMessage(content="Response 1", id="ai1"),
        HumanMessage(content="Turn 2", id="h2"),
        AIMessage(content="Response 2", id="ai2"),
        HumanMessage(content="Turn 3", id="h3"),
        AIMessage(content="Response 3", id="ai3"),
    ]
    mock_agent = MagicMock()
    mock_agent.get_state.return_value = MagicMock(values={"messages": messages})

    with patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        summarize_history(mock_agent, "thread-1")

    mock_agent.get_state.assert_called_once_with({"configurable": {"thread_id": "thread-1"}})
    mock_agent.update_state.assert_not_called()  # Updates should not occur if summarization fails
