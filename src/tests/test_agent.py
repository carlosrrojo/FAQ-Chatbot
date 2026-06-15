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


def test_truncate_history_snapping():
    from src.domain.conversation_service import truncate_history
    from langchain_core.messages import SystemMessage, ToolMessage
    
    # Setup test messages: System + 6 convo messages. MAX_HISTORY_TURNS = 2 (so max_messages = 4)
    # 0: Human
    # 1: AI (with tool call)
    # 2: Tool (tool response) - tentative truncation boundary starts here (len 6 - 4 = index 2)
    # 3: AI (final turn response)
    # 4: Human (next turn start)
    # 5: AI
    sys_msg = SystemMessage(content="System")
    messages = [
        sys_msg,
        HumanMessage(content="H1", id="h1"),
        AIMessage(content="AI1 with tool call", id="ai1", tool_calls=[{"name": "tool", "args": {}, "id": "tc1"}]),
        ToolMessage(content="Tool result", id="t1", tool_call_id="tc1"),
        AIMessage(content="AI1 response", id="ai2"),
        HumanMessage(content="H2", id="h2"),
        AIMessage(content="AI2 response", id="ai3"),
    ]
    
    # We patch MAX_HISTORY_TURNS to 2 for this test
    with patch("src.domain.conversation_service.MAX_HISTORY_TURNS", 2):
        truncated = truncate_history(messages)
        
    # Boundary should snap forwards from index 2 (ToolMessage) to index 4 (HumanMessage "h2")
    # Kept conversation messages should be only "h2" and "ai3"
    assert len(truncated) == 3
    assert truncated[0] == sys_msg
    assert truncated[1].id == "h2"
    assert truncated[2].id == "ai3"


@patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", False)
def test_manage_memory_hard_truncation_snapping():
    from langchain_core.messages import ToolMessage
    # Setup test messages: 6 convo messages. MAX_HISTORY_TURNS = 2 (max_messages = 4)
    # 0: Human
    # 1: AI (with tool call)
    # 2: Tool (tool response) - tentative boundary
    # 3: AI (final turn response)
    # 4: Human (next turn start)
    # 5: AI
    messages = [
        HumanMessage(content="H1", id="h1"),
        AIMessage(content="AI1 with tool call", id="ai1", tool_calls=[{"name": "tool", "args": {}, "id": "tc1"}]),
        ToolMessage(content="Tool result", id="t1", tool_call_id="tc1"),
        AIMessage(content="AI1 response", id="ai2"),
        HumanMessage(content="H2", id="h2"),
        AIMessage(content="AI2 response", id="ai3"),
    ]
    
    with patch("src.rag.agent.MAX_HISTORY_TURNS", 2):
        result = _manage_memory({"messages": messages})
        
    # Snaps forward to index 4 (HumanMessage "h2").
    # Therefore, indices 0, 1, 2, and 3 must be removed.
    removed = result["messages"]
    assert len(removed) == 4
    remove_ids = [m.id for m in removed]
    assert "h1" in remove_ids
    assert "ai1" in remove_ids
    assert "t1" in remove_ids
    assert "ai2" in remove_ids


@patch("src.rag.agent._llm")
def test_summarize_history_snapping(mock_llm):
    from langchain_core.messages import ToolMessage, SystemMessage
    from src.rag.agent import summarize_history
    mock_llm.invoke.return_value = MagicMock(content="Summary of first turn.")

    # Setup test messages: 6 convo messages. MAX_HISTORY_TURNS = 2 (max_messages = 4)
    messages = [
        HumanMessage(content="H1", id="h1"),
        AIMessage(content="AI1 with tool call", id="ai1", tool_calls=[{"name": "tool", "args": {}, "id": "tc1"}]),
        ToolMessage(content="Tool result", id="t1", tool_call_id="tc1"),
        AIMessage(content="AI1 response", id="ai2"),
        HumanMessage(content="H2", id="h2"),
        AIMessage(content="AI2 response", id="ai3"),
    ]

    mock_agent = MagicMock()
    mock_agent.get_state.return_value = MagicMock(values={"messages": messages})

    with patch("src.rag.agent.MAX_HISTORY_TURNS", 2), \
         patch("src.rag.agent.CONVERSATION_SUMMARIZATION_ENABLED", True):
        summarize_history(mock_agent, "thread-1")

    mock_agent.get_state.assert_called_once_with({"configurable": {"thread_id": "thread-1"}})
    mock_agent.update_state.assert_called_once()
    
    update_args = mock_agent.update_state.call_args
    assert update_args[0][0] == {"configurable": {"thread_id": "thread-1"}}
    assert update_args[1]["as_node"] == "manage_memory"
    
    returned_messages = update_args[0][1]["messages"]
    assert len(returned_messages) == 5  # 4 RemoveMessages + 1 SystemMessage (summary)
    
    remove_ids = [m.id for m in returned_messages if isinstance(m, RemoveMessage)]
    assert len(remove_ids) == 4
    assert "h1" in remove_ids
    assert "ai1" in remove_ids
    assert "t1" in remove_ids
    assert "ai2" in remove_ids

    summary_msg = [m for m in returned_messages if isinstance(m, SystemMessage)][0]
    assert summary_msg.id == "conversation_summary"
    assert "Summary of first turn" in summary_msg.content

    # Verify LLM prompt contains the snapped messages and not the kept ones
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert "H1" in prompt_sent
    assert "AI1 with tool call" in prompt_sent
    assert "Tool result" in prompt_sent
    assert "AI1 response" in prompt_sent
    assert "H2" not in prompt_sent
    assert "AI2 response" not in prompt_sent


def test_system_prompt_decoupling():
    from src.rag.agent import ModelCaller
    from langchain_core.messages import SystemMessage
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Mocked response")
    
    with patch("src.rag.agent._llm_with_tools", mock_llm):
        # 1. Test Static injection at composition time
        caller = ModelCaller(system_prompt_template="STATIC TEMPLATE: {response_language}")
        state = {"messages": [HumanMessage(content="Hola, ¿cómo estás? Me gustaría saber el horario.", id="h1")]}
        caller(state)
        
        # Verify the system message format
        messages_sent = mock_llm.invoke.call_args[0][0]
        assert isinstance(messages_sent[0], SystemMessage)
        assert messages_sent[0].content == "STATIC TEMPLATE: Spanish"
        
        # 2. Test Dynamic Override via RunnableConfig
        config = {"configurable": {"system_prompt_template": "DYNAMIC TEMPLATE: {response_language}"}}
        caller(state, config=config)
        
        messages_sent = mock_llm.invoke.call_args[0][0]
        assert isinstance(messages_sent[0], SystemMessage)
        assert messages_sent[0].content == "DYNAMIC TEMPLATE: Spanish"
