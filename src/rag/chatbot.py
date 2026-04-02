"""
chatbot.py
----------
Thin adapter over SupervisorAgent.

Provides:
  - ask_question(question, language, session_id) for existing callers
  - get_supervisor(session_id) for session-aware callers

Each unique session_id gets its own SupervisorAgent instance with
independent conversation memory. Memory is held in-process (resets
on server restart).
"""

from src.rag.agents import SupervisorAgent

# In-process session registry: session_id → SupervisorAgent
_supervisors: dict[str, SupervisorAgent] = {}


def get_supervisor(session_id: str = "default") -> SupervisorAgent:
    """Return (or create) a SupervisorAgent for the given session."""
    if session_id not in _supervisors:
        _supervisors[session_id] = SupervisorAgent(session_id=session_id)
    return _supervisors[session_id]


def ask_question(question: str, language: str = "Auto", session_id: str = "guest") -> str:
    """
    Ask a question and get an answer string.

    Parameters
    ----------
    question   : the user's message
    language   : kept for API compatibility (routing/LLM handles language automatically)
    session_id : unique identifier for the user session (used for per-user memory)
    """
    supervisor = get_supervisor(session_id)
    result = supervisor.run(question)
    return result["answer"]