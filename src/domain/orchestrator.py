# src/domain/orchestrator.py - FR-AGT-01..04
import logging
from langchain_core.messages import HumanMessage
from src.domain.models import ChatRequest, ChatResponse
from src.domain.ports import IMemoryStore
from src.telemetry import timed
from src.rag.agent import build_graph

logger = logging.getLogger(__name__)

# Out-of-scope fallback messages, keyed by detected language
_OOS_RESPONSES = {
    "es": (
        "Lo siento, esa pregunta está fuera del ámbito de información que puedo "
        "proporcionar sobre Espazo Nature. Para consultas más específicas, puedes "
        "contactarnos directamente. ¿Puedo ayudarte con algo relacionado con nuestros "
        "alojamientos, servicios o entorno?"
    ),
    "en": (
        "I'm sorry, that question falls outside the scope of information I can provide "
        "about Espazo Nature. For more specific enquiries, please contact us directly. "
        "Can I help you with something related to our accommodation, services, or surroundings?"
    ),
}


class RAGOrchestrator:
    """
    Domain facade over the LangGraph agentic graph.
    Responsibilities:
    - Accept ChatRequest from transport layer
    - Invoke the compiled LangGraph agent
    - Apply history truncation policy (FR-MEM-03)
    - Apply out-of-scope detection (FR-AGT-04)
    - Return ChatResponse
    """

    def __init__(self, memory_store: IMemoryStore) -> None:
        self._memory = memory_store
        self._agent = self._build_agent()

    def _build_agent(self):
        """
        Builds and compiles the LangGraph StateGraph.
        This method consolidates the graph construction currently in agent.py.
        """
        if self._memory is None:
            return build_graph(skip_memory=True)
        return build_graph(checkpointer=self._memory.get_checkpointer())

    @timed("orchestrator.generate_reply")
    def generate_reply(self, request: ChatRequest) -> ChatResponse:
        config = {"configurable": {"thread_id": request.sender_id}}
        input_state = {"messages": [HumanMessage(content=request.text)]}

        result = self._agent.invoke(input_state, config=config)
        reply_text = result["messages"][-1].content

        if self._memory is not None:
            try:
                self._memory.touch_session(request.sender_id)
            except Exception as e:
                logger.error("Failed to touch session for %s: %s", request.sender_id, e)

        return ChatResponse(
            text=reply_text,
            sender_id=request.sender_id,
            platform=request.platform,
        )