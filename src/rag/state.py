from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # Conversation history — messages accumulate via operator.add
    messages:  Annotated[List[BaseMessage], operator.add]
    # The user's latest question, extracted by supervisor
    query:     str
    # Chunks retrieved from the vector store
    context:   List[str]
    # Final answer produced by the RAG node
    answer:    str
    # Which agent the supervisor last routed to
    route:     str