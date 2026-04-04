from langgraph.graph import StateGraph, END
from state import AgentState
from rag_node import rag_node
from langchain_core.messages import HumanMessage


def supervisor_node(state: AgentState) -> AgentState:
    # Classify intent → "rag" | "weather" | "calendar"
    last_msg = state["messages"][-1].content
    return {"query": last_msg, "route": classify_intent(last_msg)}

def route_decision(state: AgentState) -> str:
    return state["route"]   # "rag" | "weather" | "calendar"

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("rag",        rag_node)
#graph.add_node("weather",   weather_node)   # your other agents
#graph.add_node("calendar",  calendar_node)

# Entry point
graph.set_entry_point("supervisor")

# Conditional routing from supervisor
graph.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "rag":      "rag"
    }
)

# Each agent terminates after responding
graph.add_edge("rag",      END)
#graph.add_edge("weather",  END)
#graph.add_edge("calendar", END)

app = graph.compile()

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)],
        "query": "", "context": [], "answer": "", "route": "",
    })
    print("Bot: ", result["answer"])