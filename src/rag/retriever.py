from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.rag.tools import get_weather, export_to_google_calendar, retrieve_documents



db_path = "data/chroma_db"

embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = Chroma(collection_name="recursive_espazo_nature",
                         embedding_function=embeddings,
                         persist_directory=db_path)
retriever = vectorstore.as_retriever()

retriever_tool = retrieve_documents
all_tools = [retriever_tool, get_weather, export_to_google_calendar]

response_model = ChatOllama(model="llama3.1")


def generate_query_or_respond(state: MessagesState):
    """Generate a query or respond to the user."""
    sys_msg = {"role": "system", "content": """
    You have access to a set of tools that can be used to retrieve information from a document.
    Only use it when it is necessary to answer the question.
    """}
    messages = [sys_msg] + state["messages"]
    response = (
        response_model
        .bind_tools(all_tools).invoke(messages)
    )
    return {"messages": [response]}


#test_input = {"messages": [{"role": "user", "content": "hello!"}]}
#generate_query_or_respond(test_input)["messages"][-1].pretty_print()


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = ChatOllama(model="llama3.1")

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    # Find the most recent HumanMessage
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or msg.type == "human":
            question = msg.content
            break
            
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    
    # Check how many times we've tried to retrieve in the current turn
    retrieve_count = 0
    for msg in reversed(state["messages"]):
        # Stop counting once we hit the original user message for this turn
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
            # If it's the original user message, it won't have the 'is_rewrite' flag
            if not getattr(msg, "additional_kwargs", {}).get("is_rewrite", False):
                break
                
        if (hasattr(msg, 'name') and msg.name == "retrieve_documents") or (isinstance(msg, dict) and msg.get("name") == "retrieve_documents"):
            retrieve_count += 1

    if score == "yes" or retrieve_count >= 2:
        return "generate_answer"
    else:
        return "rewrite_question"

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "CRITICAL: Do NOT translate proper nouns or keywords into another language. Keep them in their original language.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human") or (isinstance(msg, dict) and msg.get("role") == "user"):
            question = msg.content if not isinstance(msg, dict) else msg["content"]
            break
            
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content, additional_kwargs={"is_rewrite": True})]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = ""
    for msg in reversed(state["messages"]):
         if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human") or (isinstance(msg, dict) and msg.get("role") == "user"):
            question = msg.content if not isinstance(msg, dict) else msg["content"]
            break

    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


#=================
#= WORKFLOW =
#=================

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode(all_tools))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
def grade_or_end(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last tool called was our document retriever, we grade it.
    if last_message.name == "retrieve_documents":
        return grade_documents(state)
    # Otherwise, for other tools like weather or calendar, directly generate an answer based on their output.
    return "generate_answer"

workflow.add_conditional_edges(
    "retrieve",
    grade_or_end,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        ):
            for node, update in chunk.items():
                print(f"Update from node {node}:")
                update["messages"][-1].pretty_print()
                print("\n\n")