from langchain.agents.middleware import dynamic_prompt, ModelRequest

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.globals import set_debug
from dotenv import load_dotenv
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain.agents import AgentState

# Configuration
DB_PATH = "data/chroma_db"
MODEL_NAME = "llama3.1"
# Debug mode
set_debug(False)
load_dotenv()

"""
    Create a RAG Agent to retrieve information to help answer a query.
"""
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma(collection_name="espazo_nature",
        embedding_function=embeddings,
        persist_directory=DB_PATH)
    retrieved_docs = vectorstore.similarity_search(query, k=3, filter={
        "$or": [
            {"section": {"$in": [query]}},
            {"keywords": {"$in": [query]}}
        ]
    })

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# 2 STEP CHAIN
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma(collection_name="espazo_nature",
        embedding_function=embeddings,
        persist_directory=DB_PATH)
    last_query = request.state["messages"][-1].text
    retrieved_docs = vectorstore.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a custom service assistant from company Espazo Nature.",
        "Espazo Nature is a company that provides glamping services in Galicia, Spain.",
        "You have access to a tool that retrieves context from a document with information about the company.",
        "Use it to answer the user's question.",
        "If the question is in Spanish, answer in Spanish. If the question is in English, answer in English.",
        f"\n\n{docs_content}"
    )

    return system_message

# TEST MEMORY SUPPORT
class MemorySupport(AgentState):
    user_id: str

class CustomMiddleware(AgentMiddleware):
    state_schema = MemorySupport
    tools = [retrieve_context]

    def before_model(self, state: MemorySupport, runtime) -> dict[str, Any] | None:
        return {"user_id": state.user_id}


if __name__ == "__main__":
    tools = [] # [retrieve_context]
    model = ChatOllama(model=MODEL_NAME)
    prompt = (
        "You have access to a tool that retrieves context from a document.",
        "Use it to answer the user's question.",
        "If the question is in Spanish, answer in Spanish. If the question is in English, answer in English."
    )
    agent = create_agent(
        model,
        tools,
        middleware=[
            prompt_with_context,
            SummarizationMiddleware(
                model, trigger=("tokens", 4000), keep=("messages", 10)
            )
        ]
    )
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()