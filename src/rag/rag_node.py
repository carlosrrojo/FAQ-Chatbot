from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from state import AgentState
from config import MODEL_NAME, DATA_PATH, DB_PATH, COLLECTION

llm = ChatOllama(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model = MODEL_NAME)

prompt = ChatPromptTemplate.from_template("""
You are a helpful customer service agent for our company.
Answer the question using ONLY the context below.
Answer in the same language as the question.
If the answer is not in the context, delegate to human through contact info.

Contact info:
- Phone: 600 000 000
- Email: [EMAIL_ADDRESS]

Context:
{context}

Question: {question}
""")

vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LangChain LCEL chain: retrieve → format → prompt → LLM → parse
rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# — The LangGraph node function —
def rag_node(state: AgentState) -> AgentState:
    query   = state["query"]
    # Retrieve relevant chunks
    chunks  = retriever.invoke(query)
    context = [doc.page_content for doc in chunks]
    # Run the chain
    answer  = rag_chain.invoke(query)
    # Write results back into shared state
    return {"context": context, "answer": answer}