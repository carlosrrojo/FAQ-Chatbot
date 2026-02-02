from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Configuration
DB_PATH = "data/chroma_db"
MODEL_NAME = "llama3.3"

def get_rag_chain():
    # 1. Initialize Embeddings & Vector Store
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # 2. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. LLM
    llm = ChatOllama(model=MODEL_NAME)

    # 4. Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the user's question based strictly on the provided context.

    Response Language: {language}

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # 5. Create Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def ask_question(question: str, language: str = "Auto"):
    chain = get_rag_chain()
    
    target_lang = language
    if language == "Auto":
        target_lang = "the same language as the question"
        
    response = chain.invoke({"input": question, "language": target_lang})
    return response["answer"]

if __name__ == "__main__":
    # Test Interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = ask_question(user_input)
        print(f"Bot: {answer}\n")
