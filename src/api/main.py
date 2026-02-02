from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.rag.chatbot import ask_question
from src.api.whatsapp import router as whatsapp_router
from src.api.instagram import router as instagram_router
from dotenv import load_dotenv
import logging
import os

# Load Environment Variables
load_dotenv()

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot")

# Include Routers
app.include_router(whatsapp_router)
app.include_router(instagram_router)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest"
    language: str = "Auto"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Simple endpoint for direct testing (Postman/Curl).
    """
    try:
        logger.info(f"Received message from {request.user_id}: {request.message} (Lang: {request.language})")
        answer = ask_question(request.message, request.language)
        logger.info(f"Generated answer: {answer}")
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
