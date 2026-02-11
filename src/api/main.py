
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag.chatbot import ask_question
from src.rag.ingest import DATA_PATH
from src.rag.watcher import start_watcher
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

# Global Observer
observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting document watcher on {DATA_PATH}...")
    global observer
    try:
        # Ensure path exists, absolute path
        abs_path = os.path.abspath(DATA_PATH)
        if not os.path.exists(abs_path):
             os.makedirs(abs_path)
             
        observer = start_watcher(abs_path)
    except Exception as e:
        logger.error(f"Failed to start watcher: {e}")
        
    yield
    
    # Shutdown
    if observer:
        logger.info("Stopping document watcher...")
        observer.stop()
        observer.join()

app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

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
