from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag.ingest import DATA_PATH
from src.rag.watcher import start_watcher
from src.api.whatsapp import WhatsAppClient
from dotenv import load_dotenv
import logging
import os
from flask import Flask, request, jsonify
from src.rag.rag_as_agent import generate_reply

# Load Environment Variables
load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global Observer
observer = None

# ── Clients ──────────────────────────────────────────────────────────────────
whatsapp = WhatsAppClient(
    phone_number_id=os.environ["WA_PHONE_NUMBER_ID"],
    access_token=os.environ["META_ACCESS_TOKEN"],
)

"""instagram = InstagramClient(
    page_id=os.environ["IG_PAGE_ID"],
    access_token=os.environ["META_ACCESS_TOKEN"],
)"""

VERIFY_TOKEN = os.environ["WEBHOOK_VERIFY_TOKEN"]

# ── Webhook verification (GET) ────────────────────────────────────────────────
@app.get("/webhook")
def verify_webhook():
    """
    Meta sends a GET request to verify your webhook endpoint.
    Must return hub.challenge if the token matches.
    """
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return challenge, 200

    logger.warning("Webhook verification failed. Check your WEBHOOK_VERIFY_TOKEN.")
    return "Forbidden", 403

# ── Incoming events (POST) ────────────────────────────────────────────────────
@app.post("/webhook")
def handle_webhook():
    """
    Receives all incoming events from Meta (WhatsApp + Instagram).
    Must respond with 200 quickly; heavy work should be queued in production.
    """
    data = request.get_json(silent=True)
    if not data:
        return "Bad Request", 400

    object_type = data.get("object")

    try:
        if object_type == "whatsapp_business_account":
            _handle_whatsapp(data)

        elif object_type == "instagram":
            _handle_instagram(data)

        else:
            logger.warning("Unknown object type: %s", object_type)

    except Exception as exc:
        # Always return 200 so Meta doesn't retry; log the error internally
        logger.exception("Error processing webhook: %s", exc)

    return "EVENT_RECEIVED", 200

# ── WhatsApp dispatcher ───────────────────────────────────────────────────────
def _handle_whatsapp(data: dict):
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])

            for msg in messages:
                msg_type = msg.get("type")
                sender   = msg.get("from")          # E.164 phone number

                if msg_type == "text":
                    user_text = msg["text"]["body"]
                    logger.info("[WA] Message from %s: %s", sender, user_text)

                    reply = generate_reply(
                        platform="whatsapp",
                        user_message=user_text,
                        sender_id=sender,
                    )
                    whatsapp.send_text(to=sender, text=reply)

                elif msg_type == "interactive":
                    # Button / list reply
                    interactive = msg.get("interactive", {})
                    if interactive.get("type") == "button_reply":
                        user_text = interactive["button_reply"]["title"]
                    else:
                        user_text = interactive.get("list_reply", {}).get("title", "")

                    logger.info("[WA] Interactive from %s: %s", sender, user_text)
                    reply = generate_reply("whatsapp", user_text, sender)
                    whatsapp.send_text(to=sender, text=reply)

                else:
                    logger.info("[WA] Unsupported message type '%s' from %s", msg_type, sender)
                    whatsapp.send_text(
                        to=sender,
                        text="Sorry, I can only process text messages at the moment.",
                    )


# ── Instagram dispatcher ──────────────────────────────────────────────────────
def _handle_instagram(data: dict):
    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            msg       = event.get("message", {})

            if not msg or msg.get("is_echo"):
                # Ignore echo events (messages sent BY the page)
                continue

            if "text" in msg:
                user_text = msg["text"]
                logger.info("[IG] DM from %s: %s", sender_id, user_text)

                reply = generate_reply(
                    platform="instagram",
                    user_message=user_text,
                    sender_id=sender_id,
                )
                #instagram.send_text(recipient_id=sender_id, text=reply)

            elif "attachments" in msg:
                logger.info("[IG] Attachment from %s (unsupported)", sender_id)
                """instagram.send_text(
                    recipient_id=sender_id,
                    text="Thanks for your message! I can only read text for now. How can I help you?",
                )"""



# WATCHDOG
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

"""
app = FastAPI(title="Espazo Nature Chatbot", lifespan=lifespan)

# Include Routers
app.include_router(whatsapp_router)
app.include_router(instagram_router)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest"
    language: str = "Auto"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    Simple endpoint for direct testing (Postman/Curl).
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
"""

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="localhost", port=port, debug=False)