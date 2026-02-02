import os
import requests
import logging
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from src.api.utils import verify_webhook
from src.rag.chatbot import ask_question

router = APIRouter()
logger = logging.getLogger(__name__)

WHATSAPP_API_URL = "https://graph.facebook.com/v17.0"

def send_whatsapp_message(to_number: str, message_text: str):
    """
    Sends a text message via WhatsApp Cloud API.
    """
    token = os.getenv("WHATSAPP_API_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    
    if not token or not phone_number_id:
        logger.error("WhatsApp credentials not found in environment variables.")
        return

    url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text}
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"WhatsApp message sent to {to_number}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send WhatsApp message: {e}")
        if response is not None:
             logger.error(f"Response: {response.text}")

async def process_whatsapp_message(body: dict):
    """
    Process the incoming webhook payload.
    """
    try:
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # Check if it's a message
        if "messages" not in value:
            # Could be a status update (sent, delivered, read) - ignore for now
            return

        message = value["messages"][0]
        from_number = message["from"]
        msg_type = message["type"]
        
        # We only handle text messages for now
        if msg_type == "text":
            text_body = message["text"]["body"]
            logger.info(f"Received WhatsApp message from {from_number}: {text_body}")
            
            # Ask the Brain
            answer = ask_question(text_body)
            
            # Send Reply
            send_whatsapp_message(from_number, answer)
        else:
             logger.info(f"Received non-text message type: {msg_type}")
             send_whatsapp_message(from_number, "Sorry, I can only understand text messages for now.")

    except (IndexError, KeyError) as e:
        logger.error(f"Error parsing WhatsApp payload: {e}")

@router.get("/whatsapp/webhook")
async def whatsapp_verification(request: Request):
    return await verify_webhook(request)

@router.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming WhatsApp messages.
    """
    body = await request.json()
    
    # Process in background to return 200 OK quickly to Meta
    background_tasks.add_task(process_whatsapp_message, body)
    
    return {"status": "received"}
