import os
import requests
import logging
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from src.api.utils import verify_webhook
from src.rag.chatbot import ask_question

router = APIRouter()
logger = logging.getLogger(__name__)

INSTAGRAM_API_URL = "https://graph.facebook.com/v17.0"

def send_instagram_message(to_user_id: str, message_text: str):
    """
    Sends a text message via Instagram Graph API.
    """
    token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    
    # Note: For IG, we usually post to /me/messages or /{ig_account_id}/messages
    # using the Page Access Token associated with the linked Facebook Page.
    # The 'to' field is the PSID (Page Scoped ID) of the user.
    
    if not token:
        logger.error("Instagram credentials not found in environment variables.")
        return

    # Using 'me' is common if the token is a Page Access Token for the linked page
    url = f"{INSTAGRAM_API_URL}/me/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "recipient": {"id": to_user_id},
        "message": {"text": message_text}
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Instagram message sent to {to_user_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Instagram message: {e}")
        if response is not None:
             logger.error(f"Response: {response.text}")

async def process_instagram_message(body: dict):
    """
    Process the incoming webhook payload.
    """
    try:
        entry = body.get("entry", [])[0]
        # Instagram structure usually has 'messaging' list
        messaging_events = entry.get("messaging", [])
        
        for event in messaging_events:
            if "message" in event and "text" in event["message"]:
                sender_id = event["sender"]["id"]
                message_text = event["message"]["text"]
                
                logger.info(f"Received Instagram message from {sender_id}: {message_text}")
                
                # Ask the Brain
                answer = ask_question(message_text)
                
                # Send Reply
                send_instagram_message(sender_id, answer)
            else:
                logger.info("Received Instagram event without text message.")

    except (IndexError, KeyError) as e:
        logger.error(f"Error parsing Instagram payload: {e}")

@router.get("/instagram/webhook")
async def instagram_verification(request: Request):
    return await verify_webhook(request)

@router.post("/instagram/webhook")
async def instagram_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming Instagram messages.
    """
    body = await request.json()
    
    # Process in background
    background_tasks.add_task(process_instagram_message, body)
    
    return {"status": "received"}
