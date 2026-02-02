import os
import logging
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

async def verify_webhook(request: Request):
    """
    Common logic to verify the webhook for WhatsApp and Instagram.
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge.
    """
    verify_token = os.getenv("VERIFY_TOKEN")
    
    # query_params is a dict-like object
    params = request.query_params
    
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == verify_token:
            logger.info("Webhook verified successfully!")
            # The challenge is an integer usually, but must be returned as plain text int
            return int(challenge)
        else:
            logger.warning("Webhook verification failed. Token mismatch.")
            raise HTTPException(status_code=403, detail="Verification failed")
    
    # If parameters are missing, it might not be a verification request
    # But usually this endpoint is strictly for verification (GET) or receiving data (POST)
    # So if it's GET without params, it's an error.
    raise HTTPException(status_code=400, detail="Missing parameters")
