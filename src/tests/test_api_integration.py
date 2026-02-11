from fastapi.testclient import TestClient
from src.api.main import app
import os
from unittest.mock import patch

client = TestClient(app)

# Set env vars for testing
os.environ["VERIFY_TOKEN"] = "MySecretToken"
os.environ["WHATSAPP_API_TOKEN"] = "mock_token"
os.environ["WHATSAPP_PHONE_NUMBER_ID"] = "mock_id"
os.environ["INSTAGRAM_ACCESS_TOKEN"] = "mock_token"

def test_whatsapp_verification():
    # Test correct token
    response = client.get("/whatsapp/webhook", params={
        "hub.mode": "subscribe",
        "hub.verify_token": "MySecretToken",
        "hub.challenge": "12345"
    })
    assert response.status_code == 200
    assert response.text == "12345"

    # Test incorrect token
    response = client.get("/whatsapp/webhook", params={
        "hub.mode": "subscribe",
        "hub.verify_token": "WrongToken",
        "hub.challenge": "12345"
    })
    assert response.status_code == 403

def test_instagram_verification():
    # Test correct token
    response = client.get("/instagram/webhook", params={
        "hub.mode": "subscribe",
        "hub.verify_token": "MySecretToken",
        "hub.challenge": "67890"
    })
    assert response.status_code == 200
    assert response.text == "67890"

@patch("src.api.whatsapp.ask_question")
@patch("src.api.whatsapp.requests.post")
def test_whatsapp_message_handling(mock_post, mock_ask):
    mock_ask.return_value = "This is a mock answer."
    mock_post.return_value.status_code = 200

    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "123456789",
                        "type": "text",
                        "text": {"body": "Hello bot"}
                    }]
                }
            }]
        }]
    }

    response = client.post("/whatsapp/webhook", json=payload)
    assert response.status_code == 200

@patch("src.api.instagram.ask_question")
@patch("src.api.instagram.requests.post")
def test_instagram_message_handling(mock_post, mock_ask):
    mock_ask.return_value = "This is a mock answer."
    mock_post.return_value.status_code = 200

    payload = {
        "object": "instagram",
        "entry": [{
            "messaging": [{
                "sender": {"id": "987654321"},
                "message": {"text": "Hello insta"}
            }]
        }]
    }

    response = client.post("/instagram/webhook", json=payload)
    assert response.status_code == 200
