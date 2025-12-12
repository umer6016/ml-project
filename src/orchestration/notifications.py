import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").strip()

def notify_discord(message: str) -> bool:
    """Sends a notification to Discord. Returns True if successful."""
    if not WEBHOOK_URL:
        print("Warning: WEBHOOK_URL not set. Skipping notification.")
        return False
    
    data = {"content": message}
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send notification: {e}")
        if 'response' in locals() and response is not None:
             print(f"Response status: {response.status_code}")
             print(f"Response body: {response.text}")
        return False
