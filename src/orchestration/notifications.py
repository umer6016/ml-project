import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Sanitize URL: Remove whitespace and replace PTB/Canary domains with standard domain
WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").strip()
WEBHOOK_URL = WEBHOOK_URL.replace("ptb.discord.com", "discord.com").replace("canary.discord.com", "discord.com")

def notify_discord(message: str) -> tuple[bool, str]:
    """Sends a notification to Discord. Returns (Success, status_message)."""
    if not WEBHOOK_URL:
        msg = "Warning: WEBHOOK_URL not set. Skipping notification."
        print(msg)
        return False, msg
    
    data = {"content": message}
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        response.raise_for_status()
        return True, "Notification sent successfully."
    except requests.exceptions.RequestException as e:
        error_msg = f"Request Failed: {e}"
        if 'response' in locals() and response is not None:
             error_msg += f" | Status: {response.status_code} | Body: {response.text}"
        print(error_msg)
        return False, error_msg
