from src.orchestration.notifications import notify_discord
import os
from dotenv import load_dotenv

# Force load .env if available
load_dotenv()

if __name__ == "__main__":
    print("Testing Discord notification...")
    webhook = os.getenv("WEBHOOK_URL")
    if not webhook:
        print("ERROR: WEBHOOK_URL environment variable is NOT set.")
    else:
        print(f"WEBHOOK_URL is set (starts with {webhook[:10]}...)")
        notify_discord("Test notification from Stock Prediction System debugging script")
