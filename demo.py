import requests
import pandas as pd
import json
import random

# Configuration
API_URL = "http://localhost:8000"
DATA_PATH = "data/processed/AAPL_processed.csv"

def run_demo():
    print("Starting Stock Prediction System Demo")
    print("========================================")
    
    # 1. Check API Health
    print("\n1. Checking API Health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print(f"API is Healthy: {response.json()}")
        else:
            print(f"API Error: {response.status_code}")
            return
    except Exception as e:
        print(f"Connection Failed: {e}")
        print("Make sure Docker containers are running!")
        return

    # 2. Load Sample Data
    print(f"\n2. Loading sample data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        # Pick a random row
        sample = df.sample(1).iloc[0]
        
        input_data = {
            "sma_20": float(sample['sma_20']),
            "sma_50": float(sample['sma_50']),
            "rsi": float(sample['rsi']),
            "macd": float(sample['macd'])
        }
        
        print(f"   Selected Sample (Date: {sample.get('timestamp', 'N/A')}):")
        print(json.dumps(input_data, indent=4))
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 3. Predict Price (Regression)
    print("\n3. Requesting Price Prediction (Regression)...")
    try:
        response = requests.post(f"{API_URL}/predict/price", json=input_data)
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: ${result['prediction']:.2f}")
            print(f"   Actual Next Close: ${sample.get('target_price', 'N/A')}")
        else:
            print(f"Request Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # 4. Predict Direction (Classification)
    print("\n4. Requesting Direction Prediction (Classification)...")
    try:
        response = requests.post(f"{API_URL}/predict/direction", json=input_data)
        if response.status_code == 200:
            result = response.json()
            direction = "UP" if result['prediction'] == 1.0 else "DOWN"
            print(f"Prediction: {direction}")
            actual_dir = "UP" if sample.get('target_direction') == 1 else "DOWN"
            print(f"   Actual Direction: {actual_dir}")
        else:
            print(f"Request Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n========================================")
    print("Demo Completed!")

if __name__ == "__main__":
    run_demo()
