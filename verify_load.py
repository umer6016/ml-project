
import joblib
import os
from pathlib import Path

def test_load():
    # Simulate the logic in main.py
    # We are running this script from ROOT, so we need to construct the path 
    # as if we were in src/api/main.py to test that specific logic, 
    # OR provided we know the structure, just test access to the models dir.
    
    # Let's test the ACTUAL logic we put in main.py.
    # We will assume this script is placed at src/api/debug_load.py to match depth
    # But I will write it to root and adjust logic for testing purposes, 
    # OR just write it to src/api/verify_load.py
    pass

if __name__ == "__main__":
    # We will assume this file is at ROOT/verify_load.py
    # So ROOT is just Path(__file__).parent
    
    ROOT_DIR = Path(__file__).resolve().parent
    models_dir = ROOT_DIR / "models"
    
    print(f"Checking models dir: {models_dir}")
    
    symbol = "AAPL"
    reg_path = models_dir / symbol / "regression_model.pkl"
    
    if reg_path.exists():
        print(f"FOUND: {reg_path}")
        try:
            model = joblib.load(reg_path)
            print("SUCCESS: Model loaded correctly.")
        except Exception as e:
            print(f"FAILURE: Model found but failed to load: {e}")
    else:
        print(f"FAILURE: Model file not found at {reg_path}")
