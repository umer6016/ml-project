from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import List

app = FastAPI(title="Stock Prediction API", version="1.0.0")

# Global variables to store models
models = {}

class PredictionInput(BaseModel):
    sma_20: float
    sma_50: float
    rsi: float
    macd: float

class PredictionOutput(BaseModel):
    prediction: float
    model_type: str

@app.on_event("startup")
def load_models():
    """Load models on startup."""
    model_dir = "models"
    try:
        # Load latest models (assuming single symbol for demo or specific path)
        # In a real app, we might load models dynamically based on symbol
        # Here we look for a generic or specific model
        # For demo purposes, we'll try to load 'AAPL' models if they exist, else generic
        
        # Check for AAPL models first
        symbol = "AAPL" 
        reg_path = f"{model_dir}/{symbol}/regression_model.pkl"
        clf_path = f"{model_dir}/{symbol}/classification_model.pkl"
        
        if os.path.exists(reg_path):
            models['regression'] = joblib.load(reg_path)
            print(f"Loaded regression model from {reg_path}")
        
        if os.path.exists(clf_path):
            models['classification'] = joblib.load(clf_path)
            print(f"Loaded classification model from {clf_path}")
            
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.post("/predict/price", response_model=PredictionOutput)
def predict_price(input_data: PredictionInput):
    if 'regression' not in models:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    
    features = [[input_data.sma_20, input_data.sma_50, input_data.rsi, input_data.macd]]
    prediction = models['regression'].predict(features)[0]
    return {"prediction": prediction, "model_type": "regression"}

@app.post("/predict/direction", response_model=PredictionOutput)
def predict_direction(input_data: PredictionInput):
    if 'classification' not in models:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    
    features = [[input_data.sma_20, input_data.sma_50, input_data.rsi, input_data.macd]]
    prediction = models['classification'].predict(features)[0]
    return {"prediction": float(prediction), "model_type": "classification"}

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if 'regression' not in models:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    
    try:
        df = pd.read_csv(file.file)
        required_cols = ['sma_20', 'sma_50', 'rsi', 'macd']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_cols}")
        
        features = df[required_cols]
        predictions = models['regression'].predict(features)
        
        results = df.copy()
        results['predicted_price'] = predictions
        
        # Return as JSON records
        return results.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
