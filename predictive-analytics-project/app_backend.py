# app_backend.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Predictive Analytics AI Pipeline",
    description="FastAPI backend serving a healthcare cost prediction model."
)

# Core validation schema
class PredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the individual")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    smoker: int = Field(..., ge=0, le=1, description="0 for Non-Smoker, 1 for Smoker")

# Load model artifacts safely
if os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl"):
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    model, scaler = None, None

@app.get("/")
def home():
    return {"status": "Online", "message": "Healthcare Prediction API is running. Go to /docs for Swagger UI."}

@app.post("/predict")
def predict_charges(data: PredictionInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model files missing. Run train_model.py first.")
    
    try:
        # Convert incoming JSON data into NumPy array
        input_array = np.array([[data.age, data.bmi, data.smoker]])
        
        # Transform data using the exact training scaler
        scaled_array = scaler.transform(input_array)
        
        # Generate prediction
        prediction = model.predict(scaled_array)[0]
        
        return {
            "status": "success",
            "predicted_charges_usd": round(float(prediction), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))