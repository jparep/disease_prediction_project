import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load model once at startup
MODEL_PATH = Path(os.getenv("SAVEMODEL_PATH", "/app/models/disease_prediction"))
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model directory does not exist: {MODEL_PATH}")

try:
    from keras.export import TFSMLayer
    model = TFSMLayer(str(MODEL_PATH), call_endpoint="serving_default")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Initialize FastAPI with metadata
app = FastAPI(
    title="Disease Prediction API", 
    version="1.0",
    description="API for disease prediction from medical features"
)

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Limit to needed methods
    allow_headers=["Content-Type", "Authorization"],
)

class InputData(BaseModel):
    features: list[float] = Field(..., example=[0.1, 0.5, 1.3, -0.7, 2.1])
    
    class Config:
        schema_extra = {
            "example": {"features": [0.1, 0.5, 1.3, -0.7, 2.1]}
        }

@app.post("/predict")
async def predict(data: InputData):
    """Make a prediction based on input features."""
    try:
        # Pre-allocate with correct type
        input_array = np.asarray([data.features], dtype=np.float32)
        
        # Get prediction efficiently
        predictions = model.call(input_array)
        probability = float(predictions.numpy().item())
        predicted_label = int(probability > 0.5)
        
        return {
            "prediction": predicted_label,
            "probability": probability,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))