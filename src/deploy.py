import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from keras.export import TFSMLayer  # Keras 3+ model loading

# Ensure model path is set correctly
MODEL_PATH = Path(os.getenv("SAVEMODEL_PATH", "/app/models/disease_prediction"))

# Verify model existence
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model directory does not exist: {MODEL_PATH}")

# Load model using TensorFlow's TFSMLayer for Keras 3+ compatibility
try:
    model = TFSMLayer(str(MODEL_PATH), call_endpoint="serving_default")
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f" Error loading model: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="Disease Prediction API", version="1.0")

# Enable CORS (Optional - useful for frontend apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class InputData(BaseModel):
    features: list[float] = Field(..., example=[0.1, 0.5, 1.3, -0.7, 2.1])  # Explicit typing

@app.post("/predict", summary="Predict disease likelihood")
async def predict(data: InputData):
    """Make a prediction based on input features."""
    try:
        input_array = np.array([data.features], dtype=np.float32)
        predictions = model.call(input_array)  # Use .call() explicitly
        predicted_label = int(predictions.numpy()[0, 0] > 0.5)
        return {
            "prediction": predicted_label,
            "raw_output": float(predictions.numpy()[0, 0]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
