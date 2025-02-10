import tensorflow as tf
from keras.export import TFSMLayer  # ✅ Correct import for Keras 3+
import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Ensure model path is correct
SAVEMODEL_PATH = os.getenv("SAVEMODEL_PATH", "/app/models/disease_prediction")

# Verify model path exists before loading
if not os.path.exists(SAVEMODEL_PATH):
    raise RuntimeError(f"Model directory does not exist: {SAVEMODEL_PATH}")

# Load model using TensorFlow's TFSMLayer for Keras 3 compatibility
model = TFSMLayer(SAVEMODEL_PATH, call_endpoint="serving_default")

print(f"✅ Model loaded successfully from {SAVEMODEL_PATH}")

# FastAPI App
app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
async def predict(data: InputData):
    input_array = np.array([data.features], dtype=np.float32)
    predictions = model(input_array)
    predicted_label = int(predictions[0, 0] > 0.5)
    return {"prediction": predicted_label, "raw_output": float(predictions[0, 0])}
