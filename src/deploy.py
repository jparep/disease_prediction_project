import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import SAVEMODEL_PATH


# Load the model
model = tf.keras.models.load_model(SAVEMODEL_PATH)
print(f"Model loaded successfully from {SAVEMODEL_PATH}")

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API", version="0.1", docs_url="/")

# Define request body format
class InputData(BaseModel):
    features: list # List of feature values
    
@app.post("/predict")
async def predict(data: InputData):
    """"API Endpoint to predict heart disease risk."""
    
    # Convert input data to NumPy array
    input_array = np.array([data.features], dtype=np.float32)
    
    # Make predcitions
    predictions = model.predict(input_array)
    predicted_label = int(predictions[0, 0] > 0.5) # Convert probability to binary label
    return {"prediction": predicted_label, "raw_output": float(predictions[0, 0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # Run API on port 8000
    
    