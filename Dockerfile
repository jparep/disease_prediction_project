# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire project, including the SavedModel folder
COPY . .

# Set environment variable for TensorFlow
ENV SAVEMODEL_PATH="/app/models/disease_prediction"

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
