# Use Python 3.11 Slim as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose the port 8000 for FastAPI
EXPOSE 8000

# Run the FadtAPI application
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]