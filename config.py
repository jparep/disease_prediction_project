import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# File paths
RAW_DATA_FILE = os.path.join(DATA_PATH, "raw_data.csv")
TRAIN_DATA_FILE = os.path.join(DATA_PATH, "train_data.csv")
TEST_DATA_FILE = os.path.join(DATA_PATH, "test_data.csv")
SCALER_FILE = os.path.join(MODEL_PATH, "scaler.joblib")
MODEL_FILE = os.path.join(MODEL_PATH, "model.joblib")
