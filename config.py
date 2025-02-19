import os

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Define data file paths
RAW_DATA_FILE = os.path.join(DATA_PATH, "raw_data.csv")
TRAIN_DATA_FILE = os.path.join(DATA_PATH, "train_data.csv")
TEST_DATA_FILE = os.path.join(DATA_PATH, "test_data.csv")
SCALER_FILE = os.path.join(MODEL_PATH, "scaler.joblib")

# Define model file paths
MODEL_H5 = os.path.join(MODEL_PATH, "model.h5")                 # Legacy Keras format
MODEL_KERAS = os.path.join(MODEL_PATH, "model.keras")           # Current Keras format
SAVEMODEL_PATH = os.path.join(MODEL_PATH, "disease_prediction") # SavedModel format (recommended)

# Define target column
TARGET = "heart_disease"
