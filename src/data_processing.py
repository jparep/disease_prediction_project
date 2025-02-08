import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix ImportError by adding the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RAW_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, SCALER_FILE

def load_data():
    """Load raw health data from CSV file."""
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        return df
    except FileNotFoundError:
        print(f"Error: {RAW_DATA_FILE} not found. Ensure the file is present.")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise


def preprocess_data(df):
    """
    Preprocess health dataset:
    - Extract features & target
    - Scale features using StandardScaler
    - Split into train & test sets
    """
    if "heart_disease" not in df.columns:
        raise ValueError("Dataset must contain 'heart_disease' as a target column.")

    features = df.drop(columns=["heart_disease"])
    target = df["heart_disease"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler


def save_to_csv(X_train, X_test, scaler):
    """Save processed train/test data to CSV and persist the scaler."""
    try:
        # Save processed data
        pd.DataFrame(X_train).to_csv(TRAIN_DATA_FILE, index=False)
        pd.DataFrame(X_test).to_csv(TEST_DATA_FILE, index=False)

        # Save scaler for future inference
        joblib.dump(scaler, SCALER_FILE)
        print(f"Scaler saved at {SCALER_FILE}")

    except Exception as e:
        print(f"Error saving processed data: {e}")
        raise


if __name__ == "__main__":
    print("🔹 Starting Data Processing Pipeline...")

    # Load raw dataset
    df = load_data()
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"Processed data: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")

    # Save processed data
    save_to_csv(X_train, X_test, scaler)
    print("Data Preprocessing Completed Successfully!")
