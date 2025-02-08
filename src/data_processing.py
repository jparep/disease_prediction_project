import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix ImportError by adding the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RAW_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, SCALER_FILE, TARGET


def load_data():
    """Load raw health data from CSV file."""
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {RAW_DATA_FILE} not found. Ensure the file is present.")
        raise
    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        raise


def preprocess_data():
    """
    Preprocess health dataset:
    - Extract features & target
    - Scale features using StandardScaler
    - Split into train & test sets
    """
    df = load_data()
    if TARGET not in df.columns:
        raise ValueError(f"❌ Dataset must contain '{TARGET}' as a target column.")

    # Split into features and target
    features = df.drop(columns=[TARGET])
    target = df[TARGET]

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    print(f"✅ Processed data: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


def save_to_csv():
    """Save processed train/test data to CSV and persist the scaler."""
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    try:
        # Save train & test data
        train_df = pd.DataFrame(X_train)
        train_df["target"] = y_train.values
        train_df.to_csv(TRAIN_DATA_FILE, index=False)

        test_df = pd.DataFrame(X_test)
        test_df["target"] = y_test.values
        test_df.to_csv(TEST_DATA_FILE, index=False)

        print(f"✅ Train and test datasets saved at:\n   {TRAIN_DATA_FILE}\n   {TEST_DATA_FILE}")

        # Save scaler for inference
        joblib.dump(scaler, SCALER_FILE)
        print(f"✅ Scaler saved at {SCALER_FILE}")

    except Exception as e:
        print(f"❌ Error saving processed data: {e}")
        raise


if __name__ == "__main__":
    print("🔹 Starting Data Processing Pipeline...")
    save_to_csv()
    print("Data Preprocessing Completed Successfully!")
