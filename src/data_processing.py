import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from config import RAW_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, SCALER_FILE 

# Load data
def load_data()
    """Load data file from file path"""
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        raise
    
# Data Preprocessing: Scaling & Splitting
def scale_data(df, features, target):
    """Preprocess data and save processed data and scaler for future use"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def save_to_csv(X_train, X_test, scaler):
    """"Save split data into to csv files"""
    # Save processed data
    pd.DataFrame(X_train).to_csv(TRAIN_DATA_FILE, index=False)
    pd.DataFrame(X_test).to_csv(TEST_DATA_FILE, index=False)

    # Save scaler for future use
    with open(SCALER_FILE, "wb") as f:
        joblib.dump(scaler, f)

if __name__ == "__main__":
    
    # Load data
    df = load_data()
    
    # Split data into features and target
    if "heart_disease" in df.columns:
        features = df.drop(columns=["heart_disease"])
        target = df["heart_disease"]
    else:
        print
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = scale_data(df, features, target)
    
    # Save processed data
    save_to_csv(X_train, X_test, scaler)