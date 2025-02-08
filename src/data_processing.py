import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
FILE_PATH = "data/raw_data.csv"

# Data Preprocessing: Scaling & Splitting
def preprocess_data(df):
    """Preprocess data and save processed data and scaler for future use"""
    features = df.drop(columns=["heart_disease"])
    target = df["heart_disease"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Save processed data
    pd.DataFrame(X_train).to_csv("data/train_data.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/test_data.csv", index=False)

    # Save scaler for future use
    with open("models/scaler.pkl", "wb") as f:
        joblib.dump(scaler, f)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    # Load data
    df = pd.read_csv(FILE_PATH)
    preprocess_data()
