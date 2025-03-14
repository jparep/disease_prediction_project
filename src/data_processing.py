import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Add project root to path - consider using relative imports instead
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, SCALER_FILE, TARGET

def load_data():
    """Load raw health data from CSV file."""
    try:
        return pd.read_csv(RAW_DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {RAW_DATA_FILE} not found")
    except Exception as e:
        raise IOError(f"Data loading error: {e}")

def preprocess_data():
    """
    Preprocess health dataset: extract features & target, scale, and split
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    df = load_data()
    
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET}'")
    
    # Split features and target
    features = df.drop(columns=[TARGET])
    target = df[TARGET]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split train/test with stratification if classification problem
    return (*train_test_split(
        features_scaled, target, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True
    ), scaler)

def save_processed_data():
    """Save processed datasets and scaler for model training."""
    X_train, X_test, y_train, y_test, scaler = preprocess_data()
    
    # Ensure output directories exist
    for file_path in [TRAIN_DATA_FILE, TEST_DATA_FILE, SCALER_FILE]:
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    
    # Use more efficient operations
    pd.DataFrame(X_train).assign(target=y_train.values).to_csv(TRAIN_DATA_FILE, index=False)
    pd.DataFrame(X_test).assign(target=y_test.values).to_csv(TEST_DATA_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)
    
    return {
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "feature_count": X_train.shape[1]
    }

if __name__ == "__main__":
    try:
        stats = save_processed_data()
        print(f"✅ Preprocessing complete: {stats['train_samples']} train samples, "
              f"{stats['test_samples']} test samples with {stats['feature_count']} features")
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        sys.exit(1)