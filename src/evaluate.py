import os
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from data_processing import preprocess_data
from config import MODEL_FILE

# Load Model (ensure `.keras` format is used)
model = tf.keras.models.load_model(MODEL_FILE)

# Load train and test data from `preprocess_data()` directly
X_train, X_test, y_train, y_test, _ = preprocess_data()

# Make predictions (convert to flat array)
y_pred = (model.predict(X_test).ravel() > 0.5).astype(int)

# Evaluation report
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))
