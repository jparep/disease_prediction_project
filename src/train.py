import os
# Suppress TensorFlow CUDA, cuDNN, and CPU optimization warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all messages, 1 = INFO, 2 = WARNINGS, 3 = ERRORS only
# Force CPU-only execution (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# Disable all GPUs and force CPU usage
tf.config.set_visible_devices([], 'GPU')

from model import build_model
from data_processing import preprocess_data
from config import SAVEMODEL_PATH, MODEL_KERAS

# Load train-test datasets
X_train, X_test, y_train, y_test, _ = preprocess_data()

# Build model
model = build_model(X_train.shape[1])

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save model in `.keras` format (For development & testing)
model.save(MODEL_KERAS)  
print(f"Model saved successfully in .keras format at {MODEL_KERAS}")

# Save model in TensorFlow SavedModel format (For deployment)
model.save(SAVEMODEL_PATH, save_format="tf")
print(f"Model saved successfully in TensorFlow SavedModel format at {SAVEMODEL_PATH}")
