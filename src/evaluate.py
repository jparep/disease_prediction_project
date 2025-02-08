import os
# Suppress TensorFlow CUDA, cuDNN, and CPU optimization warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all messages, 1 = INFO, 2 = WARNINGS, 3 = ERRORS only
# Force CPU-only execution (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
# Disable all GPUs and force CPU usage
tf.config.set_visible_devices([], 'GPU')

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

# Print accuracy score
accuracy = (y_pred == y_test).mean()
print(f"Model Accuracy: {accuracy:.4f}")
