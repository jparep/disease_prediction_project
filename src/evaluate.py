import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from data_processing import preprocess_data
from config import MODEL_FILE

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

# Disable GPU usage (force CPU-only execution)
tf.config.set_visible_devices([], 'GPU')

# Load preprocessed data
X_train, X_test, y_train, y_test, _ = preprocess_data()

# Load model (ensure it is in `.keras` format)
model = tf.keras.models.load_model(MODEL_FILE)

# Predict and threshold outputs
y_pred = (np.squeeze(model.predict(X_test, batch_size=32, verbose=0)) > 0.5).astype(int)

# Display classification report
print("\n🔹 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# Calculate accuracy
accuracy = np.mean(y_pred == y_test, dtype=np.float32)
print(f"\n Model Accuracy: {accuracy:.4f}")
