import os
import datetime
import tensorflow as tf
from model import build_model
from data_processing import preprocess_data
from config import SAVEMODEL_PATH, MODEL_KERAS

# Suppress TensorFlow CUDA, cuDNN, and CPU optimization warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all messages, 1 = INFO, 2 = WARNINGS, 3 = ERRORS only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only execution

# Disable all GPUs and force CPU usage
tf.config.set_visible_devices([], "GPU")

# Model versioning for industrial deployment
MODEL_VERSION = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVEMODEL_VERSIONED = os.path.join(SAVEMODEL_PATH, MODEL_VERSION)

# Load train-test datasets
X_train, X_test, y_train, y_test, _ = preprocess_data()

# Build model
model = build_model(X_train.shape[1])

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Print final training accuracy & loss
final_train_loss, final_train_acc = model.evaluate(X_train, y_train, verbose=0)
final_test_loss, final_test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Train Accuracy: {final_train_acc:.4f}, Loss: {final_train_loss:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.4f}, Loss: {final_test_loss:.4f}")

# Save model in `.keras` format (for development & testing)
try:
    model.save(MODEL_KERAS)
    print(f"Model saved successfully in .keras format at {MODEL_KERAS}")
except Exception as e:
    print(f"Error saving model in .keras format: {e}")

# Save model in TensorFlow SavedModel format (for deployment)
try:
    model.save(SAVEMODEL_VERSIONED)
    print(f"Model saved successfully in TensorFlow SavedModel format at {SAVEMODEL_VERSIONED}")
except Exception as e:
    print(f"Error saving model in SavedModel format: {e}")
