import pandas as pd
import tensorflow as tf
from model import build_model
from data_processing import preprocess_data

# Train test datasets
X_train, X_test, y_train, y_test, _ = preprocess_data()

# Build model
model = build_model(X_train.shape[1])

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=(X_test, y_test))
