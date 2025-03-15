#!/usr/bin/env python3
"""
ML Model Training Script - Production Ready
Handles training, evaluation, and model export with proper logging and error handling.
"""
import os
import sys
import time
import logging
import argparse
import datetime
from typing import Tuple, Dict, Any

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    TensorBoard, 
    ReduceLROnPlateau
)

from model import build_model
from data_processing import preprocess_data
from config import SAVEMODEL_PATH, MODEL_KERAS, LOGS_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def configure_environment(use_gpu: bool = False, memory_growth: bool = True) -> None:
    """Configure TensorFlow environment settings."""
    # Suppress TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    
    if not use_gpu:
        logger.info("Forcing CPU-only execution")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")
    else:
        try:
            # Configure GPU memory growth to avoid OOM errors
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
                for gpu in gpus:
                    if memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu}")
                    
                    # Optional: Configure memory limit if needed
                    # tf.config.experimental.set_virtual_device_configuration(
                    #    gpu,
                    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                    # )
        except Exception as e:
            logger.warning(f"Error configuring GPUs: {e}")
            logger.info("Falling back to CPU execution")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.config.set_visible_devices([], "GPU")

def create_callbacks(model_version: str) -> list:
    """Create training callbacks."""
    # Ensure directories exist
    os.makedirs(os.path.join(LOGS_PATH, model_version), exist_ok=True)
    checkpoint_path = os.path.join(LOGS_PATH, model_version, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model during training
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, "model-{epoch:02d}-{val_loss:.4f}.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(LOGS_PATH, model_version),
            histogram_freq=1,
            update_freq='epoch'
        ),
        
        # Learning rate scheduler
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    epochs: int = 50,
    **model_params
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train model with proper error handling and performance tracking.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        **model_params: Additional parameters to pass to build_model
        
    Returns:
        Tuple of (trained model, training history)
    """
    start_time = time.time()
    logger.info(f"Starting model training with {X_train.shape[0]} samples")
    logger.info(f"Input shape: {X_train.shape[1]}")
    
    # Create unique model version ID
    model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Build model with input dimension from data
        model = build_model(X_train.shape[1], **model_params)
        model.summary(print_fn=logger.info)
        
        # Train with callbacks
        callbacks = create_callbacks(model_version)
        
        # Use mixed precision for faster training if using compatible GPU
        if tf.config.list_physical_devices('GPU'):
            try:
                # Enable mixed precision
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Using mixed precision training")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
        
        # Fit model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Training time tracking
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return model, history.history
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise
    
def evaluate_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, dataset_name: str = "Test") -> dict:
    """Evaluate model and return metrics."""
    try:
        logger.info(f"Evaluating model on {dataset_name} set ({X.shape[0]} samples)")
        results = model.evaluate(X, y, verbose=0)
        metrics = dict(zip(model.metrics_names, results))
        
        for metric_name, value in metrics.items():
            logger.info(f"{dataset_name} {metric_name}: {value:.4f}")
            
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        raise

def save_model(model: tf.keras.Model, save_keras: bool = True, save_tf: bool = True) -> dict:
    """Save model in multiple formats with error handling."""
    model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}
    
    try:
        # Create model directories
        os.makedirs(SAVEMODEL_PATH, exist_ok=True)
        versioned_path = os.path.join(SAVEMODEL_PATH, model_version)
        
        # Save in Keras format (for development)
        if save_keras:
            try:
                model.save(MODEL_KERAS)
                saved_paths['keras'] = MODEL_KERAS
                logger.info(f"Model saved in Keras format: {MODEL_KERAS}")
            except Exception as e:
                logger.error(f"Failed to save model in Keras format: {e}", exc_info=True)
        
        # Save in TensorFlow SavedModel format (for deployment)
        if save_tf:
            try:
                model.export(versioned_path)
                saved_paths['savedmodel'] = versioned_path
                logger.info(f"Model exported in TensorFlow SavedModel format: {versioned_path}")
                
                # Optionally save metadata about the model
                with open(os.path.join(versioned_path, "model_metadata.txt"), "w") as f:
                    f.write(f"Model version: {model_version}\n")
                    f.write(f"Created: {datetime.datetime.now().isoformat()}\n")
                    f.write(f"TensorFlow version: {tf.__version__}\n")
            except Exception as e:
                logger.error(f"Failed to export model in SavedModel format: {e}", exc_info=True)
        
        return saved_paths
    
    except Exception as e:
        logger.error(f"Error saving model: {e}", exc_info=True)
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and export TensorFlow model')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden-layers', type=str, default='128,64', 
                       help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Configure environment
    configure_environment(use_gpu=args.use_gpu)
    
    try:
        # Process arguments
        hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        X_train, X_test, y_train, y_test = preprocess_data()
        
        # Train model
        model, history = train_model(
            X_train, y_train, 
            X_test, y_test,
            batch_size=args.batch_size,
            epochs=args.epochs,
            hidden_layers=hidden_layers,
            dropout_rate=args.dropout,
            learning_rate=args.learning_rate
        )
        
        # Evaluate model
        train_metrics = evaluate_model(model, X_train, y_train, "Train")
        test_metrics = evaluate_model(model, X_test, y_test, "Test")
        
        # Save model
        saved_paths = save_model(model)
        
        logger.info("Training pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())