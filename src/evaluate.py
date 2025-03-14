import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from data_processing import preprocess_data
from config import MODEL_FILE

# Configure environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
tf.config.set_visible_devices([], 'GPU')  # Force CPU execution

def evaluate_model():
    """Evaluate trained model on test data and return performance metrics."""
    # Load data and model
    _, X_test, _, y_test, _ = preprocess_data()
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Make predictions efficiently
    predictions = model.predict(
        X_test, 
        batch_size=64,  # Increased batch size for faster inference
        verbose=0
    )
    
    # Convert predictions to binary classes
    y_pred = (np.squeeze(predictions) > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score'],
        "report": report
    }

if __name__ == "__main__":
    try:
        metrics = evaluate_model()
        
        # Print summary results
        print("\n🔹 Model Evaluation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        # Print detailed classification report
        print("\n🔹 Classification Report:")
        print(classification_report(
            y_test, y_pred, digits=4
        ))
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")