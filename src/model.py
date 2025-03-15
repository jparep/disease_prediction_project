import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from typing import List, Union

def build_model(
    input_shape: int,
    hidden_layers: List[int] = [128, 64],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    activation: str = "relu",
    output_activation: str = "sigmoid",
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy", tf.keras.metrics.AUC()],
    use_batch_norm: bool = True
) -> Model:
    """
    Build an optimized feedforward neural network with configurable architecture.
    
    Args:
        input_shape: Number of input features
        hidden_layers: List of neurons for each hidden layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for Adam optimizer
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        metrics: List of metrics to track during training
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(input_shape,), dtype=tf.float32)
    
    # First layer - special case to handle the input
    x = inputs
    
    # Build hidden layers dynamically
    for units in hidden_layers:
        x = Dense(
            units=units,
            activation=activation,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
            
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(
        units=1, 
        activation=output_activation,
        kernel_initializer="glorot_normal"  # Better for sigmoid activation
    )(x)
    
    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use AMSGrad for better convergence
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=metrics
    )
    
    return model

# Example usage:
# model = build_model(
#     input_shape=10,
#     hidden_layers=[128, 64, 32],
#     dropout_rate=0.4,
#     learning_rate=5e-4
# )