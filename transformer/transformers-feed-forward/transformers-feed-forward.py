import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    hidden_1 = np.dot(x, W1) + b1 
    hidden_1 = np.maximum(0, hidden_1)
    hidden_2 = np.dot(hidden_1, W2) + b2 
    return hidden_2