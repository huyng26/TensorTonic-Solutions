import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    X = np.asarray(X)
    y = np.asarray(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X.dot(w) + b)
        grad_w = np.dot(X.T, p - y) / X.shape[0]
        grad_b = np.mean(p - y)
        w = w - lr * grad_w 
        b=  b - lr * grad_b
    
    return (w, b)