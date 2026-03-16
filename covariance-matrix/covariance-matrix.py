import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.shape[0] < 2 or len(X.shape) != 2:
        return None

    N, D = X.shape
    mean = np.mean(X, axis=0)
    X_normalized = X - mean 
    covariance_matrix = 1 / (N-1) * np.dot(X_normalized.T, X_normalized)
    return covariance_matrix