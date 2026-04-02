import numpy as np
import math 
def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x)
    y = np.asarray(y)
    residual = x - y 
    residual = residual * residual
    return np.sqrt(np.sum(residual))