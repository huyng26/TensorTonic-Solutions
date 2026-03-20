import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here

    x = np.asarray(x, dtype= np.float32)
    if rng:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
    mask = (random_values > p).astype(np.float32)
    mask *= 1/ (1 - p)
    x *= mask
    return (x, mask) 
    
    