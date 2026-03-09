import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    params = {key: np.asarray(value) for key, value in params.items()}
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)
    z_t = _sigmoid(x.dot(params['Wz']) + h_prev.dot(params['Uz']) + params['bz'])
    r_t = _sigmoid(x.dot(params['Wr']) + h_prev.dot(params['Ur']) + params['br'])
    candidate_h = np.tanh(x.dot(params['Wh']) + (r_t * h_prev).dot(params['Uh']) + params['bh'])
    h_t = (1 - z_t) * h_prev + z_t * candidate_h
    return h_t