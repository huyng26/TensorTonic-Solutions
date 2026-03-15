import numpy as np
import math 
def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model), dtype=float)
    # [seq_len, 1]
    pos = np.arange(0, seq_len, dtype=float)[:, np.newaxis] 
    # [1, ceil(d_model/2)]
    div_term =  1 / ( base ** (2 * np.arange(0, math.ceil(d_model / 2), dtype=float) / d_model) )
    #broadcast
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term[:d_model // 2])
    return pe 