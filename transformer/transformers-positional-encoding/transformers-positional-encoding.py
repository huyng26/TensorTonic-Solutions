import numpy as np
import math
def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    base = 10000.0
    pe = np.zeros((seq_length, d_model), dtype=float)
    # [seq_len, 1]
    pos = np.arange(0, seq_length, dtype=float)[:, np.newaxis] 
    # [1, ceil(d_model/2)]
    # div_term =  1 / ( base ** (2 * np.arange(0, math.ceil(d_model / 2), dtype=float) / d_model) )
    # div_term = div_term[np.newaxis, :]
    div_term = np.exp(2 * -np.arange(0, d_model // 2, dtype=float) * math.log(base) /  d_model)
    div_term = div_term[np.newaxis, :]
    #broadcast
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe 