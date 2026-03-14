import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not max_len:
        max_len = max([len(row) for row in seqs]) 
    out = []
    for seq in seqs:
        seq = seq[:max_len]
        while len(seq) < max_len:
            seq.append(pad_value)
        out.append(seq)
    return np.asarray(out)