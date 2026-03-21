import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    b, seq_len, d_k = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1))  / math.sqrt(d_k)
    attention = torch.softmax(scores, dim = -1)
    out = torch.matmul(attention, V)
    return out 