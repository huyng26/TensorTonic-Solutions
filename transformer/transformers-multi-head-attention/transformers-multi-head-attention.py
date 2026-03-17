import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
     # Your code here
    q = np.dot(Q, W_q)
    k = np.dot(K, W_k)
    v = np.dot(V, W_v)
    batch, seq_len, d_model = Q.shape
    d_heads = d_model // num_heads
    # q = q.reshape(batch, seq_len, num_heads, d_heads)
    # print(q.shape)
    q = q.reshape(batch, num_heads, seq_len, d_heads)
    k = k.reshape(batch, num_heads, seq_len, d_heads)
    v = v.reshape(batch, num_heads, seq_len, d_heads)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_heads)
    scores = softmax(scores, axis=-1)
    attention = np.matmul(scores, v)
    attention = attention.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    out = np.dot(attention, W_o)
    return out 
    
    