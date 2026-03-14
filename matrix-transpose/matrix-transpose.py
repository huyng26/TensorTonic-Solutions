import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    M ,N = len(A), len(A[0])
    res = [[] for _ in range(N)]
    for j in range(N):
        for i in range(M):
            res[j].append(A[i][j])
    return np.asarray(res)
    
    
