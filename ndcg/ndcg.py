import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    dcgk_scores = relevance_scores[:k]
    idcgk_scores = sorted(relevance_scores, reverse = True)[:k]
    dcg_k = 0
    for i, rel in enumerate(dcgk_scores, start = 1):
        dcg_k += (2**rel -1) / math.log2(i +1)
    
    idcg_k = 0 
    for i, rel in enumerate(idcgk_scores, start = 1):
        idcg_k += (2**rel -1) / math.log2(i +1)
    
    if idcg_k == 0:
        return 0.0 
    
    return dcg_k / idcg_k