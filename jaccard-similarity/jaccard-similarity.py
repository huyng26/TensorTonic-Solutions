def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a = set(set_a)
    set_b = set(set_b)
    union = set_a.union(set_b)
    intersect = set_a.intersection(set_b)
    if len(intersect) == 0:
        return 0
    return len(intersect) / len(union)



















































































































    