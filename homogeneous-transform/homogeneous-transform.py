import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T)
    points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[np.newaxis, :]
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    result = T @ points.T 
    return result[:3, :].squeeze().T
