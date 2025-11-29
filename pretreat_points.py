import numpy as np

def pretreat_points(pts, orientate=False, normalization=False, rectification_orientation=False):

    if orientate:
        mean = pts.mean(axis=0)
        pts_centered = pts - mean

        # PCA
        C = np.cov(pts_centered.T)
        vals, vecs = np.linalg.eig(C)
        pca1 = vecs[:, np.argmax(vals)]
        pca2 = vecs[:,1]
        pca3 = np.cross(pca1, pca2) + 10**(-6)

        # Rotation
        R = np.column_stack((pca1, pca2, pca3))
        R = R.T 

        pts = pts_centered@ R.T
    
    if normalization:
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        pts = (pts - mean) / std
    
    if rectification_orientation: # Only meaningful if orientated
        # Put the first point "at the top of the image" (as we often start a number)
        first_point = pts[0]
        if first_point[0] < 0: 
            theta = np.pi 
            Rx = np.array([
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]
            ])
            pts = pts @ Rx.T

    return pts

def resample_points(pts, seq_len):
    N = pts.shape[0]
    
    if N == seq_len:
        return pts
    
    orig_positions = np.linspace(0, 1, N)
    target_positions = np.linspace(0, 1, seq_len)
    
    pts_resampled = np.zeros((seq_len, 3))
    for i in range(3):
        pts_resampled[:, i] = np.interp(target_positions, orig_positions, pts[:, i])
    
    return pts_resampled
