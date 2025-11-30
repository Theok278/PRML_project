import numpy as np

def pretreat_points(pts, normalization=False):
    if normalization:
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        pts = (pts - mean) / std
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
