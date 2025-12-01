import numpy as np
import os
import pandas as pd

def load_data(files, seq_len=50, normalize=True, sigma=None, scale_shift=None, max_angle=None):
    X_list, y_list = [], []
    for f in files:
        df = pd.read_csv(f, header=None)
        pts = df.to_numpy()

        if normalize:
            mean = pts.mean(axis=0)
            std = pts.std(axis=0)
            pts = (pts - mean) / std
        pts = resample_points(pts, seq_len)
        pts = augment_sequence(pts, sigma, scale_shift, max_angle)

        X_list.append(pts.astype(np.float32))
        label = int(os.path.basename(f).split("_")[1])
        y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y

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

def augment_sequence(seq, sigma=None, scale_shift=None, max_angle=None):
    seq = seq.copy()

    if sigma:
        seq += np.random.normal(0, sigma, size=seq.shape)

    if scale_shift:
        scale = np.random.uniform(1-scale_shift, 1+scale_shift)
        seq *= scale

    if max_angle:
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        Rz = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        seq = seq @ Rz.T

    return seq

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]




