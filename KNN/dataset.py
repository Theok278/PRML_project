import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd


def pretreat_points(pts: np.ndarray, normalization: bool = True) -> np.ndarray:
    if normalization:
        mean = pts.mean(axis=0)
        std = pts.std(axis=0) + 1e-6
        pts = (pts - mean) / std
    return pts


def resample_points(pts: np.ndarray, seq_len: int) -> np.ndarray:
    n = pts.shape[0]
    if n == seq_len:
        return pts
    src = np.linspace(0.0, 1.0, n)
    tgt = np.linspace(0.0, 1.0, seq_len)
    out = np.zeros((seq_len, 3), dtype=pts.dtype)
    for i in range(3):
        out[:, i] = np.interp(tgt, src, pts[:, i])
    return out

# def resample_points(pts: np.ndarray, seq_len: int) -> np.ndarray:
#     """
#     Equal-distance (arc-length) resampling for 2D/3D trajectories.
#     pts: (N, D) where D=2 or 3
#     seq_len: target number of points
#     """
#     pts = np.asarray(pts)
#     N, D = pts.shape

#     if N == seq_len:
#         return pts.copy()

#     # Compute segment lengths
#     deltas = np.diff(pts, axis=0)
#     seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))

#     # If track has zero length (all points the same)
#     total_length = seg_lengths.sum()
#     if total_length < 1e-8:
#         return np.repeat(pts[0:1], seq_len, axis=0)

#     # Cumulative arc-length (0 ~ total_length)
#     cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)

#     # Target distances
#     target = np.linspace(0, total_length, seq_len)

#     # Output array
#     out = np.zeros((seq_len, D), dtype=float)

#     # Interpolate each dimension independently
#     for d in range(D):
#         out[:, d] = np.interp(target, cumulative, pts[:, d])

#     return out


class StrokeDataset:
    """Lightweight loader for KNN that returns flattened feature vectors."""

    def __init__(self, data_dir: str, seq_len: int = 50, normalization: bool = True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "stroke_*_*.csv")))
        if not self.files:
            raise FileNotFoundError(f"No stroke CSV found under {data_dir}")
        self.seq_len = seq_len
        self.normalization = normalization
        self.labels = [self._get_label(f) for f in self.files]
        self.num_classes = len(set(self.labels))

    def _get_label(self, path: str) -> int:
        name = os.path.basename(path)
        return int(name.split("_")[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        csv_path = self.files[idx]
        pts = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        pts = pretreat_points(pts, normalization=self.normalization)
        pts = resample_points(pts, self.seq_len)
        feat = pts.reshape(-1)  # flatten to (seq_len * 3,)
        label = self.labels[idx]
        return feat, label

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        feats = []
        labels = []
        for i in range(len(self)):
            f, y = self[i]
            feats.append(f)
            labels.append(y)
        return np.stack(feats), np.array(labels)
