import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def pretreat_points(points: np.ndarray, normalization: bool = True) -> np.ndarray:
    """Optionally normalize XYZ coordinates to zero-mean, unit-std."""
    if normalization:
        mean = points.mean(axis=0)
        std = points.std(axis=0) + 1e-6
        points = (points - mean) / std
    return points


def resample_points(points: np.ndarray, seq_len: int) -> np.ndarray:
    """Resample a variable-length stroke to a fixed-length sequence."""
    n_points = points.shape[0]
    if n_points == seq_len:
        return points

    orig_pos = np.linspace(0.0, 1.0, n_points)
    target_pos = np.linspace(0.0, 1.0, seq_len)

    resampled = np.zeros((seq_len, 3), dtype=points.dtype)
    for i in range(3):
        resampled[:, i] = np.interp(target_pos, orig_pos, points[:, i])
    return resampled

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

class DigitsStrokeDataset(Dataset):
    """Dataset for 3D digit strokes stored as CSV files."""

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 128,
        normalization: bool = True,
        file_list: List[str] | None = None,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.normalization = normalization

        if file_list is not None:
            self.files = sorted(file_list)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, "stroke_*_*.csv")))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No CSV files found under {data_dir}")

        self.labels = [self._extract_label(path) for path in self.files]
        self.num_classes = len(set(self.labels))
        print(f"Loaded {len(self.files)} samples across {self.num_classes} classes")

    def _extract_label(self, path: str) -> int:
        filename = os.path.basename(path)
        # Expected pattern: stroke_LABEL_XXXX.csv
        return int(filename.split("_")[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        csv_path = self.files[idx]
        pts = pd.read_csv(csv_path, header=None).values

        pts = pretreat_points(pts, normalization=self.normalization)
        pts = resample_points(pts, self.seq_len)

        seq = torch.from_numpy(pts).float()  # (seq_len, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label
