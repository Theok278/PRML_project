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


class StrokeDataset:
    """Loads stroke CSVs and returns flattened feature vectors for SVM."""

    def __init__(self, data_dir: str, seq_len: int = 50, normalization: bool = True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "stroke_*_*.csv")))
        if not self.files:
            raise FileNotFoundError(f"No stroke CSV found under {data_dir}")
        self.seq_len = seq_len
        self.normalization = normalization
        self.labels = [self._get_label(f) for f in self.files]
        self.num_classes = len(set(self.labels))

    def _get_label(self, path: str) -> int:
        return int(os.path.basename(path).split("_")[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        csv_path = self.files[idx]
        pts = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        pts = pretreat_points(pts, normalization=self.normalization)
        pts = resample_points(pts, self.seq_len)
        feat = pts.reshape(-1)
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
