import glob
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augmentation import (
    AirWritingAugmentation,
    MovementFeatureExtractor,
    AccelerationFeatureExtractor,
    get_input_dim
)


def pretreat_points(points: np.ndarray, normalization: bool = True) -> np.ndarray:
    """Optionally normalize XYZ coordinates to zero-mean, unit-std."""
    if normalization:
        mean = points.mean(axis=0)
        std = points.std(axis=0) + 1e-6
        points = (points - mean) / std

    # GLOBAL_MEAN = np.array([-15.422633, 276.60189251, -38.86954802], dtype=np.float32)
    # GLOBAL_STD  = np.array([44.45173615, 101.60893454, 32.08909528], dtype=np.float32)  
    # if normalization:
    #     points = (points - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
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
        augmentation: Optional[str] = None,  # 'light', 'medium', 'strong', or None
        movement_features: Optional[str] = None,  # 'concat', 'replace', 'all', etc.
        training: bool = True,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.normalization = normalization
        self.training = training
        self.movement_features = movement_features

        # Setup augmentation (only for training)
        if augmentation is not None and training:
            aug_config = {
                'light': {
                    'rotation_range': 10.0,
                    'scale_range': (1, 1),
                    'translation_std': 0.0,
                    'jitter_std': 0.0,
                    'time_warp_sigma': 0.0,
                    'dropout_ratio': 0.0,
                },
                None: {
                    'rotation_range': 0.0,
                    'scale_range': (1, 1),
                    'translation_std': 0.0,
                    'jitter_std': 0.0,
                    'time_warp_sigma': 0.0,
                    'dropout_ratio': 0.0,
                },
            }
            assert augmentation in aug_config, f"Unknown augmentation type: {augmentation}"
            config = aug_config.get(augmentation, aug_config[None])
            self.augment = AirWritingAugmentation(**config)
            print(f"Training with {augmentation} augmentation")
        else:
            self.augment = None

        # Setup movement feature extractor (for both training and testing!)
        if movement_features is not None:
            if movement_features in ['concat', 'replace', 'velocity_only']:
                self.movement_extractor = MovementFeatureExtractor(mode=movement_features)
            elif movement_features in ['all', 'velocity_acceleration', 'acceleration_only']:
                self.movement_extractor = AccelerationFeatureExtractor(mode=movement_features)
            else:
                raise ValueError(f"Unknown movement_features mode: {movement_features}")

            mode_str = "training" if training else "testing"
            print(f"{mode_str.capitalize()} with movement features: {movement_features} (input_dim={get_input_dim(movement_features)})")
        else:
            self.movement_extractor = None

        if file_list is not None:
            self.files = sorted(file_list)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, "stroke_*_*.csv")))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No CSV files found under {data_dir}")

        self.labels = [self._extract_label(path) for path in self.files]
        self.num_classes = len(set(self.labels))
        aug_str = f" (augmentation: {augmentation})" if augmentation and training else ""
        print(f"Loaded {len(self.files)} samples across {self.num_classes} classes{aug_str}")

    def _extract_label(self, path: str) -> int:
        filename = os.path.basename(path)
        # Expected pattern: stroke_LABEL_XXXX.csv
        return int(filename.split("_")[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        csv_path = self.files[idx]
        pts = pd.read_csv(csv_path, header=None).values


        pts = resample_points(pts, self.seq_len) 

        # 1. Apply augmentation first (only in training)
        if self.augment is not None:
            pts = self.augment(pts)

        pts = pretreat_points(pts, normalization=self.normalization)

        # 2. Extract movement features (both training and testing!)
        if self.movement_extractor is not None:
            pts = self.movement_extractor(pts)

        seq = torch.from_numpy(pts).float()  # (seq_len, 3/6/9 depending on movement_features)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label
