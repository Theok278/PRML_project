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
    get_input_dim
)
from resample import resample_points

# global mean/std for different feature dimensions
# 12D，movement_features='all' + use_resample=True + seq_len=64
GLOBAL_MEAN_12 = np.array([
    -8.40791521e+00,  2.63744078e+02, -3.63322639e+01,
    -6.91013829e-02, -1.96790448e+00,  1.58798460e-01,
    -5.59129511e-03, -3.11485447e-01,  1.70422818e-02,
     1.76752153e-01,  4.99350874e-01,  4.67816359e+02
], dtype=np.float32)

GLOBAL_STD_12 = np.array([
    3.90405219e+01,  8.06492217e+01,  3.03436617e+01,
    5.18651684e+00,  5.97024110e+00,  2.07546654e+00,
    6.12902726e-01,  6.84123668e-01,  2.08214794e-01,
    2.89165406e-01,  2.95448709e-01,  2.49097978e+02
], dtype=np.float32)

# other dimensions derived from the 12D stats
# movement_features='none': x, y, z
GLOBAL_MEAN_3 = GLOBAL_MEAN_12[:3]   
GLOBAL_STD_3 = GLOBAL_STD_12[:3]

# movement_features='cat_move': x, y, z, dx, dy, dz
GLOBAL_MEAN_6 = GLOBAL_MEAN_12[:6]   
GLOBAL_STD_6 = GLOBAL_STD_12[:6]

# movement_features='cat_dir': x, y, z, dx, dy, dz, dir_x, dir_y, dir_z
GLOBAL_MEAN_9 = GLOBAL_MEAN_12[:9]   
GLOBAL_STD_9 = GLOBAL_STD_12[:9]

class DigitsStrokeDataset(Dataset):
    """Dataset for 3D digit strokes stored as CSV files."""

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 128,
        normalization: bool = True,
        file_list: List[str] | None = None,
        augmentation: Optional[str] = None,  # 'light', 'medium', 'strong', or None
        movement_features: Optional[str] = None,  # 'none', 'cat_move', 'cat_dir', 'all'
        training: bool = True,
        use_resample: bool = False,  # If True, use resample; if False, use padding/truncation
        resample_method: str = 'arclength',  # 'temporal' or 'arclength' (only used if use_resample=True)

        # Feature mean/std for normalization (can be passed from outside, e.g., for N-fold CV validation)
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,

        # Whether to use predefined global statistics (for standalone training)
        use_global_stats: bool = False,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.normalization = normalization
        self.training = training
        self.movement_features = movement_features
        self.use_resample = use_resample
        self.resample_method = resample_method
        self.use_global_stats = use_global_stats

        # Feature mean/std for normalization
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = None if feature_std is None else np.asarray(feature_std, dtype=np.float32)

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
            if movement_features not in ['none', 'cat_move', 'cat_dir', 'all']:
                raise ValueError(
                    f"Unknown movement_features mode: {movement_features}. "
                    f"Must be one of: 'none', 'cat_move', 'cat_dir', 'all'"
                )
            self.movement_extractor = MovementFeatureExtractor(mode=movement_features)
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

        # Compute feature mean/std if needed
        if movement_features is not None and self.normalization and (self.feature_mean is None or self.feature_std is None):
            if use_global_stats:
                # Use predefined global statistics
                if movement_features == 'all':
                    self.feature_mean = GLOBAL_MEAN_12.copy()
                    self.feature_std = GLOBAL_STD_12.copy()
                    print(f"[DigitsStrokeDataset] Using global statistics (12D)")
                elif movement_features == 'cat_dir':
                    self.feature_mean = GLOBAL_MEAN_9.copy()
                    self.feature_std = GLOBAL_STD_9.copy()
                    print(f"[DigitsStrokeDataset] Using global statistics (9D)")
                elif movement_features == 'cat_move':
                    self.feature_mean = GLOBAL_MEAN_6.copy()
                    self.feature_std = GLOBAL_STD_6.copy()
                    print(f"[DigitsStrokeDataset] Using global statistics (6D)")
                elif movement_features == 'none':
                    self.feature_mean = GLOBAL_MEAN_3.copy()
                    self.feature_std = GLOBAL_STD_3.copy()
                    print(f"[DigitsStrokeDataset] Using global statistics (3D)")
                else:
                    raise ValueError(
                        f"Global statistics not available for movement_features='{movement_features}'. "
                        f"Supported: 'none' (3D), 'cat_move' (6D), 'cat_dir' (9D), 'all' (12D)"
                    )
            elif self.training:
                # Calculate from training set
                print("[DigitsStrokeDataset] Computing dataset-level mean/std over features...")
                self.feature_mean, self.feature_std = self._compute_feature_mean_std()
                print("[DigitsStrokeDataset] Done. mean =", self.feature_mean)
                print("[DigitsStrokeDataset] Done. std  =", self.feature_std)
            else:
                # Validation/test set without provided statistics and not using global statistics
                if movement_features is not None:
                    raise RuntimeError(
                        "For validation/test set with movement_features, you must either:\n"
                        "  1. Pass feature_mean/std from training set (recommended for N-fold CV), or\n"
                        "  2. Use use_global_stats=True to use predefined global statistics"
                    )

    def _extract_label(self, path: str) -> int:
        filename = os.path.basename(path)
        # Expected pattern: stroke_LABEL_XXXX.csv
        return int(filename.split("_")[1])

    def __len__(self) -> int:
        return len(self.files)

    def _compute_feature_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std of movement features over the entire dataset.
        """
        if self.movement_extractor is None:
            raise RuntimeError(
                "movement_extractor is None, cannot compute feature statistics. "
                "This should not happen. Please set movement_features to one of: "
                "'none', 'cat_move', 'cat_dir', 'all'"
            )

        count = 0
        mean = None
        m2 = None

        for path in self.files:
            pts = pd.read_csv(path, header=None).values.astype(np.float64)  # (N,3)
            if pts.shape[0] < 2:
                continue

            if self.use_resample:
                pts_proc = resample_points(pts, self.seq_len, method=self.resample_method)
            else:
                pts_proc = pts

            feats = self.movement_extractor(pts_proc)  # (T, feat_dim)

            if mean is None:
                feat_dim = feats.shape[1]
                mean = np.zeros(feat_dim, dtype=np.float64)
                m2 = np.zeros(feat_dim, dtype=np.float64)

            for x in feats:  # x: (feat_dim,)
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                m2 += delta * delta2

        if count < 2:
            raise RuntimeError("Not enough feature points to compute statistics")

        var = m2 / (count - 1)
        std = np.sqrt(var) + 1e-8

        return mean.astype(np.float32), std.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        csv_path = self.files[idx]
        pts = pd.read_csv(csv_path, header=None).values

        # Store original length before any processing
        orig_len = pts.shape[0]

        # 1. Apply augmentation first (only in training)
        if self.augment is not None:
            pts = self.augment(pts)

        # 2. Resample or keep original length based on use_resample flag
        if self.use_resample:
            pts = resample_points(pts, self.seq_len, method=self.resample_method)

        # 3. Extract movement features (both training and testing!)
        if self.movement_extractor is not None:
            pts = self.movement_extractor(pts)  # (T, feat_dim)

        # 4. Normalize if needed
        if self.movement_extractor is not None and self.normalization:
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("feature_mean/std is None, but normalization=True. "
                                   "For test/val set, please pass trainset's mean/std into the constructor.")
            pts = (pts - self.feature_mean) / self.feature_std

        # 5. Pad or truncate to seq_len (if not using resample)
        if self.use_resample:
            # Already resampled to seq_len, all positions are valid
            T = pts.shape[0]
            if T != self.seq_len:
                if T > self.seq_len:
                    pts = pts[:self.seq_len]
                    mask = np.ones(self.seq_len, dtype=bool)
                else:
                    pad_len = self.seq_len - T
                    pts = np.pad(pts, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                    mask = np.concatenate([np.ones(T, dtype=bool),
                                           np.zeros(pad_len, dtype=bool)])
            else:
                mask = np.ones(self.seq_len, dtype=bool)
        else:
            # Use padding/truncation
            actual_len = pts.shape[0]
            if actual_len > self.seq_len:
                # Truncate if longer
                pts = pts[:self.seq_len]
                mask = np.ones(self.seq_len, dtype=bool)
            elif actual_len < self.seq_len:
                # Pad if shorter
                pad_len = self.seq_len - actual_len
                pts = np.pad(pts, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                mask = np.concatenate([np.ones(actual_len, dtype=bool),
                                       np.zeros(pad_len, dtype=bool)])
            else:
                mask = np.ones(self.seq_len, dtype=bool)

        seq = torch.from_numpy(pts).float()      # (seq_len, feat_dim)
        mask_tensor = torch.from_numpy(mask)     # (seq_len,)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label, mask_tensor