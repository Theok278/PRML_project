import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DigitsPointCloudDataset(Dataset):
    """
    Dataset for 3D handwritten digits point clouds.

    Each sample is a CSV file containing (x, y, z) coordinates.

    Args:
        data_dir: Path to folder containing CSV files.
        seq_len: Resample point cloud to this length (default: 128).
        transform: Optional transforms applied on the data_dict.
        normalization: Whether to normalize points to zero-mean, unit-std.
    """

    def __init__(
        self,
        data_dir,
        seq_len: int = 128,
        transform=None,
        normalization: bool = True,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.normalization = normalization

        # Get all CSV files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        # Extract labels from filenames, assuming: stroke_LABEL_xxxx.csv
        self.labels = []
        for f in self.files:
            filename = os.path.basename(f)
            # Example: stroke_3_0123.csv -> label = 3
            label = int(filename.split("_")[1])
            self.labels.append(label)

        self.num_classes = len(set(self.labels))
        print(f"Loaded {len(self.files)} samples with {self.num_classes} classes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load point cloud from CSV -> numpy array of shape (N, 3)
        csv_path = self.files[idx]
        points = pd.read_csv(csv_path, header=None).values  # (N, 3)

        # Preprocess: orientation + normalization
        points = self.pretreat_points(points)

        # Resample to fixed sequence length
        points = self.resample_points(points, self.seq_len)  # (seq_len, 3)

        # Convert to tensor
        points = torch.from_numpy(points).float()  # (seq_len, 3)

        # -------- coord: 仍然使用 3 维 xyz --------
        coord = points  # (N, 3)

        # -------- feat: 扩展到 9 维特征 --------
        # 这里先用最简单的方式：[x, y, z, 0, 0, 0, 0, 0, 0]
        # 这样可以和官方的 Linear(9 -> C) 对齐使用预训练权重
        zeros = torch.zeros(points.shape[0], 6, dtype=points.dtype)
        feat = torch.cat([points, zeros], dim=1)  # (N, 9)

        label = self.labels[idx]

        data_dict = {
            "coord": coord,  # (N, 3)
            "feat": feat,    # (N, 9)
            "label": torch.tensor(label, dtype=torch.long),
            "grid_size": 0.01,  # 你原来就有的字段，保留
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict

    def pretreat_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Apply PCA-based orientation and optional normalization.

        pts: (N, 3) numpy array
        """
        if self.normalization:
            mean = pts.mean(axis=0)
            std = pts.std(axis=0) + 1e-6
            pts = (pts - mean) / std

        return pts

    def resample_points(self, pts: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Resample point sequence to fixed length using 1D interpolation along the stroke.

        pts: (N, 3)
        seq_len: target number of points
        """
        N = pts.shape[0]

        if N == seq_len:
            return pts

        # Parameterize original points in [0, 1]
        orig_positions = np.linspace(0.0, 1.0, N)
        target_positions = np.linspace(0.0, 1.0, seq_len)

        pts_resampled = np.zeros((seq_len, 3), dtype=pts.dtype)
        for i in range(3):
            pts_resampled[:, i] = np.interp(
                target_positions, orig_positions, pts[:, i]
            )

        return pts_resampled


def collate_fn(batch):
    """
    Custom collate function for batching.
    Input batch is a list of data_dict from DigitsPointCloudDataset.__getitem__.
    """
    # Shapes:
    #   coord: (N, 3)
    #   feat:  (N, 9)
    coords = torch.stack([item["coord"] for item in batch])  # (B, N, 3)
    feats = torch.stack([item["feat"] for item in batch])    # (B, N, 9)
    labels = torch.stack([item["label"] for item in batch])  # (B,)

    batch_size = len(batch)
    num_points = coords.shape[1]

    # offset[i] = 第 i 个样本在展平后的起始 index
    offset = torch.tensor([i * num_points for i in range(batch_size)], dtype=torch.long)

    return {
        "coord": coords.reshape(-1, 3),   # (B*N, 3)
        "feat": feats.reshape(-1, 9),     # (B*N, 9)
        "offset": offset,                 # (B,)
        "label": labels,                  # (B,)
    }