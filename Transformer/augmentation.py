"""
Data Augmentation for Air Writing (3D Temporal Point Clouds)

Specialized augmentations for temporal 3D stroke data that preserve:
1. Temporal order (sequence must remain intact)
2. Stroke semantics (shape/topology should stay recognizable)
3. Writing dynamics (speed patterns are informative)
"""

import numpy as np
from typing import Optional, Tuple


class AirWritingAugmentation:
    """
    Augmentation suite for air writing temporal point clouds

    Design principles:
    - Preserve temporal order (no shuffling)
    - Maintain stroke topology
    - Simulate natural variations in writing
    """

    def __init__(
        self,
        rotation_range: float = 15.0,  # degrees
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_std: float = 0.1,
        jitter_std: float = 0.02,
        time_warp_sigma: float = 0.2,
        dropout_ratio: float = 0.1
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_std = translation_std
        self.jitter_std = jitter_std
        self.time_warp_sigma = time_warp_sigma
        self.dropout_ratio = dropout_ratio


    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations

        Args:
            points: (N, 3) temporal point cloud
        Returns:
            Augmented points (N, 3)
        """
        points = points.copy()

        # 1. Random 3D rotation (simulates different viewing angles)
        if self.rotation_range > 0:
            points = self.random_rotation_xy(points)

        # 2. Random scaling (simulates writing size variations)
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            points = self.random_scale(points)

        # 3. Random translation (simulates different writing positions)
        if self.translation_std > 0:
            points = self.random_translation(points)

        # 4. Point jitter (simulates hand tremor/noise)
        if self.jitter_std > 0:
            points = self.random_jitter(points)

        # 5. Time warping (simulates speed variations)
        if self.time_warp_sigma > 0:
            points = self.time_warp(points)

        # 6. Point dropout (simulates missing data/occlusion)
        if self.dropout_ratio > 0:
            points = self.random_dropout(points)

        return points
    
    def random_rotation_xy(self, points: np.ndarray) -> np.ndarray:
        """
        Random rotation in the XY plane (around Z-axis).

        Equivalent to 2D rotation for air-writing digits.
        Only rotates x,y while keeping z unchanged.
        """
        # angle range still uses self.rotation_range (in degrees)
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rad = np.deg2rad(angle)

        # 2D rotation matrix
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad),  np.cos(rad)]
        ], dtype=np.float32)

        rotated = points.copy()
        rotated[:, :2] = points[:, :2] @ R.T   # rotate x,y ONLY

        return rotated

    def random_rotation_3d(self, points: np.ndarray) -> np.ndarray:
        """
        Random 3D rotation around random axis

        Simulates different viewing angles during air writing
        """
        angle = np.random.uniform(-self.rotation_range, self.rotation_range, size=3)
        angle_rad = np.deg2rad(angle)
        rx, ry, rz = angle_rad

        # Random axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx
        return points @ R.T

    def random_scale(self, points: np.ndarray) -> np.ndarray:
        """
        Random uniform scaling

        Simulates different writing sizes (small vs large digits)
        """
        scale = np.random.uniform(*self.scale_range)

        # Center before scaling
        center = points.mean(axis=0)
        points = (points - center) * scale + center

        return points

    def random_translation(self, points: np.ndarray) -> np.ndarray:
        """
        Random translation

        Simulates writing at different positions in 3D space
        """
        translation = np.random.randn(3) * self.translation_std
        return points + translation

    def random_jitter(self, points: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to each point

        Simulates hand tremor and sensor noise
        """
        noise = np.random.randn(*points.shape) * self.jitter_std
        return points + noise

    def time_warp(self, points: np.ndarray) -> np.ndarray:
        """
        Temporal warping (smooth non-linear time distortion)

        Simulates variations in writing speed while preserving temporal order

        Uses smooth random warping to avoid breaking stroke continuity
        """
        n_points = len(points)

        # Generate smooth warping curve
        # Create warping at keypoints, then interpolate
        n_keypoints = max(4, n_points // 20)
        keypoints_idx = np.linspace(0, n_points - 1, n_keypoints)

        # Random warping at keypoints (cumulative, monotonically increasing)
        warp_keypoints = np.cumsum(np.random.randn(n_keypoints) * self.time_warp_sigma)
        warp_keypoints = warp_keypoints - warp_keypoints.min()  # Start from 0
        warp_keypoints = warp_keypoints / warp_keypoints.max() * (n_points - 1)  # End at n_points-1

        # Interpolate to get warp for all points
        original_idx = np.arange(n_points)
        warped_idx = np.interp(original_idx, keypoints_idx, warp_keypoints)

        # Resample points using warped indices
        warped_points = np.zeros_like(points)
        for i in range(3):
            warped_points[:, i] = np.interp(warped_idx, original_idx, points[:, i])

        return warped_points

    def random_dropout(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly drop points (with temporal coherence)

        Simulates missing data or occlusion during writing
        Uses segment-wise dropout to maintain temporal continuity
        """
        n_points = len(points)
        n_drop = int(n_points * self.dropout_ratio)

        if n_drop == 0:
            return points

        # Segment-wise dropout (drop consecutive points)
        n_segments = max(1, n_drop // 5)
        segment_length = n_drop // n_segments

        keep_mask = np.ones(n_points, dtype=bool)

        for _ in range(n_segments):
            start_idx = np.random.randint(0, n_points - segment_length)
            keep_mask[start_idx:start_idx + segment_length] = False

        # Keep at least 50% of points
        if keep_mask.sum() < n_points // 2:
            return points

        # Interpolate to maintain original length
        kept_points = points[keep_mask]
        kept_indices = np.where(keep_mask)[0]

        interpolated = np.zeros_like(points)
        for i in range(3):
            interpolated[:, i] = np.interp(
                np.arange(n_points),
                kept_indices,
                kept_points[:, i]
            )

        return interpolated


class MixUp:
    """
    MixUp augmentation for point clouds

    Linearly interpolates between two samples:
        x_mixed = λ * x1 + (1-λ) * x2
        y_mixed = λ * y1 + (1-λ) * y2

    Note: Returns soft labels, requires loss that supports them
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter (smaller = less mixing)
        """
        self.alpha = alpha

    def __call__(
        self,
        points1: np.ndarray,
        label1: int,
        points2: np.ndarray,
        label2: int,
        num_classes: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix two samples

        Args:
            points1: First point cloud (N, 3)
            label1: First label
            points2: Second point cloud (N, 3)
            label2: Second label
            num_classes: Number of classes

        Returns:
            Mixed points, mixed label (one-hot)
        """
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix points
        mixed_points = lam * points1 + (1 - lam) * points2

        # Mix labels (one-hot)
        mixed_label = np.zeros(num_classes, dtype=np.float32)
        mixed_label[label1] += lam
        mixed_label[label2] += (1 - lam)

        return mixed_points, mixed_label


class CutMix:
    """
    CutMix for temporal point clouds

    Replaces a temporal segment from one sample with another
    More suitable for temporal data than spatial cutout
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        points1: np.ndarray,
        label1: int,
        points2: np.ndarray,
        label2: int,
        num_classes: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix two samples

        Args:
            points1: First point cloud (N, 3)
            label1: First label
            points2: Second point cloud (N, 3)
            label2: Second label
            num_classes: Number of classes

        Returns:
            Mixed points, mixed label (one-hot)
        """
        n_points = len(points1)

        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)

        # Temporal cut size
        cut_len = int(n_points * (1 - lam))

        # Random temporal cut position
        cut_start = np.random.randint(0, n_points - cut_len + 1)
        cut_end = cut_start + cut_len

        # Replace segment
        mixed_points = points1.copy()
        mixed_points[cut_start:cut_end] = points2[cut_start:cut_end]

        # Mix labels
        mixed_label = np.zeros(num_classes, dtype=np.float32)
        mixed_label[label1] += lam
        mixed_label[label2] += (1 - lam)

        return mixed_points, mixed_label


class MovementFeatureExtractor:
    """
    Extract movement features from temporal point cloud data

    Computes velocity/displacement features: dx, dy, dz = (x,y,z)_t - (x,y,z)_(t-1)
    and concatenates them with the original position at each timestep.

    This captures motion dynamics which are important for handwriting recognition.
    """

    def __init__(self, mode: str = 'concat'):
        """
        Args:
            mode: 'concat' - concatenate [x,y,z,dx,dy,dz] -> 6D
                  'replace' - replace with [dx,dy,dz] -> 3D
                  'velocity_only' - return only velocity features
        """
        self.mode = mode

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Extract movement features from point cloud

        Args:
            points: (N, 3) temporal point cloud [x, y, z] at each timestep
        Returns:
            features: (N, 3 or 6) depending on mode
                      - concat: (N, 6) [x, y, z, dx, dy, dz]
                      - replace: (N, 3) [dx, dy, dz]
        """
        n_points = len(points)

        # Compute displacement: (x,y,z)_t - (x,y,z)_(t-1)
        # For t=0, use zero displacement (no previous point)
        displacements = np.zeros_like(points)
        displacements[1:] = points[1:] - points[:-1]  # dx, dy, dz for t >= 1


        if self.mode == 'concat':
            # Concatenate position and velocity: [x, y, z, dx, dy, dz]
            features = np.concatenate([points, displacements], axis=-1)  # (N, 6)
        elif self.mode == 'replace':
            # Replace with velocity only: [dx, dy, dz]
            features = displacements  # (N, 3)
        elif self.mode == 'velocity_only':
            # Only velocity features (same as replace)
            features = displacements  # (N, 3)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return features.astype(np.float32)


class AccelerationFeatureExtractor:
    """
    Extract acceleration features from temporal point cloud data

    Computes acceleration: ddx, ddy, ddz = velocity_t - velocity_(t-1)
    Optionally includes position, velocity, and acceleration.
    """

    def __init__(self, mode: str = 'all'):
        """
        Args:
            mode: 'all' - [x,y,z,dx,dy,dz,ddx,ddy,ddz] -> 9D
                  'velocity_acceleration' - [dx,dy,dz,ddx,ddy,ddz] -> 6D
                  'acceleration_only' - [ddx,ddy,ddz] -> 3D
        """
        self.mode = mode

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Extract acceleration features

        Args:
            points: (N, 3) temporal point cloud
        Returns:
            features: (N, 3/6/9) depending on mode
        """
        n_points = len(points)

        # First-order difference: velocity
        velocity = np.zeros_like(points)
        velocity[1:] = points[1:] - points[:-1]

        # Second-order difference: acceleration
        acceleration = np.zeros_like(points)
        acceleration[1:] = velocity[1:] - velocity[:-1]

        if self.mode == 'all':
            # Position + velocity + acceleration
            features = np.concatenate([points, velocity, acceleration], axis=-1)  # (N, 9)
        elif self.mode == 'velocity_acceleration':
            # Velocity + acceleration
            features = np.concatenate([velocity, acceleration], axis=-1)  # (N, 6)
        elif self.mode == 'acceleration_only':
            # Only acceleration
            features = acceleration  # (N, 3)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return features.astype(np.float32)

def get_input_dim(movement_features: str = None) -> int:
    """
    Helper function to determine input dimension based on movement features

    Args:
        movement_features: Movement feature mode

    Returns:
        input_dim: 3, 6, or 9

    Examples:
        input_dim = get_input_dim('concat')  # Returns 6
        input_dim = get_input_dim('all')     # Returns 9
        input_dim = get_input_dim(None)      # Returns 3
    """
    if movement_features in ['concat']:
        return 6
    elif movement_features in ['all']:
        return 9
    elif movement_features in ['velocity_acceleration']:
        return 6
    elif movement_features in [None, 'none', 'replace', 'velocity_only', 'acceleration_only']:
        return 3
    else:
        raise ValueError(f"Unknown movement features mode: {movement_features}")
