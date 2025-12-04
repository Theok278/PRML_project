import numpy as np
from typing import Optional, Tuple


class AirWritingAugmentation:
    """
    Augmentation suite for air writing temporal point clouds
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

class MovementFeatureExtractor:
    """
    Extract rich geometric + dynamic features for handwriting trajectory.

    Modes:
        - 'none': Only position (x, y, z) -> 3D
        - 'cat_move': Position + displacement (x, y, z, dx, dy, dz) -> 6D
        - 'cat_dir': Position + displacement + direction (x, y, z, dx, dy, dz, dir_x, dir_y, dir_z) -> 9D
        - 'all': Full features (x, y, z, dx, dy, dz, dir_x, dir_y, dir_z, curvature, s_norm, total_length) -> 12D
    """

    def __init__(self, mode: str = 'none'):
        """
        Args:
            mode: Feature extraction mode
                  'none' - Only position [x,y,z] -> 3D
                  'cat_move' - Position + movement [x,y,z,dx,dy,dz] -> 6D
                  'cat_dir' - Position + movement + direction [x,y,z,dx,dy,dz,dir_x,dir_y,dir_z] -> 9D
                  'all' - All features [x,y,z,dx,dy,dz,dir_x,dir_y,dir_z,cur,snorm,tlen] -> 12D
        """
        assert mode in ['none', 'cat_move', 'cat_dir', 'all'], \
            f"Unknown mode: {mode}. Must be one of: none, cat_move, cat_dir, all"
        self.mode = mode

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Extract features from trajectory

        Args:
            points: (N, 3) xyz sequence
        Returns:
            features: (N, 3/6/9/12) feature matrix depending on mode
        """
        pts = points.astype(np.float32)
        N = len(pts)

        # Base: Position
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        if self.mode == 'none':
            # Only position
            return np.stack([x, y, z], axis=1)  # (N, 3)

        # Compute displacement
        disp = np.zeros_like(pts)
        disp[1:] = pts[1:] - pts[:-1]
        dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]

        if self.mode == 'cat_move':
            # Position + displacement
            return np.stack([x, y, z, dx, dy, dz], axis=1)  # (N, 6)

        # Compute unit direction vector
        step = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8
        dir_vec = disp / step
        dir_x, dir_y, dir_z = dir_vec[:, 0], dir_vec[:, 1], dir_vec[:, 2]

        if self.mode == 'cat_dir':
            # Position + displacement + direction
            return np.stack([x, y, z, dx, dy, dz, dir_x, dir_y, dir_z], axis=1)  # (N, 9)

        # Compute curvature-like feature
        ddir = np.zeros_like(dir_vec)
        ddir[1:] = dir_vec[1:] - dir_vec[:-1]
        curvature_like = np.linalg.norm(ddir, axis=1)

        # Compute normalized arc-length
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumulative = np.insert(np.cumsum(seg_len), 0, 0)
        total_length = cumulative[-1] + 1e-8
        s_norm = cumulative / total_length

        # Total length as a feature
        total_len_feat = np.full((N,), total_length, dtype=np.float32)

        # All features
        return np.stack([
            x, y, z,
            dx, dy, dz,
            dir_x, dir_y, dir_z,
            curvature_like,
            s_norm,
            total_len_feat
        ], axis=1)  # (N, 12)

def get_input_dim(movement_features: str = None) -> int:
    """
    Helper function to determine input dimension based on movement features

    Args:
        movement_features: Movement feature mode

    Returns:
        input_dim: 3, 6, 9, or 12

    Examples:
        input_dim = get_input_dim('none')        # Returns 3
        input_dim = get_input_dim('cat_move')    # Returns 6
        input_dim = get_input_dim('cat_dir')     # Returns 9
        input_dim = get_input_dim('all')         # Returns 12
    """
    if movement_features in ['none', None]:
        return 3
    elif movement_features in ['cat_move']:
        return 6
    elif movement_features in ['cat_dir']:
        return 9
    elif movement_features in ['all']:
        return 12
    else:
        raise ValueError(f"Unknown movement features mode: {movement_features}")
