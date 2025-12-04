"""
Resampling functions for temporal trajectories
"""

import numpy as np


def resample_points_temporal(points: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Temporal (equal-time) resampling: linear interpolation in time domain. Treats the sequence as evenly spaced in time and linearly interpolates
    to a target number of points.

    Args:
        points: (N, D) trajectory where D=2 or 3
        seq_len: target number of points

    Returns:
        resampled: (seq_len, D) resampled trajectory
    """
    n_points = points.shape[0]
    if n_points == seq_len:
        return points.copy()

    # Uniform temporal positions
    orig_pos = np.linspace(0.0, 1.0, n_points)
    target_pos = np.linspace(0.0, 1.0, seq_len)

    # Interpolate each dimension
    resampled = np.zeros((seq_len, points.shape[1]), dtype=points.dtype)
    for i in range(points.shape[1]):
        resampled[:, i] = np.interp(target_pos, orig_pos, points[:, i])

    return resampled


def resample_points_arclength(points: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Arc-length (equal-distance) resampling: interpolation along spatial trajectory. Resamples points at equal distances along the spatial path, preserving
    geometric properties better than temporal resampling.

    Args:
        points: (N, D) trajectory where D=2 or 3
        seq_len: target number of points

    Returns:
        resampled: (seq_len, D) resampled trajectory
    """
    pts = np.asarray(points)
    N, D = pts.shape

    if N == seq_len:
        return pts.copy()

    # Compute segment lengths
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))

    # If track has zero length (all points the same)
    total_length = seg_lengths.sum()
    if total_length < 1e-8:
        return np.repeat(pts[0:1], seq_len, axis=0)

    # Cumulative arc-length (0 ~ total_length)
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)

    # Target distances (equally spaced along arc-length)
    target_distances = np.linspace(0, total_length, seq_len)

    # Interpolate each dimension independently
    resampled = np.zeros((seq_len, D), dtype=float)
    for d in range(D):
        resampled[:, d] = np.interp(target_distances, cumulative, pts[:, d])

    return resampled


def resample_points(points: np.ndarray, seq_len: int, method: str = 'arclength') -> np.ndarray:
    """
    Resample a trajectory to a fixed number of points

    Args:
        points: (N, D) trajectory where D=2 or 3
        seq_len: target number of points
        method: resampling method
                'temporal' - equal-time interpolation
                'arclength' - equal-distance interpolation (default)

    Returns:
        resampled: (seq_len, D) resampled trajectory
    """
    if method == 'temporal':
        return resample_points_temporal(points, seq_len)
    elif method == 'arclength':
        return resample_points_arclength(points, seq_len)
    else:
        raise ValueError(f"Unknown resampling method: {method}. Must be 'temporal' or 'arclength'")
