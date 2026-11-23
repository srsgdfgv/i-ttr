"""Small rigid-body helpers. Quaternions are MuJoCo order (w, x, y, z)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(m).as_quat()[..., [3, 0, 1, 2]]


def boxes_to_world(boxes: np.ndarray, pos: np.ndarray, quat: np.ndarray):
    """boxes (M, 6) in body frame -> centers (T, M, 3), rotations (T, 3, 3), half sizes (M, 3)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    R = quat_to_mat(quat).reshape(-1, 3, 3)
    centers = np.einsum("tij,mj->tmi", R, boxes[:, :3]) + np.asarray(pos).reshape(-1, 1, 3)
    return centers, R, boxes[:, 3:]


def points_to_boxes_distance(points: np.ndarray, centers: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    """
    Min distance from point sets to axis-aligned-in-body boxes (zero inside).
    points (T, N, 3); centers (T, M, 3); R (T, 3, 3) shared by all boxes; half (M, 3) -> (T,).
    """
    diff = points[:, :, None, :] - centers[:, None, :, :]  # (T, N, M, 3)
    local = np.einsum("tnmi,tij->tnmj", diff, R)
    outside = np.maximum(np.abs(local) - half[None, None], 0.0)
    return np.linalg.norm(outside, axis=-1).min(axis=(1, 2))


def sample_box_surface(boxes: np.ndarray, spacing: float = 0.004) -> np.ndarray:
    """Body-frame surface points of each box, (N, 3)."""
    pts = []
    for cx, cy, cz, hx, hy, hz in np.asarray(boxes, dtype=np.float64).reshape(-1, 6):
        h = np.array([hx, hy, hz])
        axes = [np.linspace(-a, a, max(2, int(np.ceil(2 * a / spacing)) + 1)) for a in h]
        for k in range(3):
            i, j = [a for a in range(3) if a != k]
            gi, gj = np.meshgrid(axes[i], axes[j], indexing="ij")
            for sign in (-1.0, 1.0):
                p = np.zeros((gi.size, 3))
                p[:, i], p[:, j], p[:, k] = gi.ravel(), gj.ravel(), sign * h[k]
                pts.append(p + [cx, cy, cz])
    return np.unique(np.concatenate(pts).round(6), axis=0)
