"""Convert existing baseline labels into keep_idx under a shared keep ratio."""

from __future__ import annotations

import numpy as np

from tsr.resample import resample_by_weights, uniform_keep_idx


def labels_to_weights(labels: np.ndarray) -> np.ndarray:
    """DemoSpeedup / kin_seg convention: 0=precision (keep dense), 1=casual (sparse)."""
    labels = np.asarray(labels).ravel()
    return (labels == 0).astype(np.float64)


def keep_idx_from_binary_labels(labels: np.ndarray, rho: float) -> np.ndarray:
    return resample_by_weights(labels_to_weights(labels), rho=rho)


def keep_idx_demospeedup(labels: np.ndarray, low_v: int = 2, high_v: int = 4) -> np.ndarray:
    """
    Exact numpy port of DemoSpeedup `process_action_label` index selection
    (third_party/act/act_utils.py), including its i=-1 start that reads labels[-1].
    """
    label = np.asarray(labels).ravel()
    horizon = len(label)
    indices = []
    i = -1
    while i < horizon:
        if label[i] == 0 and i + low_v < horizon:
            i += low_v
            indices.append(i)
        elif label[i] == 1:
            if i + high_v < horizon and np.all(label[i : i + high_v] == 1):
                i += high_v
                indices.append(i)
            else:
                next_zero = np.flatnonzero(label[i + 1 :] == 0)
                if len(next_zero) > 0:
                    i = i + 1 + int(next_zero[0])
                    indices.append(i)
                else:
                    break
        else:
            i += 1
    return np.asarray(indices, dtype=np.int64)


def keep_idx_uniform(t: int, rho: float) -> np.ndarray:
    return uniform_keep_idx(t, rho)


def keep_idx_full(t: int) -> np.ndarray:
    return np.arange(t, dtype=np.int64)


def apply_fixed_lead_to_labels(labels: np.ndarray, lead_frames: int) -> np.ndarray:
    """Expand precision (0) backward by lead_frames — same semantics as kin_seg lead."""
    out = np.asarray(labels, dtype=np.int32).copy()
    if lead_frames <= 0:
        return out
    prec = np.flatnonzero(out == 0)
    for i in prec:
        start = max(0, int(i) - lead_frames)
        out[start : i + 1] = 0
    return out
