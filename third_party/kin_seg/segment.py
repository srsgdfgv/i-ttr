"""Cluster kinematic features into precision(0) / casual(1) labels."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def standardize_features(X_list: list[np.ndarray], fit_global: bool = True):
    scaler = StandardScaler()
    if fit_global:
        scaler.fit(np.concatenate(X_list, axis=0))
    out = []
    for X in X_list:
        if fit_global:
            out.append(scaler.transform(X))
        else:
            s = StandardScaler()
            out.append(s.fit_transform(X))
    return out, scaler


def _assign_precision_casual(Xz: np.ndarray, raw_labels: np.ndarray) -> np.ndarray:
    """
    Map clusters -> {0 precision, 1 casual}.
    Casual = higher macro speed (dim0) and typically lower micro/macro ratio (dim2).
    """
    labels = raw_labels.copy()
    uniq = [u for u in np.unique(labels) if u >= 0]
    if len(uniq) == 0:
        return np.ones(len(labels), dtype=np.int32)
    if len(uniq) == 1:
        return np.ones(len(labels), dtype=np.int32)

    scores = {}
    for u in uniq:
        m = Xz[labels == u].mean(axis=0)
        # dim0 macro, dim1 wrist, dim2 ratio — casual scores high on macro, low on ratio
        scores[u] = float(m[0] - 0.5 * m[2])
    casual_id = max(scores, key=scores.get)
    precision_id = min(scores, key=scores.get)
    out = np.zeros(len(labels), dtype=np.int32)
    out[labels == casual_id] = 1
    out[labels == precision_id] = 0
    for u in uniq:
        if u not in (casual_id, precision_id):
            out[labels == u] = 1
    out[labels < 0] = 1
    return out


def cluster_gmm(Xz: np.ndarray, n_components: int = 2, seed: int = 0) -> np.ndarray:
    gmm = GaussianMixture(n_components=n_components, random_state=seed, covariance_type="full")
    raw = gmm.fit_predict(Xz)
    return _assign_precision_casual(Xz, raw)


def cluster_hdbscan(Xz: np.ndarray, min_cluster_size: int = 15) -> np.ndarray:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    raw = clusterer.fit_predict(Xz)
    return _assign_precision_casual(Xz, raw)


def apply_early_casual_prior(labels: np.ndarray, early_frac: float = 0.22) -> np.ndarray:
    """Entropy labels treat early transport as casual; kin over-marked precision there."""
    out = labels.copy()
    n = len(out)
    cut = int(early_frac * n)
    out[:cut] = 1
    return out


def apply_speed_gate(
    labels: np.ndarray,
    macro: np.ndarray,
    percentile: float = 45.0,
) -> np.ndarray:
    """
    Within an episode, frames faster than `percentile` of macro speed cannot stay
    precision — fixes medium-speed FP vs entropy casual.
    """
    out = labels.copy()
    thr = np.percentile(macro, percentile)
    out[(out == 0) & (macro > thr)] = 1
    return out


def apply_gripper_override(
    labels: np.ndarray,
    gripper_dg: np.ndarray,
    dg_thresh: float = 0.015,
    half_window: int = 10,
) -> np.ndarray:
    """Force precision (0) around significant gripper peak/rising-edge events."""
    out = labels.copy()
    t = len(out)
    above = gripper_dg > dg_thresh
    events = []
    for i in range(t):
        if not above[i]:
            continue
        left = gripper_dg[i - 1] if i > 0 else -np.inf
        right = gripper_dg[i + 1] if i + 1 < t else -np.inf
        if gripper_dg[i] >= left and gripper_dg[i] >= right:
            events.append(i)
        elif i == 0 or not above[i - 1]:
            events.append(i)
    for i in events:
        a = max(0, i - half_window)
        b = min(t, i + half_window + 1)
        out[a:b] = 0
    return out


def apply_hysteresis(labels: np.ndarray, n_frames: int = 5) -> np.ndarray:
    if len(labels) == 0:
        return labels
    out = np.empty_like(labels)
    cur = int(labels[0])
    out[0] = cur
    pending = None
    count = 0
    for i in range(1, len(labels)):
        v = int(labels[i])
        if v == cur:
            pending = None
            count = 0
            out[i] = cur
            continue
        if pending != v:
            pending = v
            count = 1
        else:
            count += 1
        if count >= n_frames:
            cur = pending
            pending = None
            count = 0
        out[i] = cur
    return out


def apply_precision_lead(labels: np.ndarray, lead_frames: int) -> np.ndarray:
    """
    Shift each precision (0) segment start earlier by `lead_frames`.
    Protects approach-to-precision frames that would otherwise be accelerated.
    """
    if lead_frames <= 0 or len(labels) == 0:
        return labels
    out = labels.copy()
    t = len(out)
    i = 0
    while i < t:
        if out[i] != 0:
            i += 1
            continue
        start = i
        while i < t and out[i] == 0:
            i += 1
        a = max(0, start - lead_frames)
        out[a:start] = 0
    return out


def segment_episode(
    X: np.ndarray,
    gripper_dg: np.ndarray,
    method: str = "gmm",
    seed: int = 0,
    dg_thresh: float = 0.025,
    gripper_half_window: int = 6,
    hysteresis_n: int = 5,
    Xz: Optional[np.ndarray] = None,
    macro: Optional[np.ndarray] = None,
    early_frac: float = 0.14,
    speed_percentile: float = 70.0,
    precision_lead_frames: int = 0,
) -> np.ndarray:
    """
    Returns labels [T] with 0=precision, 1=casual.

    Pipeline (order matters; calibrated on 50 insertion demos vs /labels):
      cluster → speed gate → gripper override → early casual → hysteresis
      → precision lead → reapply early casual.

    Early prior must run *after* gripper override: otherwise grip peaks in the
    transport phase re-paint the whole start as precision (main early-FP bug).
    Precision lead runs after hysteresis, then early casual is reapplied so lead
    cannot paint the transport head as precision.
    """
    if Xz is None:
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
    if method == "gmm":
        labels = cluster_gmm(Xz, seed=seed)
    elif method == "hdbscan":
        labels = cluster_hdbscan(Xz)
    else:
        raise ValueError(method)

    if macro is None:
        macro = X[:, 0]
    labels = apply_speed_gate(labels, macro, percentile=speed_percentile)
    labels = apply_gripper_override(labels, gripper_dg, dg_thresh, gripper_half_window)
    # Force early transport casual even if gripper override painted peaks.
    labels = apply_early_casual_prior(labels, early_frac=early_frac)
    labels = apply_hysteresis(labels, hysteresis_n)
    labels = apply_precision_lead(labels, precision_lead_frames)
    # Lead must not re-paint early transport as precision.
    labels = apply_early_casual_prior(labels, early_frac=early_frac)
    return labels.astype(np.int32)
