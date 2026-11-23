"""Budget-constrained resampling: importance weights -> keep_idx."""

from __future__ import annotations

import numpy as np


def density_from_weights(
    w: np.ndarray,
    rho: float,
    d_min: float = 0.25,
) -> np.ndarray:
    """
    Map per-frame weights in [0, 1] to a local keep-density whose mean equals rho.

    d_min is the lowest density (= 1 / max stride). Max density is 1.0 (keep every frame).
    """
    if not d_min <= rho <= 1.0:
        raise ValueError(f"rho={rho} outside achievable range [{d_min}, 1.0]")
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 0:
        return w
    w = np.clip(w, 0.0, 1.0)
    base = d_min + (1.0 - d_min) * w

    def mean_at(scale: float) -> float:
        return float(np.clip(scale * base, d_min, 1.0).mean())

    lo, hi = 0.0, 1.0 / max(d_min, 1e-6)
    # Expand upper bound until mean >= rho (or saturates).
    for _ in range(40):
        if mean_at(hi) >= rho:
            break
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < rho:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * base, d_min, 1.0)


def keep_indices_from_density(d: np.ndarray) -> np.ndarray:
    """Cumulative-density sampler: keep a frame each time the integral crosses an integer."""
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        return np.zeros(0, dtype=np.int64)
    c = np.cumsum(d)
    # Prefer early frames when density is high: keep whenever floor jumps.
    floors = np.floor(np.concatenate([[0.0], c[:-1]]))
    keep = c >= floors + 1.0
    keep[0] = True
    keep[-1] = True
    return np.flatnonzero(keep).astype(np.int64)


def resample_by_weights(w: np.ndarray, rho: float, **kwargs) -> np.ndarray:
    """Convenience: weights + target keep ratio -> keep_idx."""
    d = density_from_weights(w, rho=rho, **kwargs)
    return keep_indices_from_density(d)


def keep_idx_from_frequency(f: np.ndarray, dt: float, s_max: int, rho: float | None = None) -> np.ndarray:
    """
    Sampling frequency f (Hz) -> keep_idx.
    rho=None: natural density clip(f * dt, 1/s_max, 1); else rescaled to mean keep ratio rho.
    """
    w = np.clip(np.asarray(f, dtype=np.float64) * dt, 0.0, 1.0)
    if rho is None:
        return keep_indices_from_density(np.clip(w, 1.0 / s_max, 1.0))
    return resample_by_weights(w, rho=rho, d_min=1.0 / s_max)


def uniform_keep_idx(t: int, rho: float) -> np.ndarray:
    """Evenly spaced keep indices with approximate keep ratio rho."""
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho={rho} outside (0, 1]")
    if t <= 0:
        return np.zeros(0, dtype=np.int64)
    n = max(2, int(round(rho * t)))
    n = min(n, t)
    idx = np.linspace(0, t - 1, n)
    return np.unique(np.round(idx).astype(np.int64))
