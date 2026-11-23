"""Time-based smoothing and differentiation (windows in seconds, not frames)."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def window_frames(window_sec: float, dt: float, polyorder: int = 2) -> int:
    n = max(polyorder + 3, int(round(window_sec / dt)))
    return n + 1 - n % 2


def smooth(x: np.ndarray, dt: float, window_sec: float, deriv: int = 0) -> np.ndarray:
    """Savitzky-Golay along axis 0; deriv=1 returns d/dt in per-second units."""
    x = np.asarray(x, dtype=np.float64)
    n = min(window_frames(window_sec, dt), len(x) - (1 - len(x) % 2))
    if n < 5:
        return np.gradient(x, dt, axis=0) if deriv else x
    return savgol_filter(x, n, 2, deriv=deriv, delta=dt, axis=0, mode="interp")
