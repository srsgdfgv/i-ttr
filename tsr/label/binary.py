"""Binary precision/casual labels -> sampling frequency, with shared time-based lead."""

from __future__ import annotations

import numpy as np

from tsr.label.combine import lead_max_filter


def binary_to_frequency(labels: np.ndarray, dt: float, s_max: int) -> np.ndarray:
    """0=precision -> f_max; 1=casual -> f_max/s_max. Units Hz."""
    f_max = 1.0 / dt
    prec = (np.asarray(labels).ravel() == 0).astype(np.float64)
    return prec * f_max + (1.0 - prec) * (f_max / s_max)


def frequency_from_binary_with_lead(
    labels: np.ndarray, dt: float, s_max: int, lead_sec: float
) -> np.ndarray:
    f = binary_to_frequency(labels, dt, s_max)
    return lead_max_filter(f, lead_sec, dt)
