"""Free-space approach frequency from tau theory + Weber relative precision.

f_free = clip( max(0, -g_dot) / (kappa * (g + g0)), 0, f_max )
g, g0 are normalized by the object's characteristic size; g_dot is per second; f in Hz.
"""

from __future__ import annotations

import numpy as np


def f_free(gap: np.ndarray, gap_dot: np.ndarray, kappa: float, g0: float, f_max: float) -> np.ndarray:
    approach = np.maximum(0.0, -np.asarray(gap_dot, dtype=np.float64))
    return np.clip(approach / (kappa * (np.asarray(gap, dtype=np.float64) + g0)), 0.0, f_max)
