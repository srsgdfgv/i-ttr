"""In-contact frequency from physical tolerances (V1).

f_contact = max( |v| / eps_pos, |w| / eps_rot ) while in contact, clipped to f_max.
Translation and rotation are kept separate; tolerances are task-specific and should
include perception error. Cross-demo (relational entropy) tolerance is V2.
"""

from __future__ import annotations

import numpy as np


def f_contact(
    v_rel: np.ndarray,
    w_rel: np.ndarray,
    contact: np.ndarray,
    eps_pos: float,
    eps_rot: float,
    f_max: float,
) -> np.ndarray:
    f = np.maximum(np.asarray(v_rel) / eps_pos, np.asarray(w_rel) / eps_rot)
    return np.clip(f, 0.0, f_max) * np.asarray(contact, dtype=np.float64)
