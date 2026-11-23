"""Force max frequency around geometric contact on/off events.

The pre-event window uses future information; valid for offline demo processing only.
"""

from __future__ import annotations

import numpy as np


def contact_events(contact: np.ndarray) -> np.ndarray:
    c = np.asarray(contact, dtype=bool)
    ev = np.zeros_like(c)
    ev[1:] = c[1:] != c[:-1]
    return ev


def f_event(contact: np.ndarray, dt: float, pre_sec: float, post_sec: float, f_max: float) -> np.ndarray:
    ev = contact_events(contact)
    out = np.zeros(ev.shape[0])
    pre, post = int(round(pre_sec / dt)), int(round(post_sec / dt))
    for i in np.flatnonzero(ev):
        out[max(0, i - pre) : i + post + 1] = f_max
    return out
