"""Per-pair free / contact / event frequencies -> max over active pairs, plus lead ablations.

All frequencies are in Hz and normalized by f_max = 1 / dt (never by a per-episode max).
"""

from __future__ import annotations

import numpy as np

from tsr.interaction.pairs import pair_signals
from tsr.label.contact_term import f_contact
from tsr.label.event_term import f_event
from tsr.label.tau_term import f_free


def lead_max_filter(f: np.ndarray, lead_sec: float, dt: float) -> np.ndarray:
    """f'(t) = max f over [t, t + lead]: dense regions start lead_sec earlier."""
    n = int(round(lead_sec / dt))
    if n <= 0:
        return np.asarray(f, dtype=np.float64).copy()
    f = np.asarray(f, dtype=np.float64)
    padded = np.concatenate([f, np.full(n, -np.inf)])
    return np.lib.stride_tricks.sliding_window_view(padded, n + 1).max(axis=1)


def chunk_lead_envelope(f: np.ndarray, horizon_sec: float, dt: float) -> np.ndarray:
    """Exponential backward envelope with time constant horizon_sec (ablation)."""
    f = np.asarray(f, dtype=np.float64)
    if horizon_sec <= 0:
        return f.copy()
    decay = np.exp(-dt / horizon_sec)
    out, peak = f.copy(), 0.0
    for i in range(f.size - 1, -1, -1):
        peak = max(f[i], peak * decay)
        out[i] = peak
    return out


def label_episode(cache: dict, task_cfg: dict) -> dict[str, np.ndarray]:
    """Returns total frequency `f` (T,) plus per-pair terms for diagnostics."""
    dt = float(task_cfg["dt"])
    lc = task_cfg["label"]
    f_max = 1.0 / dt
    out, total = {}, None
    for pair in task_cfg["pairs"]:
        sig = pair_signals(cache, task_cfg["entities"], pair, lc, dt)
        terms = {
            "free": f_free(sig.gap, sig.gap_dot, lc["kappa"], lc["g0"], f_max),
            "contact": f_contact(
                sig.v_rel, sig.w_rel, sig.contact, pair["eps_pos"], np.deg2rad(pair["eps_rot_deg"]), f_max
            ),
            "event": f_event(sig.contact, dt, lc["event_pre_sec"], lc["event_post_sec"], f_max),
        }
        key = f"{pair['a']}-{pair['b']}"
        out[f"gap/{key}"] = sig.gap
        out[f"contact/{key}"] = sig.contact
        for name, f in terms.items():
            out[f"f_{name}/{key}"] = f
        f_pair = np.maximum.reduce(list(terms.values()))
        total = f_pair if total is None else np.maximum(total, f_pair)
    total = chunk_lead_envelope(total, lc.get("chunk_lead_sec", 0.0), dt)
    out["f"] = lead_max_filter(total, lc.get("lead_sec", 0.0), dt)
    return out


def frame_density(f: np.ndarray, dt: float, s_max: int) -> np.ndarray:
    """Per-frame keep density f * dt, clipped to strides in [1, s_max]."""
    return np.clip(np.asarray(f, dtype=np.float64) * dt, 1.0 / s_max, 1.0)


def log_density_label(f: np.ndarray, dt: float, s_max: int) -> np.ndarray:
    """Regression target y_t = log(frame density) for a learned labeler (TTR-Net)."""
    return np.log(frame_density(f, dt, s_max))
