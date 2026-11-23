"""Interaction graph: per active pair gap, relative motion and geometric contact (no force/torque)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from tsr.geometry import boxes_to_world, points_to_boxes_distance, quat_to_mat, sample_box_surface
from tsr.signals import smooth


@dataclass
class PairSignals:
    gap: np.ndarray  # (T,) smoothed, normalized by char_size of b
    gap_dot: np.ndarray  # (T,) per second
    v_rel: np.ndarray  # (T,) m/s, b relative to a's frame
    w_rel: np.ndarray  # (T,) rad/s
    contact: np.ndarray  # (T,) bool


def _entity(cache: dict, spec: dict) -> dict:
    if "effector" in spec:
        s = spec["effector"]
        return {"points": cache[f"eff_pts_{s}"], "pos": cache[f"eff_pos_{s}"], "quat": cache[f"eff_quat_{s}"]}
    pose = cache[f"obj_pose_{spec['object']}"]
    boxes = np.asarray(spec["boxes"], dtype=np.float64)
    R = quat_to_mat(pose[:, 3:])
    points = np.einsum("tij,nj->tni", R, sample_box_surface(boxes)) + pose[:, None, :3]
    return {"points": points, "pos": pose[:, :3], "quat": pose[:, 3:], "boxes": boxes, "char_size": spec["char_size"]}


def pair_distance(a: dict, b: dict) -> np.ndarray:
    """Min surface distance (m) between entity a (points or boxes) and object entity b."""
    d = points_to_boxes_distance(a["points"], *boxes_to_world(b["boxes"], b["pos"], b["quat"]))
    if "boxes" in a:
        d = np.minimum(d, points_to_boxes_distance(b["points"], *boxes_to_world(a["boxes"], a["pos"], a["quat"])))
    return d


def relative_motion(a: dict, b: dict, dt: float, window_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Speed of b's pose expressed in a's frame: translational (m/s) and angular (rad/s)."""
    Ra, Rb = quat_to_mat(a["quat"]), quat_to_mat(b["quat"])
    p_rel = np.einsum("tji,tj->ti", Ra, b["pos"] - a["pos"])
    v = np.linalg.norm(smooth(p_rel, dt, window_sec, deriv=1), axis=-1)
    R_rel = np.einsum("tji,tjk->tik", Ra, Rb)
    step = Rotation.from_matrix(np.einsum("tji,tjk->tik", R_rel[:-1], R_rel[1:])).as_rotvec()
    w = np.linalg.norm(step, axis=-1) / dt
    w = smooth(np.concatenate([w[:1], w]), dt, window_sec)
    return v, np.maximum(w, 0.0)


def hysteresis(gap: np.ndarray, on: float, off: float) -> np.ndarray:
    """Contact starts when gap < on and ends only when gap > off."""
    out = np.zeros(len(gap), dtype=bool)
    state = False
    for i, g in enumerate(gap):
        state = g < on if not state else g <= off
        out[i] = state
    return out


def pair_signals(cache: dict, entities: dict, pair: dict, label_cfg: dict, dt: float) -> PairSignals:
    a, b = _entity(cache, entities[pair["a"]]), _entity(cache, entities[pair["b"]])
    win = label_cfg["smooth_sec"]
    g_raw = pair_distance(a, b) / b["char_size"]
    gap = np.maximum(smooth(g_raw, dt, win), 0.0)
    gap_dot = smooth(g_raw, dt, win, deriv=1)
    v, w = relative_motion(a, b, dt, label_cfg["motion_smooth_sec"])
    contact = hysteresis(gap, label_cfg["g_contact"], label_cfg["g_release"])
    return PairSignals(gap, gap_dot, v, w, contact)
