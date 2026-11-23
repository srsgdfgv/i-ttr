"""Extract kinematic features X(t) from Aloha 14-DoF qpos trajectories."""

from __future__ import annotations

import numpy as np

# Per-arm layout in qpos/action (14 = left 7 + right 7):
# 0 waist, 1 shoulder, 2 elbow, 3 forearm_roll, 4 wrist_angle, 5 wrist_rotate, 6 gripper
LARGE_IDX = (0, 1, 2)  # waist, shoulder, elbow
WRIST_IDX = (4, 5)
GRIPPER_IDX = 6


def _arm_slice(qpos: np.ndarray, arm: str) -> np.ndarray:
    if arm == "left":
        return qpos[:, 0:7]
    if arm == "right":
        return qpos[:, 7:14]
    raise ValueError(arm)


def _window_rms_velocity(joints: np.ndarray, win: int) -> np.ndarray:
    t = joints.shape[0]
    vel = np.zeros_like(joints)
    vel[1:] = joints[1:] - joints[:-1]
    vel[0] = vel[1] if t > 1 else 0.0
    speed = np.linalg.norm(vel, axis=-1)
    out = np.zeros(t, dtype=np.float64)
    half = win // 2
    for i in range(t):
        a = max(0, i - half)
        b = min(t, i - half + win)
        if b <= a:
            b = min(t, a + 1)
        out[i] = np.sqrt(np.mean(speed[a:b] ** 2))
    return out


def _window_wrist_variance(joints: np.ndarray, win: int) -> np.ndarray:
    t = joints.shape[0]
    out = np.zeros(t, dtype=np.float64)
    half = win // 2
    for i in range(t):
        a = max(0, i - half)
        b = min(t, i - half + win)
        if b - a < 2:
            out[i] = 0.0
            continue
        out[i] = float(np.mean(np.var(joints[a:b], axis=0)))
    return out


def arm_features(arm_qpos: np.ndarray, win: int = 5) -> tuple[np.ndarray, np.ndarray]:
    large = arm_qpos[:, list(LARGE_IDX)]
    wrist = arm_qpos[:, list(WRIST_IDX)]
    return _window_rms_velocity(large, win), _window_wrist_variance(wrist, win)


def gripper_delta(qpos: np.ndarray) -> np.ndarray:
    left_g = qpos[:, GRIPPER_IDX]
    right_g = qpos[:, 7 + GRIPPER_IDX]
    dg_l = np.zeros_like(left_g)
    dg_r = np.zeros_like(right_g)
    dg_l[1:] = np.abs(left_g[1:] - left_g[:-1])
    dg_r[1:] = np.abs(right_g[1:] - right_g[:-1])
    dg_l[0] = dg_l[1] if len(dg_l) > 1 else 0.0
    dg_r[0] = dg_r[1] if len(dg_r) > 1 else 0.0
    return np.maximum(dg_l, dg_r)


def extract_features(qpos: np.ndarray, win: int = 5) -> dict:
    """
    qpos: [T, 14]

    X columns (3D, dual-arm pooled — stabler for GMM than raw 4D):
      0: macro speed = max(vL, vR)
      1: wrist activity = max(wVarL, wVarR)
      2: log1p(wrist / (macro + eps))  relative micro/macro ratio

    Also returns legacy 4D arm-wise features under key X4 for plots.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    assert qpos.ndim == 2 and qpos.shape[1] == 14, qpos.shape
    v_l, w_l = arm_features(_arm_slice(qpos, "left"), win=win)
    v_r, w_r = arm_features(_arm_slice(qpos, "right"), win=win)
    macro = np.maximum(v_l, v_r)
    wrist = np.maximum(w_l, w_r)
    ratio = np.log1p(wrist / (macro + 1e-6))
    X = np.stack([macro, wrist, ratio], axis=-1)
    X4 = np.stack([v_l, w_l, v_r, w_r], axis=-1)
    return {"X": X, "X4": X4, "gripper_dg": gripper_delta(qpos), "macro": macro, "wrist": wrist}
