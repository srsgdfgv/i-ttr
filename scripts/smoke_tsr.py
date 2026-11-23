#!/usr/bin/env python3
"""Offline smoke test of the TSR label -> keep_idx path on synthetic signals (no GPU, no data)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.act_dataset import save_keep_idx  # noqa: E402
from tsr.geometry import boxes_to_world, points_to_boxes_distance  # noqa: E402
from tsr.label.binary import frequency_from_binary_with_lead  # noqa: E402
from tsr.label.combine import chunk_lead_envelope, lead_max_filter  # noqa: E402
from tsr.label.contact_term import f_contact  # noqa: E402
from tsr.label.event_term import f_event  # noqa: E402
from tsr.label.tau_term import f_free  # noqa: E402
from tsr.resample import keep_idx_from_frequency  # noqa: E402
from tsr.signals import smooth  # noqa: E402



def main() -> None:
    dt, T = 0.02, 300
    f_max = 1 / dt
    t = np.arange(T) * dt

    # point-to-box distance: unit box at origin
    pts = np.array([[[2.0, 0, 0], [0.5, 0.5, 0.5], [0, 3.0, 4.0]]])
    d = points_to_boxes_distance(pts, *boxes_to_world([[0, 0, 0, 1, 1, 1]], np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]])))
    assert np.isclose(d[0], 0.0), d
    pts = np.array([[[3.0, 0, 0]]])
    d = points_to_boxes_distance(pts, *boxes_to_world([[0, 0, 0, 1, 1, 1]], np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]])))
    assert np.isclose(d[0], 2.0), d

    # derivative in per-second units, window in seconds
    x = 3.0 * t
    assert np.allclose(smooth(x, dt, 0.12, deriv=1)[5:-5], 3.0), "smooth derivative"

    # approach 5 -> 0 over 3 s, contact for 2 s, then recede
    gap = np.concatenate([np.linspace(5, 0, 150), np.zeros(100), np.linspace(0, 2, 50)])
    gdot = np.gradient(gap, dt)
    ff = f_free(gap, gdot, kappa=0.1, g0=0.25, f_max=f_max)
    assert ff.max() <= f_max and ff[-40:].max() == 0.0, "free term clipped / zero when receding"
    assert ff[140] > ff[20], "free term rises as gap closes"
    print(f"f_free far={ff[20]:.1f}Hz near={ff[140]:.1f}Hz")

    contact = gap < 0.15
    v = np.where(contact, 0.004, 0.0)
    w = np.where(contact, np.deg2rad(1.0), 0.0)
    fc = f_contact(v, w, contact, eps_pos=0.002, eps_rot=np.deg2rad(2.0), f_max=f_max)
    assert np.isclose(fc[contact].max(), 2.0) and fc[~contact].max() == 0.0, "contact = max(v/eps_pos, w/eps_rot)"

    fe = f_event(contact, dt, pre_sec=0.1, post_sec=0.1, f_max=f_max)
    onset = int(np.flatnonzero(contact)[0])
    assert fe[onset - 5] == f_max and fe[onset - 6] == 0.0, "event pre-window is 0.1 s"

    f = np.maximum.reduce([ff, fc, fe])
    step = np.zeros(T)
    step[200:] = 10.0
    led = lead_max_filter(step, 0.4, dt)
    assert led[180] == 10.0 and led[179] == 0.0, "lead extends 0.4 s earlier"
    env = chunk_lead_envelope(step, 0.5, dt)
    assert env[199] > env[150] > 0 and env[200] == 10.0, "chunk-lead decays backward"

    s_max = 4
    k_nat = keep_idx_from_frequency(f, dt, s_max)
    r_nat = len(k_nat) / T
    assert 1 / s_max - 0.02 <= r_nat <= 1.0 and np.diff(k_nat).max() <= s_max
    k_bud = keep_idx_from_frequency(f, dt, s_max, rho=0.4)
    assert abs(len(k_bud) / T - 0.4) < 0.02, len(k_bud) / T
    dense = np.diff(k_bud)[(k_bud[:-1] > onset - 5) & (k_bud[:-1] < onset + 5)]
    assert dense.max() == 1, "event window keeps every frame"
    print(f"keep natural={r_nat:.3f} budget(0.4)={len(k_bud) / T:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "episode_0.npz"
        save_keep_idx(p, k_bud, dt)
        z = np.load(p)
        assert np.allclose(z["t_keep"], k_bud * dt) and np.allclose(np.diff(z["t_keep"]), z["dt_keep"][:-1])

    # Binary labels share the same time-based lead as TSR.
    labels = np.ones(T, dtype=np.int32)
    labels[200:240] = 0
    fb = frequency_from_binary_with_lead(labels, dt, s_max=4, lead_sec=0.4)
    assert fb[180] == 1 / dt and fb[179] == (1 / dt) / 4, "binary lead 0.4s"
    print("smoke_tsr: OK")


if __name__ == "__main__":
    main()
