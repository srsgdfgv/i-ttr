#!/usr/bin/env python3
"""Offline smoke test: Uniform / DemoSpeedup 2-4 / weight resampling without GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsr.baselines import keep_idx_demospeedup, keep_idx_uniform  # noqa: E402
from tsr.resample import resample_by_weights  # noqa: E402


def main() -> None:
    t = 400
    rho = 0.35
    rng = np.random.default_rng(0)

    u = keep_idx_uniform(t, rho)
    print(f"uniform keep={len(u)}/{t} ratio={len(u)/t:.3f}")
    assert abs(len(u) / t - rho) < 0.01

    for r in (0.3, 0.5, 0.9):
        k = resample_by_weights(rng.random(t), r)
        print(f"weight  rho={r} ratio={len(k)/t:.3f}")
        assert abs(len(k) / t - r) < 0.01

    for bad in (0.1, 1.2):
        try:
            resample_by_weights(rng.random(t), bad)
        except ValueError:
            print(f"weight  rho={bad} -> ValueError (expected)")
        else:
            raise AssertionError(f"rho={bad} should raise")

    labels = np.ones(t, dtype=np.int32)
    labels[250:320] = 0
    d = keep_idx_demospeedup(labels)
    strides = np.diff(d)
    in_prec = (d[:-1] >= 250) & (d[1:] < 320)
    in_casual = (d[1:] < 250)
    print(f"demospeedup keep={len(d)}/{t} prec strides={np.unique(strides[in_prec])} "
          f"casual strides={np.unique(strides[in_casual])}")
    assert set(np.unique(strides[in_prec])) == {2}
    assert set(np.unique(strides[in_casual])) <= {4}
    print("smoke_keep_idx: OK")


if __name__ == "__main__":
    main()
