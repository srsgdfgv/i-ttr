#!/usr/bin/env python3
"""Verify KeepIdxEpisodicDataset matches official EpisodicDataset when keep=all frames."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ROOT / "third_party" / "act"))

from act.act_utils import EpisodicDataset, get_norm_stats  # noqa: E402
from adapters.act_dataset import KeepIdxEpisodicDataset, save_keep_idx  # noqa: E402


def main() -> None:
    task = "sim_insertion_human"
    data_dir = ROOT / "data" / task
    paths = sorted(data_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise SystemExit(f"no demos in {data_dir}")
    n = min(3, len(paths))
    ids = [int(p.stem.split("_")[1]) for p in paths[:n]]
    stats = get_norm_stats(str(data_dir), n)
    cams = ["top"]

    with tempfile.TemporaryDirectory() as tmp:
        keep_dir = Path(tmp)
        for i in ids:
            with h5py.File(data_dir / f"episode_{i}.hdf5", "r") as f:
                T = len(f["action"])
            save_keep_idx(keep_dir / f"episode_{i}.npz", np.arange(T), dt=0.02)

        torch.manual_seed(0)
        np.random.seed(0)
        official = EpisodicDataset(ids, str(data_dir), cams, stats, speedup=False)
        torch.manual_seed(0)
        np.random.seed(0)
        ours = KeepIdxEpisodicDataset(ids, str(data_dir), cams, stats, keep_dir=str(keep_dir))

        for idx in range(len(ids)):
            # Re-seed so both draw the same start index / keep index.
            torch.manual_seed(100 + idx)
            np.random.seed(100 + idx)
            a = official[idx]
            torch.manual_seed(100 + idx)
            np.random.seed(100 + idx)
            b = ours[idx]
            for name, x, y in zip(("image", "qpos", "action", "is_pad"), a, b):
                if not torch.allclose(x.float(), y.float(), atol=1e-5, rtol=1e-5):
                    raise SystemExit(f"mismatch on {name} episode_ids[{idx}] max|d|="
                                     f"{(x.float() - y.float()).abs().max().item()}")
        print(f"check_dataset_equiv: OK ({n} episodes, keep=all)")


if __name__ == "__main__":
    main()
