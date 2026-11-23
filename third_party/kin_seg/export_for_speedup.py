#!/usr/bin/env python3
"""Copy dataset and map /labels_kin -> /labels for official --speedup (does not touch original data)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, default="data/sim_insertion_human")
    ap.add_argument("--dst_dir", type=str, default="data/sim_insertion_human_kin")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    if not src.is_absolute():
        src = (_REPO / src).resolve()
    if not dst.is_absolute():
        dst = (_REPO / dst).resolve()

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to replace")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    paths = sorted(src.glob("episode_*.hdf5"))
    if not paths:
        raise SystemExit(f"No episodes in {src}")

    for p in paths:
        out = dst / p.name
        shutil.copy2(p, out)
        with h5py.File(out, "r+") as root:
            if "labels_kin" not in root:
                raise SystemExit(f"{p.name} missing /labels_kin; run run_label_kin first")
            lab = root["labels_kin"][()].astype(np.float32)
            if "labels" in root:
                del root["labels"]
            root.create_dataset("labels", data=lab)
        print(f"exported {out.name}")

    print(f"Done: {len(paths)} episodes -> {dst}")
    print("Use kin_seg/run_speedup_wrapper.py with --task_name sim_insertion_human_kin")


if __name__ == "__main__":
    main()
