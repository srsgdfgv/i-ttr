#!/usr/bin/env python3
"""Fail fast with a clear reason before a long lead sweep."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def has_key(f, key: str) -> bool:
    node = f
    for part in key.split("/"):
        if part not in node:
            return False
        node = node[part]
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sim_insertion_human")
    ap.add_argument("--num_episodes", type=int, default=None)
    ap.add_argument("--need_labels", action="store_true", help="require /labels in HDF5")
    ap.add_argument("--skip_smoke", action="store_true")
    ap.add_argument("--skip_render", action="store_true")
    args = ap.parse_args()

    try:
        import torch
    except ImportError as e:
        fail(f"torch import: {e}")
    ok(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        fail("CUDA not available")

    data = ROOT / "data" / args.task
    paths = sorted(data.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        fail(f"no episode_*.hdf5 under {data} (run scripts/link_data.sh)")
    n = args.num_episodes or len(paths)
    if len(paths) < n:
        fail(f"need {n} demos, found {len(paths)} in {data}")
    required = ["action", "observations/qpos", "observations/images/top"]
    if args.need_labels:
        required.append("labels")
    for p in paths[:n]:
        with h5py.File(p, "r") as f:
            for key in required:
                if not has_key(f, key):
                    fail(f"{p.name} missing {key}")
    ok(f"{n} demos in {data}")

    out = ROOT / "data" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    probe = out / ".write_test"
    probe.write_text("ok")
    probe.unlink()
    free_gb = shutil.disk_usage(out).free / 1e9
    ok(f"data/outputs writable, free={free_gb:.1f} GB")
    if free_gb < 20:
        fail(f"low disk ({free_gb:.1f} GB free); need headroom for ckpts")

    if not args.skip_render:
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYTHONPATH"] = f"{ROOT}:{ROOT}/third_party:{ROOT}/third_party/act:" + env.get("PYTHONPATH", "")
        code = (
            "from dm_control import mujoco; "
            "p=mujoco.Physics.from_xml_path('third_party/act/assets/bimanual_viperx_insertion.xml'); "
            "im=p.render(480,640,camera_id='top'); "
            "assert im.shape==(480,640,3), im.shape; print('render', float(im.mean()))"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"EGL render failed:\n{r.stderr[-500:]}")
        ok("MUJOCO_GL=egl render")

    if not args.skip_smoke:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_tsr.py")], cwd=ROOT)
        if r.returncode != 0:
            fail("scripts/smoke_tsr.py failed")
        ok("smoke_tsr")

    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
