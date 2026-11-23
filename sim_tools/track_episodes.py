#!/usr/bin/env python3
"""Cache per-episode interaction states: FK finger points + recovered object poses.

Primary path: hard-set recorded robot qpos each frame and let MuJoCo physics move
objects; frames with poor render-vs-recorded IoU fall back to visual tracking.

Run from the repo root with MUJOCO_GL=egl. Output: <out_dir>/episode_<i>.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "third_party", ROOT / "third_party" / "act"):
    sys.path.insert(0, str(p))

from tsr.embodiment.fk_keypoints import AlohaScene, effector_trajectories  # noqa: E402
from tsr.perception.gt_replay import SOURCE_PHYSICS, SOURCE_WELD, ObjectStateRecoverer  # noqa: E402


def track_one(cfg: dict, h5_path: Path, out_path: Path, max_frames: int | None) -> str:
    t0 = time.time()
    scene = AlohaScene(str(ROOT / cfg["xml"]))
    recoverer = ObjectStateRecoverer(scene, cfg["objects"], cfg["camera"], cfg["track"])
    with h5py.File(h5_path, "r") as f:
        qpos = f["observations/qpos"][()]
        if max_frames:
            qpos = qpos[:max_frames]
        objs = recoverer.recover(qpos, f[f"observations/images/{cfg['camera']}"])
    effs = effector_trajectories(scene, qpos)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, dt=cfg["dt"], qpos=qpos, **effs, **objs)

    parts = []
    for n in cfg["objects"]:
        iou = objs[f"iou_{n}"]
        src = objs[f"source_{n}"]
        locked = float(((src == SOURCE_PHYSICS) | (src == SOURCE_WELD)).mean())
        parts.append(f"{n}:IoU={iou.mean():.2f}/locked={locked:.0%}")
    return f"{h5_path.stem}: T={len(qpos)} {' '.join(parts)} ({time.time() - t0:.0f}s)"


def _worker(job):
    return track_one(*job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_cfg", required=True)
    ap.add_argument("--dataset_dir", default=None, help="default data/<task>")
    ap.add_argument("--out_dir", default=None, help="default data/outputs/tsr/<task>/track")
    ap.add_argument("--episodes", default="", help="comma-separated ids; empty = all")
    ap.add_argument("--max_episodes", type=int, default=None, help="first N episodes by id")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.task_cfg))
    data_dir = Path(args.dataset_dir or ROOT / "data" / cfg["task"])
    out_dir = Path(args.out_dir or ROOT / "data" / "outputs" / "tsr" / cfg["task"] / "track")
    paths = sorted(data_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    if args.episodes:
        want = {int(x) for x in args.episodes.split(",")}
        paths = [p for p in paths if int(p.stem.split("_")[1]) in want]
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    jobs = []
    for p in paths:
        out = out_dir / f"{p.stem}.npz"
        if args.skip_existing and out.exists():
            print(f"[skip] {out.name}", flush=True)
            continue
        jobs.append((cfg, p, out, args.max_frames))
    print(f"Tracking {len(jobs)} episodes -> {out_dir}")
    lines = []
    if args.workers > 1 and jobs:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for line in pool.imap_unordered(_worker, jobs):
                print(line, flush=True)
                lines.append(line)
    else:
        for job in jobs:
            line = _worker(job)
            print(line, flush=True)
            lines.append(line)
    (out_dir / "track_log.json").write_text(json.dumps({"lines": lines}, indent=2))


if __name__ == "__main__":
    main()
