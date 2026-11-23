#!/usr/bin/env python3
"""Build keep_idx sidecars for tsr / entropy / kin / full (+ shared time-based lead).

Examples:
  python scripts/make_keep_idx.py --task_cfg configs/tasks/sim_insertion_human.yaml \\
      --method tsr --lead_sec 0.4 --rho 0.45 --tag tsr_lead0.4
  python scripts/make_keep_idx.py ... --method entropy --lead_sec 0
  python scripts/make_keep_idx.py ... --method kin --lead_sec 0.8
  python scripts/make_keep_idx.py ... --method full --tag full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from adapters.act_dataset import save_keep_idx  # noqa: E402
from tsr.baselines import keep_idx_demospeedup, keep_idx_full  # noqa: E402
from tsr.label.binary import frequency_from_binary_with_lead  # noqa: E402
from tsr.label.combine import label_episode  # noqa: E402
from tsr.resample import keep_idx_from_frequency  # noqa: E402


def plot_tsr(path: Path, lab: dict, keep: np.ndarray, dt: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(lab["f"])) * dt
    pairs = sorted({k.split("/", 1)[1] for k in lab if k.startswith("gap/")})
    nrows = max(1, len(pairs)) + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.2 * nrows), sharex=True)
    if nrows == 2 and not pairs:
        axes = [axes]
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pairs):
        for term in ("free", "contact", "event"):
            ax.plot(t, lab[f"f_{term}/{p}"], lw=0.8, label=term)
        ax2 = ax.twinx()
        ax2.plot(t, lab[f"gap/{p}"], "k--", lw=0.6)
        ax2.set_ylim(0, 5)
        ax.set_ylabel(f"{p}\nHz")
        ax.legend(loc="upper left", fontsize=7)
    axes[-1].plot(t, lab["f"], lw=0.9, label="f")
    axes[-1].vlines(keep * dt, 0, 1 / dt, color="r", lw=0.3, alpha=0.5, label="kept")
    axes[-1].set_xlabel("time (s)")
    axes[-1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _kin_labels(qpos: np.ndarray) -> np.ndarray:
    from kin_seg.features import extract_features
    from kin_seg.segment import segment_episode

    feats = extract_features(qpos)
    return segment_episode(
        feats["X"],
        feats["gripper_dg"],
        method="gmm",
        macro=feats["macro"],
        precision_lead_frames=0,
    )


def _entropy_budget_ratio(dataset_dir: Path, paths: list[Path]) -> float:
    ratios = []
    for p in paths:
        path = p if p.suffix == ".hdf5" else dataset_dir / f"{p.stem}.hdf5"
        with h5py.File(path, "r") as f:
            if "labels" not in f:
                raise KeyError(f"{path}: missing /labels (needed for shared rho from entropy)")
            labels = f["labels"][()]
        keep = keep_idx_demospeedup(labels)
        ratios.append(len(keep) / len(labels))
    return float(np.mean(ratios))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_cfg", required=True)
    ap.add_argument("--method", choices=["tsr", "entropy", "kin", "full"], default="tsr")
    ap.add_argument("--track_dir", default=None, help="required for method=tsr")
    ap.add_argument("--dataset_dir", default=None, help="HDF5 dir; default data/<task>")
    ap.add_argument("--tag", default=None, help="default <method>_lead<sec>")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--rho", type=float, default=None, help="keep ratio; default: entropy 2/4 mean or natural")
    ap.add_argument("--lead_sec", type=float, default=None)
    ap.add_argument("--chunk_lead_sec", type=float, default=None)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--plot", type=int, default=3)
    ap.add_argument("--rho_from_entropy", action="store_true",
                    help="set rho to mean DemoSpeedup 2/4 keep ratio on /labels")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.task_cfg))
    for k in ("lead_sec", "chunk_lead_sec"):
        if getattr(args, k) is not None:
            cfg["label"][k] = getattr(args, k)
    lead_sec = float(cfg["label"].get("lead_sec", 0.0))
    s_max, dt = int(cfg["resample"]["s_max"]), float(cfg["dt"])
    dataset_dir = Path(args.dataset_dir or ROOT / "data" / cfg["task"])
    base = ROOT / "data" / "outputs" / "tsr" / cfg["task"]
    tag = args.tag or (args.method if args.method == "full" else f"{args.method}_lead{lead_sec:g}")
    out_dir = Path(args.out_dir or base / "keep" / tag)

    if args.method == "tsr":
        track_dir = Path(args.track_dir or base / "track")
        paths = sorted(track_dir.glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(f"no track cache in {track_dir}")
    else:
        paths = sorted(dataset_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(f"no HDF5 in {dataset_dir}")
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]

    h5_paths = sorted(dataset_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    if args.max_episodes is not None:
        h5_paths = h5_paths[: args.max_episodes]

    if args.method == "full":
        rho = None
    elif args.rho is not None:
        rho = args.rho
    elif args.rho_from_entropy or cfg["resample"]["rho"] is None:
        rho = _entropy_budget_ratio(dataset_dir, h5_paths)
        print(f"shared rho from entropy 2/4 = {rho:.4f}")
    else:
        rho = cfg["resample"]["rho"]

    summary = []
    for k, p in enumerate(paths):
        ep = p.stem
        lab = None
        if args.method == "full":
            with h5py.File(dataset_dir / f"{ep}.hdf5", "r") as f:
                T = len(f["action"])
            keep = keep_idx_full(T)
            f_arr = np.full(T, 1.0 / dt)
        elif args.method == "tsr":
            with np.load(p) as z:
                cache = dict(z)
            lab = label_episode(cache, cfg)
            f_arr = lab["f"]
            keep = keep_idx_from_frequency(f_arr, dt, s_max, rho)
        elif args.method == "entropy":
            with h5py.File(dataset_dir / f"{ep}.hdf5", "r") as f:
                labels = f["labels"][()]
            f_arr = frequency_from_binary_with_lead(labels, dt, s_max, lead_sec)
            keep = keep_idx_from_frequency(f_arr, dt, s_max, rho)
        elif args.method == "kin":
            with h5py.File(dataset_dir / f"{ep}.hdf5", "r") as f:
                qpos = f["observations/qpos"][()]
            labels = _kin_labels(qpos)
            f_arr = frequency_from_binary_with_lead(labels, dt, s_max, lead_sec)
            keep = keep_idx_from_frequency(f_arr, dt, s_max, rho)
        else:
            raise ValueError(args.method)

        extra = {"f": f_arr, "method": np.array(args.method)}
        if lab is not None:
            extra.update({key.replace("/", "__"): v for key, v in lab.items()})
        save_keep_idx(out_dir / f"{ep}.npz", keep, dt, **extra)
        summary.append({"episode": ep, "T": len(f_arr), "K": int(len(keep)), "ratio": float(len(keep) / len(f_arr))})
        if k < args.plot and lab is not None:
            plot_tsr(out_dir / "plots" / f"{ep}.png", lab, keep, dt)

    ratios = np.array([s["ratio"] for s in summary])
    agg = {
        "task": cfg["task"], "method": args.method, "tag": tag, "rho": rho,
        "lead_sec": lead_sec, "s_max": s_max, "n_episodes": len(summary),
        "keep_ratio_mean": float(ratios.mean()), "keep_ratio_min": float(ratios.min()),
        "keep_ratio_max": float(ratios.max()), "episodes": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2))
    print(f"{args.method} x{len(summary)} -> {out_dir}  keep ratio mean {ratios.mean():.3f} "
          f"[{ratios.min():.3f}, {ratios.max():.3f}]")


if __name__ == "__main__":
    main()
