#!/usr/bin/env python3
"""Label sim_insertion_human demos with kinematic segmentation; write /labels_kin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# allow `python -m kin_seg.run_label_kin`; relative paths resolve against the repo root
_ROOT = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kin_seg.features import extract_features
from kin_seg.report import aggregate_metrics, episode_metrics, save_json, write_report_md
from kin_seg.segment import segment_episode, standardize_features


def _list_episodes(dataset_dir: Path, episode_ids: list[int] | None = None, max_episodes: int | None = None) -> list[Path]:
    paths = sorted(dataset_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 under {dataset_dir}")
    if episode_ids is not None:
        want = set(episode_ids)
        paths = [p for p in paths if int(p.stem.split("_")[1]) in want]
        if not paths:
            raise FileNotFoundError(f"No episodes matching ids {sorted(want)} under {dataset_dir}")
    if max_episodes is not None:
        paths = paths[: max_episodes]
    return paths


def _write_dataset(path: Path, name: str, data: np.ndarray) -> None:
    with h5py.File(path, "r+") as root:
        if name in root:
            del root[name]
        root.create_dataset(name, data=data)


def _plot_episode(
    out_path: Path,
    X4: np.ndarray,
    kin: np.ndarray,
    ref: np.ndarray | None,
) -> None:
    """X4: [T,4] = vL, wVarL, vR, wVarR for readable dual-arm curves."""
    t = np.arange(len(kin))
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, X4[:, 0], label="vL", lw=0.8)
    axes[0].plot(t, X4[:, 2], label="vR", lw=0.8)
    axes[0].set_ylabel("v_large")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(t, X4[:, 1], label="wVarL", lw=0.8)
    axes[1].plot(t, X4[:, 3], label="wVarR", lw=0.8)
    axes[1].set_ylabel("wrist_var")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t, kin, label="labels_kin", lw=1.0, drawstyle="steps-mid")
    if ref is not None:
        axes[2].plot(t[: len(ref)], ref, label="labels_entropy", lw=1.0, alpha=0.7, drawstyle="steps-mid")
    axes[2].set_ylabel("label")
    axes[2].set_xlabel("t")
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="data/sim_insertion_human")
    ap.add_argument("--out_dir", type=str, default="experiments/e1_lead/outputs/kin_seg")
    ap.add_argument("--method", choices=["gmm", "hdbscan"], default="gmm")
    ap.add_argument("--win", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dg_thresh", type=float, default=0.025)
    ap.add_argument("--gripper_half_window", type=int, default=6, help="frames; 6≈0.12s at DT=0.02")
    ap.add_argument("--hysteresis_n", type=int, default=5)
    ap.add_argument("--early_frac", type=float, default=0.14, help="force casual on first frac of episode")
    ap.add_argument("--speed_percentile", type=float, default=70.0, help="macro>ep-percentile → casual")
    ap.add_argument(
        "--precision_lead_frames",
        type=int,
        default=40,
        help="shift each precision segment start earlier by N frames (calibrated N*=40 ≈0.8s)",
    )
    ap.add_argument("--plot_eps", type=int, default=5, help="number of episodes to plot")
    ap.add_argument(
        "--episode_ids",
        type=str,
        default="",
        help="comma-separated episode ids for quick iteration, e.g. 0,1,2",
    )
    ap.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="only first N episodes (sorted by id); for smoke / formula tuning",
    )
    ap.add_argument("--global_standardize", action="store_true", default=False)
    ap.add_argument("--no_global_standardize", action="store_false", dest="global_standardize")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (_REPO / dataset_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_ids = None
    if args.episode_ids.strip():
        ep_ids = [int(x) for x in args.episode_ids.split(",") if x.strip() != ""]
    paths = _list_episodes(dataset_dir, episode_ids=ep_ids, max_episodes=args.max_episodes)
    print(f"Found {len(paths)} episodes in {dataset_dir}")
    feat_pack = []
    meta = []
    for p in paths:
        with h5py.File(p, "r") as root:
            qpos = root["/observations/qpos"][()]
            ref = root["/labels"][()] if "labels" in root else None
        feats = extract_features(qpos, win=args.win)
        feat_pack.append(feats)
        meta.append({"path": p, "ref": ref, "episode_id": int(p.stem.split("_")[1])})

    X_list = [f["X"] for f in feat_pack]
    if args.global_standardize:
        Xz_list, _ = standardize_features(X_list, fit_global=True)
    else:
        Xz_list = [None] * len(X_list)

    per_ep = []
    for i, (feats, m) in enumerate(zip(feat_pack, meta)):
        X = feats["X"]
        Xz = Xz_list[i]
        labels = segment_episode(
            X,
            feats["gripper_dg"],
            method=args.method,
            seed=args.seed,
            dg_thresh=args.dg_thresh,
            gripper_half_window=args.gripper_half_window,
            hysteresis_n=args.hysteresis_n,
            Xz=Xz if args.global_standardize else None,
            macro=feats.get("macro"),
            early_frac=args.early_frac,
            speed_percentile=args.speed_percentile,
            precision_lead_frames=args.precision_lead_frames,
        )
        _write_dataset(m["path"], "/labels_kin", labels.astype(np.float32))
        _write_dataset(m["path"], "/kin_features", X.astype(np.float32))

        ref = m["ref"]
        if ref is not None:
            met = episode_metrics(labels, ref)
            met["episode_id"] = m["episode_id"]
            per_ep.append(met)
        else:
            per_ep.append(
                {
                    "episode_id": m["episode_id"],
                    "n": len(labels),
                    "accuracy": float("nan"),
                    "precision_as_0": float("nan"),
                    "recall_as_0": float("nan"),
                    "kin_precision_frac": float((labels == 0).mean()),
                    "ref_precision_frac": float("nan"),
                    "confusion": {"tp0": 0, "fp0": 0, "fn0": 0, "tn1": 0},
                }
            )

        if i < args.plot_eps:
            _plot_episode(
                out_dir / "plots" / f"episode_{m['episode_id']:02d}.png",
                feats["X4"],
                labels,
                None if ref is None else np.asarray(ref).ravel(),
            )
        print(f"ep{m['episode_id']:02d}: kin_prec={(labels==0).mean():.2f} wrote {m['path'].name}")

    valid = [e for e in per_ep if e["n"] > 0 and e["accuracy"] == e["accuracy"]]
    agg = aggregate_metrics(valid) if valid else {"n_episodes": 0}
    write_report_md(out_dir / "report.md", agg, per_ep)
    save_json(out_dir / "report.json", {"aggregate": agg, "per_episode": per_ep})
    print(f"Report -> {out_dir / 'report.md'}")
    if agg:
        print(
            f"mean accuracy vs /labels: {agg.get('accuracy', float('nan')):.4f} "
            f"(pooled {agg.get('pooled_accuracy', float('nan')):.4f})"
        )


if __name__ == "__main__":
    main()
