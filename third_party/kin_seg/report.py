"""Compare /labels_kin vs entropy /labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def episode_metrics(kin: np.ndarray, ref: np.ndarray) -> dict:
    kin = np.asarray(kin).astype(np.int32).ravel()
    ref = np.asarray(ref).astype(np.int32).ravel()
    n = min(len(kin), len(ref))
    kin, ref = kin[:n], ref[:n]
    acc = float((kin == ref).mean()) if n else 0.0
    # precision class = 0 as positive for "precision detection"
    tp = int(((kin == 0) & (ref == 0)).sum())
    fp = int(((kin == 0) & (ref == 1)).sum())
    fn = int(((kin == 1) & (ref == 0)).sum())
    tn = int(((kin == 1) & (ref == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "precision_as_0": prec,
        "recall_as_0": rec,
        "kin_precision_frac": float((kin == 0).mean()) if n else 0.0,
        "ref_precision_frac": float((ref == 0).mean()) if n else 0.0,
        "confusion": {"tp0": tp, "fp0": fp, "fn0": fn, "tn1": tn},
    }


def aggregate_metrics(per_ep: list[dict]) -> dict:
    if not per_ep:
        return {}
    keys = ["accuracy", "precision_as_0", "recall_as_0", "kin_precision_frac", "ref_precision_frac"]
    agg = {k: float(np.mean([e[k] for e in per_ep])) for k in keys}
    agg["n_episodes"] = len(per_ep)
    # pooled confusion
    tp = sum(e["confusion"]["tp0"] for e in per_ep)
    fp = sum(e["confusion"]["fp0"] for e in per_ep)
    fn = sum(e["confusion"]["fn0"] for e in per_ep)
    tn = sum(e["confusion"]["tn1"] for e in per_ep)
    agg["pooled_accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1)
    agg["pooled_confusion"] = {"tp0": tp, "fp0": fp, "fn0": fn, "tn1": tn}
    return agg


def write_report_md(path: Path, agg: dict, per_ep: list[dict], notes: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    default_notes = (
        "High agreement ⇒ kinematic prior close to entropy segmentation. "
        "Low agreement does not mean failure; inspect plots for physically plausible segments.\n\n"
        "Calibration (50 insertion demos): main systematic bug was **early FP** "
        "(t≈0–20%: kin=precision while entropy=casual). Cause: gripper override "
        "re-painted transport peaks after early-casual prior. Fix: pipeline order "
        "`cluster → speed_gate → gripper → early_casual → hysteresis`, "
        "`early_frac=0.14`, `speed_percentile=70`, milder gripper (`dg=0.025`, `hw=6`)."
    )
    lines = [
        "# Kinematic vs Entropy Label Report",
        "",
        "Label semantics: `0` = precision (less speedup), `1` = casual (more speedup).",
        "Reference = existing HDF5 `/labels` from DemoSpeedup entropy pipeline.",
        "",
        "## Aggregate",
        "",
        f"- episodes: {agg.get('n_episodes', 0)}",
        f"- mean accuracy: {agg.get('accuracy', 0):.4f}",
        f"- pooled accuracy: {agg.get('pooled_accuracy', 0):.4f}",
        f"- mean precision (class 0): {agg.get('precision_as_0', 0):.4f}",
        f"- mean recall (class 0): {agg.get('recall_as_0', 0):.4f}",
        f"- mean kin precision frac: {agg.get('kin_precision_frac', 0):.4f}",
        f"- mean ref precision frac: {agg.get('ref_precision_frac', 0):.4f}",
        f"- pooled confusion (tp0/fp0/fn0/tn1): {agg.get('pooled_confusion')}",
        "",
        "## Notes",
        "",
        notes or default_notes,
        "",
        "## Per episode",
        "",
        "| ep | accuracy | prec0 | rec0 | kin_prec_frac | ref_prec_frac |",
        "|----|----------|-------|------|---------------|---------------|",
    ]
    for e in per_ep:
        lines.append(
            f"| {e['episode_id']} | {e['accuracy']:.3f} | {e['precision_as_0']:.3f} | "
            f"{e['recall_as_0']:.3f} | {e['kin_precision_frac']:.3f} | {e['ref_precision_frac']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
