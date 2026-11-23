#!/usr/bin/env python3
"""Collect lead-sweep result_*.txt into a markdown table under experiments/e1_lead/."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_result(path: Path) -> dict:
    text = path.read_text()
    out = {"path": str(path)}
    m = re.search(r"Success rate:\s*([0-9.]+)", text)
    out["success"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"Average return:\s*([0-9.]+)", text)
    out["return"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"Average length:\s*([0-9.]+)", text)
    out["length"] = float(m.group(1)) if m else float("nan")
    for r in range(5):
        m = re.search(rf"Reward >= {r}:.*?=\s*([0-9.]+)%", text)
        out[f"ge{r}"] = float(m.group(1)) / 100.0 if m else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="data/outputs/tsr/<task>/<RUN_TAG>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    for ckpt in sorted(run_dir.glob("*_s*")):
        if not ckpt.is_dir():
            continue
        results = list(ckpt.glob("result_*.txt"))
        if not results:
            continue
        # Prefer high_gain suffix when present.
        pref = [p for p in results if "high_gain" in p.name] or results
        info = parse_result(pref[0])
        info["group"] = ckpt.name
        keep_sum = ckpt / "keep_summary.json"
        if keep_sum.exists():
            info["rho"] = json.loads(keep_sum.read_text()).get("keep_ratio_mean")
        else:
            info["rho"] = None
        rows.append(info)

    out = Path(args.out or ROOT / "experiments" / "e1_lead" / f"lead_{run_dir.name}.md")
    lines = [
        f"# Lead sweep `{run_dir.name}`",
        "",
        f"Source: `{run_dir}`",
        "",
        "| group | success | return | avg_len | keep_ratio | ≥3 | ≥2 |",
        "|-------|---------|--------|---------|------------|----|----|",
    ]
    for r in rows:
        rho = f"{r['rho']:.3f}" if r["rho"] is not None else "-"
        lines.append(
            f"| {r['group']} | {r['success']:.2f} | {r['return']:.1f} | {r['length']:.1f} | "
            f"{rho} | {r.get('ge3', float('nan')):.2f} | {r.get('ge2', float('nan')):.2f} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(rows)} groups)")


if __name__ == "__main__":
    main()
