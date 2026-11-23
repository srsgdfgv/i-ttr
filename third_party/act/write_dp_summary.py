#!/usr/bin/env python3
"""Write DP repro results markdown from result_*.txt files."""
import argparse
import re
from pathlib import Path

DT = 0.02


def parse_result(path: Path):
    text = path.read_text()
    sr = float(re.search(r"Success rate:\s*([0-9.]+)", text).group(1))
    ret = float(re.search(r"Average return:\s*([0-9.]+)", text).group(1))
    length = float(re.search(r"Average length:\s*([0-9.]+)", text).group(1))
    ladder = {}
    for m in re.finditer(
        r"Reward >= (\d+):\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%", text
    ):
        ladder[int(m.group(1))] = (
            int(m.group(2)),
            int(m.group(3)),
            float(m.group(4)),
        )
    return {
        "success_rate": sr,
        "avg_return": ret,
        "avg_length": length,
        "avg_time": length * DT,
        "ladder": ladder,
        "n_success": round(sr * 50),
    }


def fmt_pct(sr: float) -> str:
    return f"{sr * 100:.0f}% ({round(sr * 50)}/50)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_ckpt", required=True)
    ap.add_argument("--speed_ckpt", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proxy = Path(args.proxy_ckpt)
    speed = Path(args.speed_ckpt)
    a_path = proxy / "result_policy_last.txt"
    c_path = proxy / "result_policy_last_high_gain.txt"
    b_path = speed / "result_policy_last.txt"

    missing = [p for p in (a_path, b_path, c_path) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing result files: {missing}")

    A = parse_result(a_path)
    B = parse_result(b_path)
    C = parse_result(c_path)

    def row(name, g):
        return (
            f"| {name} | {fmt_pct(g['success_rate'])} | {g['avg_length']} | "
            f"{g['avg_time']:.3f}s | {g['avg_return']} |"
        )

    # best ckpt lines if present
    def best_line(log_path: Path) -> str:
        if not log_path.exists():
            return "N/A"
        lines = [
            ln.strip()
            for ln in log_path.read_text(errors="ignore").splitlines()
            if "Best ckpt" in ln or "Training finished" in ln
        ]
        return lines[-1] if lines else "N/A"

    root = Path(args.out).resolve().parents[1]  # aloha/
    proxy_log = root / f"train_dp_proxy_e{args.epochs}.log"
    speed_log = root / f"train_dp_speedup_e{args.epochs}.log"

    md = f"""# DemoSpeedup 复现实验结果（Diffusion Policy）

**设定**：`sim_insertion_human` · DP · {args.epochs} epochs · seed 0 · batch={args.batch} · 每组 50 rollouts · 成功 = reward≥4 · DT=0.02s  
**Avg time** = Avg length × DT（三位小数）

## 诚实边界

- 单 seed、{args.epochs}-epoch 代理训练；论文为更长训练（如 16000）与多 seed 平均。
- 官方建议 DP `batch=40`；本机 RTX 4060 Laptop 8GB 使用 **batch={args.batch}**。
- 仅作与 ACT 同协议的趋势验证，不声称精确复现论文 Table。

## 消融/对照表（A / B / C）

| Group | Success | Avg length (steps) | Avg time | Avg return |
|-------|---------|--------------------|----------|------------|
{row("A Baseline + normal gain", A)}
{row("B DemoSpeedup + high gain", B)}
{row("C Baseline + high gain", C)}

**解读**：比较 A 与 C 可检验单独提高夹爪增益是否解释收益；B 为 DemoSpeedup 论文评估设定。

## Reward 阶梯（成功档 ≥4）

| Threshold | A | B | C |
|-----------|---|---|---|
"""
    for thr in range(5):
        cells = []
        for g in (A, B, C):
            n, tot, pct = g["ladder"].get(thr, (0, 50, 0.0))
            cells.append(f"{n}/{tot} ({pct:g}%)")
        md += f"| ≥{thr} | " + " | ".join(cells) + " |\n"

    md += f"""
## 训练摘要

- Proxy: `{best_line(proxy_log)}`
- Speedup: `{best_line(speed_log)}`
- Proxy ckpt: `{args.proxy_ckpt}`
- Speedup ckpt: `{args.speed_ckpt}`

## 与 ACT 本地协议对照

ACT 同任务同 epoch/seed 协议下约为 A 20% / 7.168s，B 34% / 3.491s，C 20% / 7.164s（见 `learn_track/repro_results.md`）。本文件为 DP 同构实验数字。
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
