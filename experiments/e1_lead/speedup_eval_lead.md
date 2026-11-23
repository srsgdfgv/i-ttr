# Kin + precision-lead (N*=40) speedup eval

**Setting**: `sim_insertion_human` · ACT · 3000 epochs · seed 0 · 50 rollouts · success = reward≥4 · DT=0.02s  
**Avg time** = Avg length × DT

## Groups

| Group | Labels | Gripper env | Success ≥4 | ≥3 | ≥2 | ≥1 | Avg success length | Avg time | Avg return |
|-------|--------|-------------|------------|----|----|----|--------------------|----------|------------|
| **A** Baseline | none (proxy ACT) | normal | **20% (10/50)** | 52% | 62% | 82% | 358.4 | 7.17s | 281 |
| **B** Entropy DemoSpeedup | proxy entropy `/labels` | high | **34% (17/50)** | 72% | 86% | 90% | 174.5 | 3.49s | 422 |
| **D** Kin (no lead) | kinematic X(t) | high | **16% (8/50)** | 72% | 76% | 86% | 192.5 | 3.85s | 597 |
| **E** Kin + lead40 | kinematic + precision onset −40 frames | high | **28% (14/50)** | 68% | 76% | 90% | 182.2 | 3.64s | 482 |

E ckpt: `data/outputs/ACT_kin_speedup_ckpt/sim_insertion_human_lead40_e3000/`  
Best val loss **0.0654 @ epoch 2837** (comparable to B entropy speedup ~0.069).

## Hypothesis check (precision lead)

Video failure mode for D: approach-to-insert too fast → overshoot → stuck at reward 3.  
Fix: shift each precision segment start earlier by **N\*=40 (~0.8s)** (`lead_sweep.md`).

| Metric | D → E | Verdict |
|--------|-------|---------|
| Success ≥4 | 16% → **28%** | **Clear gain** vs D; **beats A (20%)** |
| vs B entropy | 28% vs 34% | Still **−6 pp**; not fully closed |
| ≥3 (contact) | 72% → 68% | Slight drop; less “stuck at 3” pile-up than D’s high return |
| Avg return | 597 → 482 | Down (fewer long reward-3 hangs) |
| Speed | 3.85s → 3.64s success | Still much faster than A (7.17s) |

## Conclusion

1. **Lead-N is validated on this task**: protecting approach frames lifts kin speedup from worse-than-baseline (16%) to **better-than-baseline (28%)**.
2. **Entropy labeling remains stronger** (34%): kinematic + lead is a cheaper approximate, not a full substitute.
3. Remaining gap to B is small enough to justify further feature work (e.g. end-effector / tangent) *or* modest N retuning—not abandoning the kinematic route.

## One-liner

> Without lead: kin speedup harms insertion (16% < A 20%). With lead40: 28%, between A and entropy B (34%), with success time ~3.6s vs baseline ~7.2s.
