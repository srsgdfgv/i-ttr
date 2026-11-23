# Precision lead-frame sweep

Hypothesis: shift each precision onset earlier by N frames to protect approach.
Reference = entropy `/labels`. DT=0.02s.

## Selection rule

- Prefer largest drop in **20–40% FN** (kin=casual, entropy=precision)
- Keep global accuracy drop ≤ 0.02 vs N=0
- Tie-break: smaller N

**Selected N\* = 40 (0.80s)**

## Results

| N | sec | accuracy | kin_prec | FN 0–20% | FN 20–40% | FN 70–100% | mean onset lag (kin-ent frames) |
|---|-----|----------|----------|----------|-----------|------------|----------------------------------|
| 0 | 0.00 | 0.7352 | 0.620 | 0.170 | 0.199 | 0.049 | 8.3 |
| 10 | 0.20 | 0.7603 | 0.722 | 0.104 | 0.092 | 0.013 | 0.3 |
| 15 | 0.30 | 0.7586 | 0.756 | 0.082 | 0.067 | 0.008 | -2.7 |
| 20 | 0.40 | 0.7543 | 0.783 | 0.070 | 0.046 | 0.004 | -4.5 |
| 25 | 0.50 | 0.7490 | 0.805 | 0.061 | 0.031 | 0.002 | -5.6 |
| 30 | 0.60 | 0.7427 | 0.822 | 0.055 | 0.020 | 0.002 | -6.3 |
| 40 | 0.80 | 0.7347 | 0.844 | 0.050 | 0.004 | 0.002 | -6.8 | ← selected

## Notes

- Baseline N=0: acc=0.7352, mid_fn=0.199, lag=8.3
- Selected: acc=0.7347 (Δ=-0.0005), mid_fn=0.004 (Δ=-0.195), lag=-6.8
