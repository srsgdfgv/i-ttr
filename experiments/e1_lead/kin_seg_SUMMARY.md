# 运动学分段 + Precision Lead

**代码**：[`../../kin_seg/`](../../kin_seg/)

## 标签对齐（vs 官方熵 `/labels`）

见 [`label_align_report.md`](label_align_report.md)（校准后 mean accuracy ≈ **0.735**）。

样例时间轴图：[`plots/`](plots/)

## Lead 扫参

见 [`lead_sweep.md`](lead_sweep.md)。选定 **N\*=40（≈0.8s @ DT=0.02）**。

## 下游 speedup（3000 epoch · 50 rollouts）

| Group | Labels | Success ≥4 | 原始 result |
|-------|--------|------------|-------------|
| D | kin，无 lead | **16%** | [`D_kin_no_lead.txt`](D_kin_no_lead.txt) |
| E | kin + lead40 | **28%** | [`E_kin_lead40.txt`](E_kin_lead40.txt) |

完整对照（含 A/B）：[`speedup_eval_lead.md`](speedup_eval_lead.md)
