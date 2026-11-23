# Kinematic segmentation（Aloha Insertion）

独立于官方 `act/` 训练逻辑的运动学分段模块：用示范 `qpos` 近似 DemoSpeedup 的 precision(0) / casual(1) 标签，再导出给 `--speedup` 训练。

完整实验叙事与对照表：[`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) · 轻量结果：[`../experiments/kin_seg/`](../experiments/kin_seg/)

## Label 语义

与官方 `process_action_label` 一致：

- `0` = precision（少加速）
- `1` = casual（多加速）

## Pipeline

```text
HDF5 qpos
  → X(t)=[macro, wrist, log1p(wrist/macro)]
  → GMM(K=2)
  → speed_gate → gripper override → early_casual → hysteresis
  → precision_lead (N frames) → reapply early_casual
  → /labels_kin
  → export: copy 并把 /labels := /labels_kin
  → run_speedup_wrapper --speedup
```

## 关键结果（本仓库实测）

| 设定 | vs 熵标签一致率 | Speedup 成功率 (50 rollouts) |
|------|-----------------|------------------------------|
| kin（校准后，N=0） | ~73.5% | **16%**（差于 baseline 20%） |
| kin + **lead N\*=40** | 标签侧中段 FN 大降 | **28%**（优于 baseline，仍低于熵版 34%） |

## 快速命令

在 tsresample 仓库根目录运行（相对路径均相对仓库根）：

```bash
export PYTHONPATH="$PWD:$PWD/third_party:$PWD/third_party/act:$PYTHONPATH"

# 1) 标注（默认 precision_lead_frames=40）
# 注意：会把 /labels_kin 原地写进 --dataset_dir 的 HDF5，共享盘上请先拷一份再跑
python -m kin_seg.run_label_kin \
  --dataset_dir data/sim_insertion_human \
  --out_dir experiments/e1_lead/outputs/kin_lead40 \
  --method gmm --plot_eps 5

# 2) 导出副本供官方 --speedup
python -m kin_seg.export_for_speedup --overwrite

# 3) 训练 / 评估（不改 act/constants.py）
python third_party/kin_seg/run_speedup_wrapper.py \
  --task_name sim_insertion_human_kin \
  --ckpt_dir data/outputs/ACT_kin_speedup_ckpt/sim_insertion_human_lead40_e3000 \
  --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 \
  --batch_size 8 --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 \
  --seed 0 --temporal_agg --speedup

python third_party/kin_seg/run_speedup_wrapper.py ... --speedup --eval
```

关闭 lead：`--precision_lead_frames 0`。

## 模块说明

| 文件 | 作用 |
|------|------|
| `features.py` | 从 qpos 提取 X(t) |
| `segment.py` | GMM + 后处理 + `apply_precision_lead` |
| `run_label_kin.py` | 批量写 `/labels_kin`、出报告/图 |
| `export_for_speedup.py` | 复制数据并映射标签 |
| `run_speedup_wrapper.py` | 运行时注册 task，调用官方 `imitate_episodes.py` |
| `report.py` | 与 `/labels` 的逐帧指标 |

## 设计约束

- **不修改** `act/` 官方训练核心；消融评估见 `act/imitate_episodes_ablation.py`。
- 熵标签是对齐参考，不是 oracle；验收看 speedup 后再训的 rollout。
