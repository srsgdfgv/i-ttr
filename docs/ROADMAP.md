# 路线图

## 已落地（V1）

| 项 | 位置 |
|----|------|
| 物体状态恢复（硬设 qpos + 物理 + 夹持锁定） | `tsr/perception/gt_replay.py`、`sim_tools/track_episodes.py` |
| 交互图标签 free / contact / event + Lead | `tsr/label/`、`configs/tasks/` |
| `tsr` / `entropy` / `kin` / `full` → `keep_idx` | `scripts/make_keep_idx.py` |
| ACT regrid 训练与 Lead 扫描 | `adapters/`、`scripts/run_lead_sweep.sh` |
| E1 Lead 旧结果摘要 | `experiments/e1_lead/*.md` |

## 稳定接口

所有压缩方法最终只产出：

```text
keep_idx: int64[K]   # 升序；另存 t_keep、dt_keep
```

下游 ACT **只通过** `adapters/` 消费 `keep_idx`（统一控制步长）。

## 后续

| 阶段 | 内容 |
|------|------|
| E1 正式 | 队友服务器：full/tsr/entropy/kin × Lead 扫描（3000 epoch） |
| V2 | 接触容差的跨示范关系熵；多模态示范先分模式再估容差 |
| E3–E5 | 主对比、泛化、Diffusion Policy |
| P4 | TTR-Net 学习标签 |

## Baseline 列表（E3）

Full / Uniform / Entropy / Entropy+Lead / Kin / Kin+Lead / I-TTR / +ChunkLead
