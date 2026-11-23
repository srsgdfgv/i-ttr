# tsresample

交互驱动的自适应时序重采样（ALOHA 仿真 ACT）。

运送段可压缩，接近/接触段需加密。本仓库用**交互图**上的几何信号构造时序分辨率，在固定保留预算下非均匀重采样，再训练 ACT，并与 DemoSpeedup 熵标签 / Kin 分段 + Lead 对照。

与 [DemoSpeedup](https://demospeedup.github.io/)（CoRL 2025）的关系：

- **不是** fork；ACT 训练栈一次性放在 `third_party/act/`，示范数据用软链接。
- 官方熵标签与 Kin-seg 仅作 baseline；本方法标签为 free（tau）/ contact（物理容差）/ event，**不使用力/力矩**。

## 仓库结构

```text
tsr/                      # 方法核心
  embodiment/             # 录制 qpos → 手指关键点（FK）
  perception/             # 物体位姿恢复（物理 + 夹持锁定 / 视觉回退）
  interaction/            # 交互对间隙与相对运动
  label/                  # free / contact / event / Lead / 二值标签
  resample.py             # 频率 → keep_idx
  baselines.py            # Full / Uniform / DemoSpeedup 2-4
adapters/                 # keep_idx → ACT（统一控制步长 regrid）
sim_tools/track_episodes.py
configs/tasks/            # 交互图与容差
scripts/                  # 预检、打标签、Lead 扫描、冒烟
third_party/act/          # vendored ACT + ALOHA sim
third_party/kin_seg/      # Kin baseline（内存打标签，不写 HDF5）
data/                     # 软链接到 HDF5（不入 git）
experiments/              # 结果摘要 md
docs/SETUP_LAB.md         # 环境与正式实验复现
```

## 快速复现

详细步骤见 [docs/SETUP_LAB.md](docs/SETUP_LAB.md)。

```bash
git clone <REPO_URL> tsresample && cd tsresample
conda env create -f environment.yml && conda activate tsresample
export PYTHONPATH="$PWD:$PWD/third_party:$PWD/third_party/act:$PYTHONPATH"
export MUJOCO_GL=egl

bash scripts/link_data.sh /path/to/aloha/data   # 需含 sim_insertion_human 与 /labels
python scripts/preflight.py --need_labels

# 流程冒烟（10 条 × 200 epoch；成功率可忽略，只看是否跑通）
METHODS="full tsr" LEADS="0 0.4" NUM_EPISODES=10 EPOCHS=200 NUM_ROLLOUTS=5 \
  RUN_TAG=smoke bash scripts/run_lead_sweep.sh

# 正式实验：单 seed=0 · 3000 epoch · 50 rollout
# 产出 full / demo(=entropy)±lead / kin±lead / tsr±lead
METHODS="full tsr" LEADS="0" SEED=0 EPOCHS=3000 NUM_ROLLOUTS=50 RUN_TAG=anchor \
  WORKERS=8 bash scripts/run_lead_sweep.sh
METHODS="full tsr entropy kin" LEADS="0 0.2 0.4 0.8" \
  SEED=0 EPOCHS=3000 NUM_ROLLOUTS=50 RUN_TAG=lead_v1 WORKERS=8 \
  bash scripts/run_lead_sweep.sh
```

输出在 `data/outputs/tsr/`（不入 git）；汇总表写入 `experiments/e1_lead/lead_<RUN_TAG>.md`。

## 方法与接口（V1）

1. **物体状态**：每帧硬设录制机器人 qpos，物理更新物体；夹持后刚体锁定到夹爪；渲染 IoU 不达标则回退视觉。
2. **标签**：任务配置中的交互对取 `max(free, contact, event)`；可选按秒的 Lead / Chunk-Lead（默认关）。
3. **压缩**：`keep_idx` + `t_keep` + `dt_keep`；同一次扫描内各方法共用熵标签 2/4 步长得到的保留率 ρ。
4. **下游**：保留帧映射到统一控制步长再训 ACT（避免把时间语义变化误当成重采样效果）。

| 模块 | 状态 |
|------|------|
| 物体真值恢复 + 交互图标签 + Lead | V1 |
| `tsr` / `entropy` / `kin` / `full` → keep_idx | V1 |
| ACT regrid + `run_lead_sweep.sh` | V1 |
| 关系熵跨示范容差 / TTR-Net | 未做（V2 / P4） |

## 引用

若使用 ACT / ALOHA 仿真或对比 DemoSpeedup，请同时引用原论文与官方仓库。第三方说明见 [NOTICE.md](NOTICE.md)、[third_party/VENDOR.md](third_party/VENDOR.md)。
