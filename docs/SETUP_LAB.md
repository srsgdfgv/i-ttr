# 实验室复现说明

面向：clone 后在 GPU 节点跑通环境、打标签、训练 ACT、rollout。

## 1. 代码与环境

```bash
git clone <REPO_URL> tsresample
cd tsresample
conda env create -f environment.yml
conda activate tsresample

export MUJOCO_GL=egl
export PYTHONPATH="$PWD:$PWD/third_party:$PWD/third_party/act:$PYTHONPATH"
```

- 仿真依赖 pip 的 `mujoco==2.3.3` + `dm_control`，不需要 `~/.mujoco` 或 `mujoco_py`。
- **所有命令在仓库根目录运行**（`DATA_DIR="data"` 为相对路径）。
- 服务器 CUDA 不是 11.8 时，改 `environment.yml` 里 torch 的 index URL / 版本后缀。

## 2. 数据

仓库不含 HDF5。挂载共享盘或本机 DemoSpeedup 数据：

```bash
bash scripts/link_data.sh /path/to/aloha/data
ls -la data/
```

`sim_insertion_human` 必须有 `episode_*.hdf5`，且含 `/labels`（共享保留率 ρ 由熵标签的 DemoSpeedup 2/4 步长统计）。

## 3. 预检

```bash
python scripts/preflight.py --task sim_insertion_human --need_labels
```

检查：CUDA、EGL 渲染、示范字段、`smoke_tsr`、磁盘与 `data/outputs` 可写。

可选一致性检查（无 GPU 也可）：

```bash
python scripts/check_dataset_equiv.py   # keep=全帧时与官方 Dataset 对齐
python scripts/smoke_tsr.py
```

## 4. 一键 Lead 扫描

```bash
bash scripts/run_lead_sweep.sh
```

常用环境变量：

| 变量 | 含义 | 示例 |
|------|------|------|
| `TASK` | 任务名 | `sim_insertion_human` |
| `METHODS` | 方法列表 | `full tsr entropy kin` |
| `LEADS` | Lead 秒数（`full` 忽略） | `0 0.2 0.4 0.8` |
| `EPOCHS` | 训练轮数 | `3000` |
| `NUM_ROLLOUTS` | 评估次数 | `50` |
| `NUM_EPISODES` | 只用前 N 条（冒烟） | `10` |
| `WORKERS` | 跟踪并行数 | `8` |
| `RUN_TAG` | 输出子目录名 | `anchor` / `lead_v1` |
| `SEED` | 随机种子 | `0` |
| `PY` | Python 可执行文件 | `python` |

流程：预检 → 物体状态缓存（可跳过已有）→ 求共享 ρ → 各组 `keep_idx` → 训练 → 高增益夹爪评估 → 写 `experiments/e1_lead/lead_<RUN_TAG>.md`。

产物目录：

```text
data/outputs/tsr/<task>/track/           # 物体/FK 缓存
data/outputs/tsr/<task>/keep/<tag>/      # keep_idx
data/outputs/tsr/<task>/<RUN_TAG>/<group>_s<seed>/   # ckpt、result_*.txt、视频
data/outputs/tsr/<task>/<RUN_TAG>/logs/<group>/      # keep/train/eval 日志
```

## 5. 建议实验顺序

**A. 流程冒烟**（可在笔记本上；成功率不作为判据）

```bash
METHODS="full tsr" LEADS="0 0.4" NUM_EPISODES=10 EPOCHS=200 NUM_ROLLOUTS=5 \
  RUN_TAG=smoke bash scripts/run_lead_sweep.sh
```

通过：三组跑完无报错；训练 loss 下降。200 epoch 出现抓不准、抖动是预期现象（欠训），与正式 3000 epoch 的 val loss 差约一个数量级。

正式设定统一为：**单 seed（`SEED=0`）· 3000 epoch · 50 rollout · 50 条示范**。

脚本里 `entropy` = DemoSpeedup 熵标签（对外可叫 **demo**）。

| 对外名称 | 脚本 | 组名示例 |
|----------|------|----------|
| full | `full` | `full_s0` |
| demo | `entropy` + lead `0` | `entropy_lead0_s0` |
| demo+lead | `entropy` + lead `>0` | `entropy_lead0.4_s0` |
| kin | `kin` + lead `0` | `kin_lead0_s0` |
| kin+lead | `kin` + lead `>0` | `kin_lead0.8_s0` |
| tsr | `tsr` + lead `0` | `tsr_lead0_s0` |
| tsr+lead | `tsr` + lead `>0` | `tsr_lead0.4_s0` |

**B. 先跑 tsr（无 lead）**（约 2–3 小时/单卡）

与已有 demo / kin 数字对比，再决定是否改方法；**此步不要开 lead**。

```bash
METHODS="tsr" LEADS="0" SEED=0 EPOCHS=3000 NUM_ROLLOUTS=50 RUN_TAG=tsr0 \
  WORKERS=8 bash scripts/run_lead_sweep.sh
```

**C. 再检验 lead**（方法确认后再跑）

```bash
METHODS="entropy kin" LEADS="0.2 0.4 0.8" \
  SEED=0 EPOCHS=3000 NUM_ROLLOUTS=50 RUN_TAG=lead_baselines WORKERS=8 \
  bash scripts/run_lead_sweep.sh
METHODS="tsr" LEADS="0.2 0.4 0.8" \
  SEED=0 EPOCHS=3000 NUM_ROLLOUTS=50 RUN_TAG=lead_tsr WORKERS=8 \
  bash scripts/run_lead_sweep.sh
```

同一次扫描内各方法共用同一 ρ；每个 lead 单独一个 policy。

## 6. 单独步骤（调试用）

```bash
# 跟踪
python sim_tools/track_episodes.py --task_cfg configs/tasks/sim_insertion_human.yaml --workers 8

# 打标签
python scripts/make_keep_idx.py --task_cfg configs/tasks/sim_insertion_human.yaml \
  --method tsr --lead_sec 0.4 --rho_from_entropy --tag tsr_lead0.4

# 训练 / 评估
python adapters/train_act.py --keep_dir data/outputs/tsr/sim_insertion_human/keep/tsr_lead0.4 \
  --task_name sim_insertion_human --ckpt_dir data/outputs/tsr/sim_insertion_human/ckpt/tsr_lead0.4_s0 \
  --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 8 \
  --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 --seed 0 --temporal_agg

NUM_ROLLOUTS=50 python adapters/train_act.py --eval --use_high_gripper_gain \
  --keep_dir data/outputs/tsr/sim_insertion_human/keep/tsr_lead0.4 \
  --task_name sim_insertion_human --ckpt_dir data/outputs/tsr/sim_insertion_human/ckpt/tsr_lead0.4_s0 \
  --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 8 \
  --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 --seed 0 --temporal_agg
```

不要与 `--speedup` 同时使用（那是 DemoSpeedup 自带的标签压缩）。

## 7. 不要做的事

- 不要提交 `*.hdf5` / `*.ckpt` / `*.mp4` / `data/outputs/`。
- 不要往共享盘上的原始 HDF5 写 `/labels_kin`（本仓库的 kin 路径只在内存算标签）。
- 不要改 `third_party/act` 训练核心；接入写在 `adapters/`。
- 不要对已有非空 ckpt 目录重复同一 `RUN_TAG`（脚本会拒绝，避免交互式 resume）。出错先看对应组的 `train.log` / `eval.log` / `keep.log`。
