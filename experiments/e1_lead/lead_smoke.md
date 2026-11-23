# Lead sweep `smoke`（流程冒烟，非正式指标）

Source: `data/outputs/tsr/sim_insertion_human/smoke`

设定：`NUM_EPISODES=10`、`EPOCHS=200`、`NUM_ROLLOUTS=5`。  
目的：验证跟踪 → 打标签 → 训练 → eval 能跑通。**成功率不具解释意义**（欠训；同评估入口换正式 ckpt 可达非零成功率）。

| group | success | return | avg_len | keep_ratio | ≥3 | ≥2 |
|-------|---------|--------|---------|------------|----|----|
| full_s0 | 0.00 | 0.0 | nan | 1.000 | 0.00 | 0.00 |
| tsr_lead0_s0 | 0.00 | 0.0 | nan | 0.416 | 0.00 | 0.00 |
| tsr_lead0.4_s0 | 0.00 | 0.0 | nan | 0.416 | 0.00 | 0.00 |
