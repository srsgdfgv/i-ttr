# E1 Lead 扫描

正式设定：单 seed=`0` · 3000 epoch · 50 rollout。

顺序：先 `tsr`（无 lead）对照已有 demo/kin → 再扫 lead（demo+lead / kin+lead / tsr+lead）。
脚本 `entropy` = DemoSpeedup = demo。

旧仓库结果摘要（对照）：

- `kin_seg_SUMMARY.md`
- `speedup_eval_lead.md`
- `lead_sweep.md`

本仓库一键扫描输出（`scripts/collect_results.py`）：

- `lead_<RUN_TAG>.md`（如本机 `lead_smoke.md`、队友 `lead_anchor.md` / `lead_lead_v1.md`）

只提交 md；ckpt / 视频 / 完整 log 留在 `data/outputs/tsr/`。
