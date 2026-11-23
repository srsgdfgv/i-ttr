# Vendored code provenance

Copied from local working tree `demospeedup` at
[srsgdfgv/demospeedup-aloha-extensions@c9256d2](https://github.com/srsgdfgv/demospeedup-aloha-extensions/commit/c9256d2),
itself based on [lingxiao-guo/DemoSpeedup](https://github.com/lingxiao-guo/DemoSpeedup) (MIT, see `act/LICENSE`).

| Path | Source | Notes |
|------|--------|-------|
| `act/` | DemoSpeedup ALOHA/ACT suite (`aloha/act`) | One-time copy. No bidirectional sync. |
| `kin_seg/` | Local demospeedup extension (`aloha/kin_seg`) | Baseline kinematic segmentation + precision lead. |
| `aloha_setup.py` | demospeedup `aloha/setup.py` | Reference only. |

## Local modifications

All scripts assume the **tsresample repo root** as cwd (`DATA_DIR="data"` in `act/constants.py`).

- `act/imitate_episodes.py`, `act/imitate_episodes_ablation.py`: default `--diffusion_policy_cfg` -> `third_party/act/image_aloha_diffusion_policy_cnn.yaml`.
- `act/imitate_episodes_ablation.py`: eval rollout count reads env `NUM_ROLLOUTS` (default 50) for smoke tests.
- `kin_seg/run_label_kin.py`, `kin_seg/export_for_speedup.py`: relative paths resolve against repo root; default `--out_dir` under `experiments/e1_lead/outputs/`.
- `kin_seg/run_speedup_wrapper.py`: chdir to repo root instead of `third_party/`.

Paper: DemoSpeedup (CoRL 2025), https://demospeedup.github.io/
