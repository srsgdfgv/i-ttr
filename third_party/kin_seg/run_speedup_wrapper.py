#!/usr/bin/env python3
"""
Run official act/imitate_episodes.py with a runtime-registered task pointing at
data/sim_insertion_human_kin — without editing act/constants.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ALOHA = Path(__file__).resolve().parents[1]
_ACT = _ALOHA / "act"
_REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(_ALOHA))
sys.path.insert(0, str(_ACT))

# imitate_episodes does `from constants import SIM_TASK_CONFIGS` (act on PYTHONPATH)
import act.constants as act_constants

sys.modules["constants"] = act_constants

TASK = "sim_insertion_human_kin"
act_constants.SIM_TASK_CONFIGS[TASK] = {
    "dataset_dir": "data/sim_insertion_human_kin",
    "num_episodes": 50,
    "episode_len": 500,
    "camera_names": ["top"],
}

if len(sys.argv) == 1:
    print(
        "Example:\n"
        "  python third_party/kin_seg/run_speedup_wrapper.py --task_name sim_insertion_human_kin "
        "--ckpt_dir data/outputs/ACT_kin_speedup_ckpt/sim_insertion_human_e3000 "
        "--policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 "
        "--batch_size 8 --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 "
        "--seed 0 --temporal_agg --speedup"
    )
    sys.exit(0)

if "--task_name" not in sys.argv:
    sys.argv[1:1] = ["--task_name", TASK]

sys.argv[0] = str(_ACT / "imitate_episodes.py")
# Keep cwd at repo root so relative data/ paths work
os_chdir = __import__("os").chdir
os_chdir(_REPO)
runpy.run_path(str(_ACT / "imitate_episodes.py"), run_name="__main__")
