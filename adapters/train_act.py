#!/usr/bin/env python3
"""
Train / eval ACT on keep_idx-compressed demos.

  python adapters/train_act.py --keep_dir data/outputs/tsr/.../keep/tsr_lead0 \\
      --task_name sim_insertion_human --ckpt_dir ... --policy_class ACT ...

Env overrides:
  NUM_EPISODES   limit demos (smoke); must match keep_dir coverage
  NUM_ROLLOUTS   eval rollouts (default 50)

Do not pass --speedup (DemoSpeedup's own compression). Use --use_high_gripper_gain
to match DemoSpeedup gripper kp at eval time.
"""

from __future__ import annotations

import functools
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "third_party", REPO / "third_party" / "act"):
    sys.path.insert(0, str(p))

import act.act_utils as act_utils  # noqa: E402
import act.constants as act_constants  # noqa: E402

from adapters.act_dataset import KeepIdxEpisodicDataset  # noqa: E402

sys.modules["constants"] = act_constants

keep_dir = None
if "--keep_dir" in sys.argv:
    i = sys.argv.index("--keep_dir")
    keep_dir = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    if "--speedup" in sys.argv:
        sys.exit("--speedup must not be combined with --keep_dir")
    act_utils.EpisodicDataset = functools.partial(KeepIdxEpisodicDataset, keep_dir=keep_dir)
elif "--eval" not in sys.argv:
    sys.exit("--keep_dir is required for training")

# Optional smoke: only use the first NUM_EPISODES demos.
if "NUM_EPISODES" in os.environ:
    n = int(os.environ["NUM_EPISODES"])
    for cfg in act_constants.SIM_TASK_CONFIGS.values():
        cfg["num_episodes"] = min(int(cfg["num_episodes"]), n)
    if keep_dir is not None:
        have = sorted(Path(keep_dir).glob("episode_*.npz"))
        if len(have) < n:
            sys.exit(f"--keep_dir has {len(have)} episodes, NUM_EPISODES={n}")

# Avoid the interactive "Load checkpoint?" prompt in non-interactive sweeps.
if os.environ.get("TSR_NO_RESUME", "1") == "1":
    import builtins

    _input = builtins.input

    def _no_resume(prompt: str = "") -> str:
        if "Load checkpoint" in prompt:
            print("TSR_NO_RESUME=1 -> not loading existing checkpoint", flush=True)
            return "n"
        return _input(prompt)

    builtins.input = _no_resume

os.chdir(REPO)
script = REPO / "third_party" / "act" / "imitate_episodes_ablation.py"
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
