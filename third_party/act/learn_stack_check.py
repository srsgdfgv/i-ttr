"""One-task learning checks: hdf5 keys + make_sim_env smoke (no training)."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np


def check_hdf5(path="data/sim_transfer_cube_scripted/episode_0.hdf5"):
    print("=== HDF5 ===", path)
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
        f.visititems(walk)


def check_env():
    print("=== make_sim_env(sim_transfer_cube_scripted) ===")
    # Headless EGL often needed in WSL
    os.environ.setdefault("MUJOCO_GL", "egl")
    from act.sim_env import make_sim_env, BOX_POSE
    from act.act_utils import sample_box_pose

    env = make_sim_env("sim_transfer_cube_scripted")
    BOX_POSE[0] = sample_box_pose()
    ts = env.reset()
    obs = ts.observation
    print("  qpos shape:", np.asarray(obs["qpos"]).shape)
    print("  image keys:", list(obs["images"].keys()))
    for k, v in obs["images"].items():
        print(f"  images[{k}] shape:", np.asarray(v).shape)
    action = np.zeros(14, dtype=np.float32)
    ts2 = env.step(action)
    print("  step ok, reward:", ts2.reward)
    print("ENV_OK")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    check_hdf5()
    check_env()
