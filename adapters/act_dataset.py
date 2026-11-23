"""ACT training adapter: consume keep_idx sidecars without touching the (shared) demo HDF5.

Downstream scheme ("regrid"): kept frames are re-indexed onto a uniform control grid.
Step i of the compressed demo is original frame keep_idx[i]; the policy observes
obs[keep_idx[i]] and is supervised on actions[keep_idx[i:]] executed one control dt apart.
The original timing (t_keep, dt_keep) is stored alongside for diagnostics / a later
time-conditioned variant.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch


def save_keep_idx(path: str | Path, keep_idx: np.ndarray, dt: float, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    t_keep = keep_idx * dt
    dt_keep = np.append(np.diff(t_keep), dt)
    np.savez_compressed(path, keep_idx=keep_idx, t_keep=t_keep, dt_keep=dt_keep, **extra)


def load_keep_idx(path: str | Path) -> np.ndarray:
    with np.load(path) as z:
        return z["keep_idx"].astype(np.int64)


class KeepIdxEpisodicDataset(torch.utils.data.Dataset):
    """Drop-in for act.act_utils.EpisodicDataset (ACT, sim) using keep_idx sidecars."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, keep_dir, **_):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.keep = {i: load_keep_idx(Path(keep_dir) / f"episode_{i}.npz") for i in episode_ids}
        self.is_sim = True

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        keep = self.keep[episode_id]
        # Match official EpisodicDataset: np.random.choice over the index set.
        i = int(np.random.choice(len(keep)))
        path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(path, "r") as root:
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            ts = int(keep[i])
            qpos = root["/observations/qpos"][ts]
            images = np.stack(
                [root[f"/observations/images/{c}"][ts] for c in self.camera_names], axis=0
            )
            action = root["/action"][()][keep[i:]]

        padded = np.zeros(original_action_shape, dtype=np.float32)
        padded[: len(action)] = action
        is_pad = np.zeros(episode_len)
        is_pad[len(action) :] = 1

        image_data = torch.from_numpy(images)
        image_data = torch.einsum("k h w c -> k c h w", image_data) / 255.0
        action_data = (torch.from_numpy(padded).float() - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (torch.from_numpy(qpos).float() - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return image_data, qpos_data, action_data, torch.from_numpy(is_pad).bool()

