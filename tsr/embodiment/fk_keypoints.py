"""FK: recorded robot qpos -> effector keypoints (finger mesh vertices) and gripper frame.

The robot is set kinematically (no dynamics), so keypoints are exact for the recorded motion.
"""

from __future__ import annotations

import numpy as np
from dm_control import mujoco

from act.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN

SIDES = ("left", "right")


class AlohaScene:
    """MuJoCo model of the ALOHA sim task, driven kinematically from 14-D qpos."""

    def __init__(self, xml_path: str, n_finger_pts: int = 64):
        self.physics = mujoco.Physics.from_xml_path(xml_path)
        m = self.physics.model
        self._finger_geoms = {
            s: [m.name2id(f"vx300s_{s}/10_{f}_gripper_finger", "geom") for f in ("left", "right")]
            for s in SIDES
        }
        self._finger_verts = {}
        for s, gids in self._finger_geoms.items():
            verts = []
            for g in gids:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                step = max(1, len(v) // n_finger_pts)
                verts.append(v[::step].copy())
            self._finger_verts[s] = verts
        self._gripper_body = {s: m.name2id(f"vx300s_{s}/gripper_link", "body") for s in SIDES}

    def set_robot(self, qpos14: np.ndarray) -> None:
        q = np.asarray(qpos14, dtype=np.float64)
        d = self.physics.data
        for k, s in enumerate(SIDES):
            arm, grip = q[7 * k : 7 * k + 6], q[7 * k + 6]
            finger = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(grip)
            d.qpos[8 * k : 8 * k + 6] = arm
            d.qpos[8 * k + 6 : 8 * k + 8] = [finger, -finger]
        self.physics.forward()

    def finger_points(self, side: str, split: bool = False):
        """World-frame finger surface points, (K, 3); split=True returns one array per finger."""
        d = self.physics.data
        pts = [
            v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            for g, v in zip(self._finger_geoms[side], self._finger_verts[side])
        ]
        return pts if split else np.concatenate(pts, axis=0)

    def gripper_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        b = self._gripper_body[side]
        d = self.physics.data
        return d.xpos[b].copy(), d.xquat[b].copy()


def effector_trajectories(scene: AlohaScene, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Per side: finger points (T, K, 3), gripper pos (T, 3), quat wxyz (T, 4), aperture (T,)."""
    out: dict[str, list] = {}
    for t, q in enumerate(qpos):
        scene.set_robot(q)
        for k, s in enumerate(SIDES):
            pos, quat = scene.gripper_pose(s)
            out.setdefault(f"eff_pts_{s}", []).append(scene.finger_points(s))
            out.setdefault(f"eff_pos_{s}", []).append(pos)
            out.setdefault(f"eff_quat_{s}", []).append(quat)
            out.setdefault(f"aperture_{s}", []).append(q[7 * k + 6])
    return {k: np.asarray(v) for k, v in out.items()}
