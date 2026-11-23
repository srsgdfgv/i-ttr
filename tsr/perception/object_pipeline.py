"""Model-based visual object tracking for ALOHA sim demos.

Human sim demos store no object state, and open-loop physics replay does not reproduce
the grasps. Instead, with the robot set kinematically from recorded qpos, each object's
6-DoF pose is fitted so that its rendered silhouette (robot occlusion included) matches
the color mask of the recorded top image:

  init      : mask centroid back-projected to the table plane, then refine (x, y, yaw)
  predict   : static, or rigidly carried by a gripper whose two fingers both touch it
  correct   : coordinate descent on IoU when it falls below `iou_refine`
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from tsr.embodiment.fk_keypoints import SIDES, AlohaScene
from tsr.geometry import boxes_to_world, mat_to_quat, points_to_boxes_distance, quat_to_mat

_MJ_OBJ_GEOM = 5
_POS_STEPS = (0.004, 0.002, 0.001)
_ROT_STEPS = np.deg2rad((4.0, 2.0, 1.0))


def color_mask(img: np.ndarray, color: str) -> np.ndarray:
    r, g, b = (img[..., i].astype(np.int32) for i in range(3))
    if color == "red":
        return (r > 100) & (r > 2 * g) & (r > 2 * b)
    if color == "blue":
        return (b > 100) & (b > 2 * r) & (b > 2 * g)
    raise ValueError(f"unknown color {color}")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


class ObjectTracker:
    def __init__(self, scene: AlohaScene, objects: dict, camera: str, track_cfg: dict, hw=(480, 640)):
        self.scene = scene
        self.physics = scene.physics
        m = self.physics.model
        self.cam = m.name2id(camera, "camera")
        self.hw = hw
        self.cfg = track_cfg
        self.objects = {}
        for name, spec in objects.items():
            body = m.name2id(spec["body"], "body")
            gids = np.array([m.name2id(g, "geom") for g in spec["geoms"]])
            boxes = np.concatenate([m.geom_pos[gids], m.geom_size[gids]], axis=1)
            self.objects[name] = {
                "qadr": m.jnt_qposadr[m.body_jntadr[body]],
                "geoms": gids,
                "boxes": boxes,
                "color": spec["color"],
                "z0": float(m.body_pos[body][2]),
            }

    # ----- scene access -----
    def set_pose(self, name: str, pose: np.ndarray) -> None:
        a = self.objects[name]["qadr"]
        self.physics.data.qpos[a : a + 7] = pose

    def _seg(self) -> np.ndarray:
        self.physics.forward()
        seg = self.physics.render(*self.hw, camera_id=self.cam, segmentation=True)
        return np.where(seg[..., 1] == _MJ_OBJ_GEOM, seg[..., 0], -1)

    def _score(self, name: str, pose: np.ndarray, obs: np.ndarray) -> float:
        self.set_pose(name, pose)
        return _iou(np.isin(self._seg(), self.objects[name]["geoms"]), obs)

    # ----- pose search -----
    def refine(self, name: str, pose: np.ndarray, obs: np.ndarray, dims=range(6)) -> tuple[np.ndarray, float]:
        best, best_s = pose.copy(), self._score(name, pose, obs)
        for ps, rs in zip(_POS_STEPS, _ROT_STEPS):
            for _ in range(4):
                improved = False
                for d in dims:
                    for sign in (1.0, -1.0):
                        cand = best.copy()
                        if d < 3:
                            cand[d] += sign * ps
                        else:
                            rot = Rotation.from_rotvec(sign * rs * np.eye(3)[d - 3]).as_matrix()
                            cand[3:] = mat_to_quat(rot @ quat_to_mat(best[3:]))
                        s = self._score(name, cand, obs)
                        if s > best_s + 1e-4:
                            best, best_s, improved = cand, s, True
                if not improved:
                    break
        self.set_pose(name, best)
        return best, best_s

    def _backproject(self, uv: np.ndarray, z: float) -> np.ndarray:
        d = self.physics.data
        pos, R = d.cam_xpos[self.cam], d.cam_xmat[self.cam].reshape(3, 3)
        h, w = self.hw
        f = 0.5 * h / np.tan(np.deg2rad(self.physics.model.cam_fovy[self.cam]) / 2)
        ray = R @ np.array([(uv[0] - w / 2) / f, -(uv[1] - h / 2) / f, -1.0])
        return pos + (z - pos[2]) / ray[2] * ray

    def init_pose(self, name: str, obs: np.ndarray) -> tuple[np.ndarray, float]:
        ys, xs = np.nonzero(obs)
        z0 = self.objects[name]["z0"]
        xy = self._backproject(np.array([xs.mean(), ys.mean()]), z0)[:2]
        pose = np.array([xy[0], xy[1], z0, 1.0, 0.0, 0.0, 0.0])
        return self.refine(name, pose, obs, dims=(0, 1, 5))

    # ----- carrying -----
    def _carrier(self, name: str, pose: np.ndarray, dist_thresh: float | None = None) -> str | None:
        """Side whose two fingers both touch the object (object pinched), else None."""
        o = self.objects[name]
        centers, R, half = boxes_to_world(o["boxes"], pose[:3], pose[3:])
        best, best_d = None, float(dist_thresh if dist_thresh is not None else self.cfg["attach_dist"])
        for s in SIDES:
            dist = max(
                points_to_boxes_distance(f[None], centers, R, half)[0]
                for f in self.scene.finger_points(s, split=True)
            )
            if dist < best_d:
                best, best_d = s, dist
        return best

    def track(self, qpos: np.ndarray, images) -> dict[str, np.ndarray]:
        """images: indexable (T, H, W, 3), e.g. an h5py dataset."""
        T = len(qpos)
        poses = {n: np.zeros((T, 7)) for n in self.objects}
        ious = {n: np.zeros(T) for n in self.objects}
        npx = {n: np.zeros(T, dtype=np.int64) for n in self.objects}
        cur, carry = {}, {n: None for n in self.objects}

        for t in range(T):
            img = images[t]
            self.scene.set_robot(qpos[t])
            for n, o in self.objects.items():
                if t == 0:
                    continue
                if carry[n] is not None:
                    side, p_rel, R_rel = carry[n]
                    p_e, q_e = self.scene.gripper_pose(side)
                    R_e = quat_to_mat(q_e)
                    cur[n] = np.concatenate([p_e + R_e @ p_rel, mat_to_quat(R_e @ R_rel)])
                self.set_pose(n, cur[n])

            for n, o in self.objects.items():
                obs = color_mask(img, o["color"])
                npx[n][t] = obs.sum()
                if t == 0:
                    cur[n], ious[n][t] = self.init_pose(n, obs)
                    continue
                s = self._score(n, cur[n], obs)
                if npx[n][t] >= self.cfg["min_pixels"] and s < self.cfg["iou_refine"]:
                    new, s_new = self.refine(n, cur[n], obs)
                    # single top view: depth is ambiguous, so trust rigid carry unless clearly wrong
                    if carry[n] is None or s_new - s >= self.cfg["carry_min_gain"]:
                        cur[n], s = new, s_new
                    else:
                        self.set_pose(n, cur[n])
                ious[n][t] = s

            for n in self.objects:
                poses[n][t] = cur[n]
                side = self._carrier(n, cur[n])
                if side is None:
                    carry[n] = None
                    continue
                p_e, q_e = self.scene.gripper_pose(side)
                R_e = quat_to_mat(q_e)
                carry[n] = (side, R_e.T @ (cur[n][:3] - p_e), R_e.T @ quat_to_mat(cur[n][3:]))

        out = {}
        for n in self.objects:
            out[f"obj_pose_{n}"] = poses[n]
            out[f"iou_{n}"] = ious[n]
            out[f"obs_px_{n}"] = npx[n]
        return out
