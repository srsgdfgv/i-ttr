"""Recover object poses for ALOHA sim demos that lack env_state.

Pipeline per frame:
  1. Hard-set recorded robot qpos; step MuJoCo so free-joint objects move.
  2. If an object is pinched by a gripper, lock it rigidly to that gripper
     (frozen relative pose at grasp onset) until release. This is the ALOHA-sim
     substitute for missing env_state during grasps.
  3. Otherwise keep the physics pose when render-vs-recorded IoU is good;
     fall back to visual refinement when it is not.
"""

from __future__ import annotations

import numpy as np

from tsr.embodiment.fk_keypoints import AlohaScene
from tsr.geometry import mat_to_quat, quat_to_mat
from tsr.perception.object_pipeline import ObjectTracker, color_mask, _iou

_MJ_OBJ_GEOM = 5
SOURCE_PHYSICS = 0
SOURCE_VISION = 1
SOURCE_WELD = 2


class ObjectStateRecoverer:
    def __init__(self, scene: AlohaScene, objects: dict, camera: str, track_cfg: dict, hw=(480, 640)):
        self.scene = scene
        self.physics = scene.physics
        self.tracker = ObjectTracker(scene, objects, camera, track_cfg, hw=hw)
        self.objects = self.tracker.objects
        self.cam = self.tracker.cam
        self.hw = hw
        self.cfg = track_cfg
        self.n_sub = int(track_cfg.get("physics_substeps", 5))
        self.iou_min = float(track_cfg.get("physics_iou_min", 0.7))
        self.attach = float(track_cfg["attach_dist"])
        self.release = float(track_cfg.get("release_dist", 0.02))

    def _write_robot(self, qpos14: np.ndarray) -> None:
        self.scene.set_robot(qpos14)
        d = self.physics.data
        d.ctrl[:] = d.qpos[:16]
        d.qvel[:16] = 0.0

    def _get_pose(self, name: str) -> np.ndarray:
        a = self.objects[name]["qadr"]
        return self.physics.data.qpos[a : a + 7].copy()

    def _object_iou(self, name: str, obs_mask: np.ndarray) -> float:
        self.physics.forward()
        seg = self.physics.render(*self.hw, camera_id=self.cam, segmentation=True)
        pred = np.isin(np.where(seg[..., 1] == _MJ_OBJ_GEOM, seg[..., 0], -1), self.objects[name]["geoms"])
        return _iou(pred, obs_mask)

    def _weld_pose(self, weld: tuple) -> np.ndarray:
        side, p_rel, R_rel = weld
        p_e, q_e = self.scene.gripper_pose(side)
        R_e = quat_to_mat(q_e)
        return np.concatenate([p_e + R_e @ p_rel, mat_to_quat(R_e @ R_rel)])

    def _make_weld(self, side: str, pose: np.ndarray) -> tuple:
        p_e, q_e = self.scene.gripper_pose(side)
        R_e = quat_to_mat(q_e)
        return side, R_e.T @ (pose[:3] - p_e), R_e.T @ quat_to_mat(pose[3:])

    def recover(self, qpos: np.ndarray, images) -> dict[str, np.ndarray]:
        T = len(qpos)
        poses = {n: np.zeros((T, 7)) for n in self.objects}
        ious = {n: np.zeros(T) for n in self.objects}
        source = {n: np.zeros(T, dtype=np.int8) for n in self.objects}
        npx = {n: np.zeros(T, dtype=np.int64) for n in self.objects}
        weld = {n: None for n in self.objects}

        self._write_robot(qpos[0])
        for n, o in self.objects.items():
            obs = color_mask(images[0], o["color"])
            npx[n][0] = int(obs.sum())
            pose, iou = self.tracker.init_pose(n, obs)
            poses[n][0], ious[n][0] = pose, iou
            source[n][0] = SOURCE_VISION
            self.tracker.set_pose(n, pose)

        for t in range(1, T):
            for s in range(1, self.n_sub + 1):
                alpha = s / self.n_sub
                q = (1.0 - alpha) * qpos[t - 1] + alpha * qpos[t]
                self._write_robot(q)
                # Keep welded objects glued during substeps so contacts stay consistent.
                for n, w in weld.items():
                    if w is not None:
                        self.tracker.set_pose(n, self._weld_pose(w))
                self.physics.step()
            self._write_robot(qpos[t])
            self.physics.forward()

            img = images[t]
            for n, o in self.objects.items():
                obs = color_mask(img, o["color"])
                npx[n][t] = int(obs.sum())
                visible = npx[n][t] >= self.cfg["min_pixels"]

                # --- already welded: stay rigid until release ---
                if weld[n] is not None:
                    pose = self._weld_pose(weld[n])
                    self.tracker.set_pose(n, pose)
                    iou = self._object_iou(n, obs) if visible else ious[n][t - 1]
                    side = self.tracker._carrier(n, pose, dist_thresh=self.release)
                    if side is None:
                        weld[n] = None
                    else:
                        poses[n][t], ious[n][t] = pose, iou
                        source[n][t] = SOURCE_WELD
                        continue

                phys = self._get_pose(n)
                self.tracker.set_pose(n, phys)
                iou = self._object_iou(n, obs) if visible else 1.0
                pose, src = phys, SOURCE_PHYSICS

                if visible and iou < self.iou_min:
                    cur = poses[n][t - 1].copy()
                    self.tracker.set_pose(n, cur)
                    s0 = self.tracker._score(n, cur, obs)
                    if s0 < self.cfg["iou_refine"]:
                        cur, s0 = self.tracker.refine(n, cur, obs)
                    pose, iou, src = cur, s0, SOURCE_VISION
                    self.tracker.set_pose(n, pose)

                # Start a weld if pinched.
                side = self.tracker._carrier(n, pose, dist_thresh=self.attach)
                if side is not None:
                    weld[n] = self._make_weld(side, pose)
                    pose = self._weld_pose(weld[n])
                    self.tracker.set_pose(n, pose)
                    src = SOURCE_WELD
                    if visible:
                        iou = self._object_iou(n, obs)

                poses[n][t], ious[n][t], source[n][t] = pose, iou, src

        out = {}
        for n in self.objects:
            out[f"obj_pose_{n}"] = poses[n]
            out[f"iou_{n}"] = ious[n]
            out[f"obs_px_{n}"] = npx[n]
            out[f"source_{n}"] = source[n]
        return out
