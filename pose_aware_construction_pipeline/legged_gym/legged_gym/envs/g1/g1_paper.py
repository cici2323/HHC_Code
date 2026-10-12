"""G1 environment connecting the final paper rewards to all three stages.

The construction adapter consumes recorded support/workspace configurations.
Workspace geometry is an explicit sphere/box approximation, configured by the
user; this is not a claim to recover the paper's unspecified mesh intersection.
"""

import json
import hashlib
from pathlib import Path

import torch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from .g1_mimic_priv_distill import G1MimicPrivDistill
from .g1_mimic_priv import local_to_global
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from rsl_rl.residual.task_rewards import PaperRewardMixin
from rsl_rl.residual.construction import ConstructionDataset, robot_sphere_box_overlap, robot_frame_context


class G1Paper(PaperRewardMixin, G1MimicPrivDistill):
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.construction_dataset = None
        self.residual_context_schema = "none"
        if self.cfg.construction.enabled:
            data_path = Path(self.cfg.construction.data_path)
            geometry_path = Path(self.cfg.construction.geometry_path)
            self.construction_dataset = ConstructionDataset(data_path, device=self.device)
            with geometry_path.open() as stream:
                geometry = json.load(stream)
            if not isinstance(geometry, dict) or geometry.get("schema_version") != 1:
                raise ValueError("Robot geometry requires schema_version=1")
            support = geometry.get("support_points")
            spheres = geometry.get("robot_spheres")
            if not isinstance(support, list) or len(support) != 2:
                raise ValueError("Robot geometry must specify exactly two support_points")
            if not isinstance(spheres, list) or not spheres:
                raise ValueError("Robot geometry must specify occupied robot_spheres")
            self._support_body_ids, self._support_offsets = self._geometry_points(support)
            self._sphere_body_ids, self._sphere_offsets = self._geometry_points(spheres)
            radii = [entry["radius"] for entry in spheres]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in radii):
                raise ValueError("Robot sphere radii must be numeric metres")
            self._sphere_radii = torch.tensor(radii, dtype=torch.float32, device=self.device)
            if not torch.isfinite(self._sphere_radii).all() or not (self._sphere_radii > 0).all():
                raise ValueError("Robot sphere radii must be finite positive metres")
            self._construction_ids = self.construction_dataset.sample(self.num_envs)
            self.residual_context_schema = {
                "format": "robot_relative_support6_workspace_corners24B_v2",
                "frame": "robot_base", "units": "metres",
                "context_dim": 6 + 24 * self.construction_dataset.num_boxes,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
            }

    def _geometry_points(self, entries):
        body_ids, offsets = [], []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("body"), str):
                raise ValueError("Geometry points require a body name and local offset")
            body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], entry["body"])
            if body_id < 0:
                raise ValueError("Unknown robot body: {}".format(entry["body"]))
            body_ids.append(body_id)
            offsets.append(entry["offset"])
        offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)
        if offsets.shape != (len(entries), 3) or not torch.isfinite(offsets).all():
            raise ValueError("Geometry offsets must be finite local [x,y,z] metres")
        return torch.tensor(body_ids, dtype=torch.long, device=self.device), offsets

    def _points_in_world(self, body_ids, offsets):
        bodies = self.rigid_body_states[:, body_ids]
        local = offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        rotated = quat_rotate(bodies[:, :, 3:7].reshape(-1, 4), local.reshape(-1, 3))
        return bodies[:, :, :3] + rotated.reshape(self.num_envs, -1, 3)

    def reset_idx(self, env_ids, init=False):
        # Reward is already computed when the base class calls reset_idx.
        super().reset_idx(env_ids, init=init)
        if self.construction_dataset is not None and len(env_ids):
            self._construction_ids[env_ids] = self.construction_dataset.sample(len(env_ids))
        if len(env_ids):
            # Parent reset changes motion IDs/time; refresh reference and local
            # robot state before the base class builds the reset observation.
            self.update_demo_obs()
            r, p, y = euler_from_quaternion(self.root_states[:, 3:7])
            for name, value in (("roll", r), ("pitch", p), ("yaw", y)):
                if hasattr(self, name):
                    getattr(self, name)[env_ids] = value[env_ids]
                else:
                    setattr(self, name, value.clone())
            _, _, reference_yaw = euler_from_quaternion(self._curr_demo_quat)
            if hasattr(self, "target_yaw"):
                self.target_yaw[env_ids] = reference_yaw[env_ids]
            else:
                self.target_yaw = reference_yaw.clone()
            quaternion = self.root_states[env_ids, 3:7]
            self.base_lin_vel[env_ids] = quat_rotate_inverse(quaternion, self.root_states[env_ids, 7:10])
            self.base_ang_vel[env_ids] = quat_rotate_inverse(quaternion, self.root_states[env_ids, 10:13])
            self.projected_gravity[env_ids] = quat_rotate_inverse(quaternion, self.gravity_vec[env_ids])

    def post_physics_step(self):
        # Advance the reference before reward/observations, not after them as in
        # the inherited mimic task. This keeps the actor's next observation and
        # the next transition's tracking target on the same reference clock.
        self._motion_times += self._motion_dt
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.0
        self.update_demo_obs()
        LeggedRobot.post_physics_step(self)
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_rigid_bodies_demo()

    def get_residual_context(self):
        if self.construction_dataset is None:
            return torch.empty(self.num_envs, 0, device=self.device)
        return robot_frame_context(
            self.construction_dataset.support_points(self._construction_ids, self.env_origins),
            self.construction_dataset.workspace_boxes(self._construction_ids, self.env_origins),
            self.root_states[:, :3], self.root_states[:, 3:7],
        )

    def motion_reward_inputs(self):
        count = self._n_demo_dof
        reference = self._curr_demo_obs_buf
        reference_velocity = reference[:, count:count + 3].clone()
        reference_velocity[self._in_place_flag] = 0
        reference_roll, reference_pitch, reference_yaw = euler_from_quaternion(self._curr_demo_quat)
        # Keep the upstream horizontal alignment while using a world-height
        # reference. Environment origins translate the common world frame.
        reference_xy = self.target_pos_abs if self.cfg.motion.global_keybody else self.root_states[:, :2]
        reference_height = self._curr_demo_root_pos[:, 2] + self.env_origins[:, 2]
        reference_root = torch.cat((reference_xy, reference_height[:, None]), dim=-1)
        reference_keypoints = local_to_global(
            self._curr_demo_quat, self._curr_demo_keybody.reshape(self.num_envs, self._num_key_bodies, 3),
            reference_root,
        )
        current_keypoints = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3]
        return {
            "dof_position": self.dof_pos[:, self._dof_ids_subset],
            "reference_dof_position": reference[:, :count],
            "keypoint_position": current_keypoints,
            "reference_keypoint_position": reference_keypoints,
            "root_velocity": self.base_lin_vel,
            "reference_root_velocity": reference_velocity,
            "root_angular_velocity": self.base_ang_vel,
            "reference_root_angular_velocity": reference[:, count + 3:count + 6],
            "heading": self.yaw, "reference_heading": reference_yaw,
            "height": self.root_states[:, 2], "reference_height": reference_height,
            "body_rotation": torch.stack((self.roll, self.pitch, self.yaw), dim=-1),
            "reference_body_rotation": torch.stack((reference_roll, reference_pitch, reference_yaw), dim=-1),
        }

    def stabilization_reward_inputs(self):
        return {
            "base_angular_velocity": self.base_ang_vel,
            "foot_velocity": self.rigid_body_states[:, self.feet_indices, 7:10],
            "contacts": self.contact_forces[:, self.feet_indices, 2] > 0,
        }

    def collaboration_reward_inputs(self):
        if self.construction_dataset is None:
            raise ValueError("Construction adaptation requires measured support/workspace data")
        boxes = self.construction_dataset.workspace_boxes(self._construction_ids, self.env_origins)
        centres = self._points_in_world(self._sphere_body_ids, self._sphere_offsets)
        return {
            **self.stabilization_reward_inputs(),
            "current_support_points": self._points_in_world(self._support_body_ids, self._support_offsets),
            "target_support_points": self.construction_dataset.support_points(self._construction_ids, self.env_origins),
            "workspace_overlap": robot_sphere_box_overlap(centres, self._sphere_radii, boxes),
        }
