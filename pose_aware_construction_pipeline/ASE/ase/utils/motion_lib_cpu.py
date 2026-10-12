# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import yaml

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *

from utils import torch_utils

import torch
import dill as pickle

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device, no_keybody=False, regen_pkl=False, load_device="cpu"):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=load_device)
        self._device = device
        self._load_device = load_device

        print("*"*20 + " Loading motion library " + "*"*20)
        motion_dir = os.path.dirname(motion_file)
        yaml_name = motion_file.split("/")[-1].split(".")[0]
        pkl_file = os.path.abspath(os.path.join(motion_dir, "../pkl", yaml_name + ".pkl"))

        if not no_keybody and not regen_pkl and os.path.exists(pkl_file):
            try:
                print('Loading from PKL file')
                self.deserialize_motions(pkl_file)
            except Exception as e:
                print(f"Error loading pkl file: {e}")
                print("Fallback: loading from yaml")
                self._device = "cpu"
                self._load_motions(motion_file, no_keybody)
                self.serialize_motions(pkl_file)
        else:
            print("No PKL file found or regeneration requested, loading from YAML")
            self._load_motions(motion_file, no_keybody)
            self.serialize_motions(pkl_file)

        motions = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._load_device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._load_device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._load_device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._load_device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._load_device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._load_device)
        self.lbp = torch.cat([m for m in self._motions_local_key_body_pos], dim=0).float().to(self._load_device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._load_device)

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n, max_difficulty=None):
        if max_difficulty is None:
            return torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        diff_weighted = self._motion_weights * (self._motion_difficulty <= max_difficulty).float()
        diff_weighted /= diff_weighted.sum()
        return torch.multinomial(diff_weighted, num_samples=n, replacement=True)

    def sample_time(self, motion_ids, truncate_time=None):
        motion_ids = motion_ids.to(self._load_device)
        phase = torch.rand(motion_ids.shape, device=self._load_device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            motion_len -= truncate_time
        return (phase * motion_len).to(self._device)

    def get_motion_difficulty(self, motion_ids):
        motion_ids = motion_ids.to(self._load_device)
        return self._motion_difficulty[motion_ids].to(self._device)

    def get_motion_files(self, motion_ids):
        return [self._motion_files[i] for i in motion_ids]

    def get_motion_length(self, motion_ids):
        motion_ids = motion_ids.to(self._load_device)
        return self._motion_lengths[motion_ids].to(self._device)

    def get_motion_fps(self, motion_ids):
        motion_ids = motion_ids.to(self._load_device)
        return self._motion_fps[motion_ids].to(self._device)

    def get_motion_num_frames(self, motion_ids):
        motion_ids = motion_ids.to(self._load_device)
        return self._motion_num_frames[motion_ids].to(self._device)

    def get_motion_description(self, motion_id):
        return self.motion_description[motion_id]

    def get_motion_left_ground_offset(self, motion_ids):
        motion_ids = motion_ids.to(self._load_device)
        return self.motion_left_ground_offset[motion_ids].to(self._device)

    def get_motion_right_ground_offset(self, motion_ids):
        motion_ids = motion_ids.to(self._load_device)
        return self.motion_right_ground_offset[motion_ids].to(self._device)

    def get_motion_state(self, motion_ids, motion_times, get_lbp=False):
        motion_ids = motion_ids.to(self._load_device)
        motion_times = motion_times.to(self._load_device)
        n = len(motion_ids)
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]
        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]
        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]
        root_vel = self.grvs[f0l]
        root_ang_vel = self.gravs[f0l]
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        dof_vel = self.dvs[f0l]
        blend = blend.unsqueeze(-1)
        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)
        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        local_rot = torch_utils.slerp(local_rot0, local_rot1, blend.unsqueeze(-1))
        dof_pos = self._local_rotation_to_dof(local_rot)
        if get_lbp:
            lbp0 = self.lbp[f0l]
            lbp1 = self.lbp[f1l]
            lbp = (1.0 - blend_exp) * lbp0 + blend_exp * lbp1
            return (root_pos.to(self._device), root_rot.to(self._device), dof_pos.to(self._device),
                    root_vel.to(self._device), root_ang_vel.to(self._device), dof_vel.to(self._device),
                    key_pos.to(self._device), lbp.to(self._device))
        return (root_pos.to(self._device), root_rot.to(self._device), dof_pos.to(self._device),
                root_vel.to(self._device), root_ang_vel.to(self._device), dof_vel.to(self._device),
                key_pos.to(self._device))

    def _load_motions(self, motion_file, no_keybody):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motions_local_key_body_pos = []
        self._motion_difficulty = []

        motion_files, motion_weights, motion_difficulty, self.motion_description, \
            self.motion_left_ground_offset, self.motion_right_ground_offset = \
            self._fetch_motion_files(motion_file)
        for f, curr_file in enumerate(motion_files):
            curr_motion = SkeletonMotion.from_file(curr_file)
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.tensor.shape[0]
            curr_len = curr_dt * (num_frames - 1)
            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._load_device)
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._load_device)
                curr_motion._skeleton_tree._parent_indices = \
                    curr_motion._skeleton_tree._parent_indices.to(self._load_device)
                curr_motion._skeleton_tree._local_translation = \
                    curr_motion._skeleton_tree._local_translation.to(self._load_device)
                curr_motion._rotation = curr_motion._rotation.to(self._load_device)
            if not no_keybody:
                curr_key_body_pos = torch.from_numpy(
                    np.load(os.path.splitext(curr_file)[0] + "_key_bodies.npy")
                ).to(self._load_device)
            else:
                curr_key_body_pos = torch.zeros((num_frames, len(self._key_body_ids), 3), 
                                                dtype=torch.float32, device=self._load_device)
            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            self._motions_local_key_body_pos.append(curr_key_body_pos)
            self._motion_weights.append(motion_weights[f])
            self._motion_files.append(curr_file)
            self._motion_difficulty.append(motion_difficulty[f])
        self._motion_difficulty = torch.tensor(self._motion_difficulty, 
                                               device=self._load_device, dtype=torch.float32)
        self._motion_lengths = torch.tensor(self._motion_lengths, 
                                           device=self._load_device, dtype=torch.float32)
        self._motion_weights = torch.tensor(self._motion_weights, 
                                           device=self._load_device, dtype=torch.float32)
        self._motion_weights /= self._motion_weights.sum()
        self._motion_fps = torch.tensor(self._motion_fps, 
                                        device=self._load_device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, 
                                       device=self._load_device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, 
                                               device=self._load_device)
        self.motion_left_ground_offset = torch.tensor(self.motion_left_ground_offset, 
                                                      device=self._load_device, dtype=torch.float32)
        self.motion_right_ground_offset = torch.tensor(self.motion_right_ground_offset, 
                                                       device=self._load_device, dtype=torch.float32)
        print(f"Loaded {len(self._motions)} motions with a total length of {self.get_total_length():.3f}s.")

    def serialize_motions(self, pkl_file):
        objects = [self._motions,
                   self._motion_lengths,
                   self._motion_weights,
                   self._motion_fps,
                   self._motion_dt,
                   self._motion_num_frames,
                   self._motion_files,
                   self._motions_local_key_body_pos,
                   self._motion_difficulty,
                   self.motion_description,
                   self.motion_left_ground_offset,
                   self.motion_right_ground_offset]
        with open(pkl_file, 'wb') as outp:
            pickle.dump(objects, outp, pickle.HIGHEST_PROTOCOL)
        print("Saved to: ", pkl_file)

    def deserialize_motions(self, pkl_file):
        with open(pkl_file, 'rb') as inp:
            objects = pickle.load(inp)
        self._motions = []
        for motion in objects[0]:
            motion.tensor = motion.tensor.to(self._load_device)
            motion._skeleton_tree._parent_indices = \
                motion._skeleton_tree._parent_indices.to(self._load_device)
            motion._skeleton_tree._local_translation = \
                motion._skeleton_tree._local_translation.to(self._load_device)
            motion._rotation = motion._rotation.to(self._load_device)
            self._motions.append(motion)
        self._motion_lengths = objects[1].to(self._load_device)
        self._motion_weights = objects[2].to(self._load_device)
        self._motion_fps = objects[3].to(self._load_device)
        self._motion_dt = objects[4].to(self._load_device)
        self._motion_num_frames = objects[5].to(self._load_device)
        self._motion_files = objects[6]
        self._motions_local_key_body_pos = objects[7]
        self._motion_difficulty = objects[8].to(self._load_device)
        self.motion_description = objects[9]
        try:
            self.motion_left_ground_offset = objects[10].to(self._load_device)
            self.motion_right_ground_offset = objects[11].to(self._load_device)
        except:
            self.motion_left_ground_offset = torch.zeros(len(self._motions), device=self._load_device)
            self.motion_right_ground_offset = torch.zeros(len(self._motions), device=self._load_device)
        print(f"Loaded {len(self._motions)} motions with a total length of {self.get_total_length():.3f}s.")

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            dir_name = os.path.join(os.path.dirname(motion_file), motion_config['motions']["root"]
            )
            motion_files, motion_weights, motion_difficulty = [], [], []
            motion_description, left_off, right_off = [], [], []
            for entry, info in motion_config['motions'].items():
                if entry == "root": continue
                motion_files.append(os.path.normpath(os.path.join(dir_name, entry + ".npy")))
                motion_weights.append(info['weight'])
                motion_difficulty.append(info['difficulty'])
                motion_description.append(info['description'])
                left_off.append(float(info.get('left_ground_offset', 0.0)))
                right_off.append(float(info.get('right_ground_offset', 0.0)))
            return motion_files, motion_weights, motion_difficulty, motion_description, left_off, right_off
        else:
            return [motion_file], [1.0], [0], ["None"], [0], [0]

    def _calc_frame_blend(self, time, length, num_frames, dt):
        phase = torch.clip(time / length, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt
        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.get_motion(0).num_joints

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []
        for f in range(num_frames - 1):
            vel = self._local_rotation_to_dof_vel(motion.local_rotation[f], motion.local_rotation[f+1], dt)
            dof_vels.append(vel)
        dof_vels.append(dof_vels[-1])
        return torch.stack(dof_vels, dim=0)

    def _local_rotation_to_dof(self, local_rot):
        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), device=self._load_device)
        for j, body_id in enumerate(self._dof_body_ids):
            off = self._dof_offsets[j]
            size = self._dof_offsets[j+1] - off
            q = local_rot[:, body_id]
            if size == 3:
                exp = torch_utils.quat_to_exp_map(q)
                dof_pos[:, off:off+3] = exp
            elif size == 2:
                exp = torch_utils.quat_to_exp_map(q)
                dof_pos[:, off:off+2] = exp[:, :2]
            else:
                theta, axis = torch_utils.quat_to_angle_axis(q)
                dof_pos[:, off] = normalize_angle(theta * axis[...,1])
        return dof_pos

    # exbody3
    # def _local_rotation_to_dof(self, local_rot):
    #     n = local_rot.shape[0]
    #     dof_pos = torch.zeros((n, self._num_dof), device=self._load_device)

    #     for j, body_id in enumerate(self._dof_body_ids):
    #         off = self._dof_offsets[j]
    #         size = self._dof_offsets[j + 1] - off
    #         q = local_rot[:, body_id]

    #         if size == 3:
    #             exp = torch_utils.quat_to_exp_map(q)
    #             dof_pos[:, off:off + 3] = exp
    #         elif size == 2:
    #             exp = torch_utils.quat_to_exp_map(q)
    #             dof_pos[:, off:off + 2] = exp[:, :2]
    #         else:
    #             theta, axis = torch_utils.quat_to_angle_axis(q)
    #             main_axis = torch.argmax(torch.abs(axis), dim=-1)  # [n]
    #             sign = torch.sign(axis[torch.arange(n), main_axis])  # [n]
    #             dof_angle = theta * sign
    #             dof_pos[:, off] = normalize_angle(dof_angle)

    #     return dof_pos

    def _local_rotation_to_dof_vel(self, rot0, rot1, dt):
        diff = quat_mul_norm(quat_inverse(rot0), rot1)
        angle, axis = quat_angle_axis(diff)
        vel = axis * angle.unsqueeze(-1) / dt
        dof_vel = torch.zeros(self._num_dof, device=self._load_device)
        for j, body_id in enumerate(self._dof_body_ids):
            off = self._dof_offsets[j]
            size = self._dof_offsets[j+1] - off
            if size >= 2:
                dof_vel[off:off+size] = vel[body_id, :size]
            else:
                dof_vel[off] = vel[body_id,1]
        return dof_vel
