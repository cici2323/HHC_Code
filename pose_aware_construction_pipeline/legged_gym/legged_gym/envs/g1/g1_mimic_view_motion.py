from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.g1.g1_mimic import *
from legged_gym.envs.g1.g1_mimic_eval import G1MimicEval
import os
from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR

class G1MimicViewMotion(G1MimicEval):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.save = True
        cfg.motion.num_envs_as_motions = True
        cfg.motion.no_keybody = True
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # assert only one env and one motion
        self.motion_frame = torch.zeros_like(self._motion_times, dtype=torch.long, device=self.device)
        self.saved_flags = torch.zeros_like(self._motion_times, dtype=torch.bool, device=self.device)
        self.total_frames = self._motion_lib.get_motion_num_frames(self._motion_ids)
        self.motor_strength *= 0.0

        self.motion_names = self._motion_lib.get_motion_files(self._motion_ids)
        self.motion_names = [name.split("/")[-1].split(".")[0] for name in self.motion_names]
        self.to_save_list = []
        for i in range(self.num_envs):
            self.to_save_list.append(torch.zeros((self.total_frames[i], len(self._key_body_ids_sim), 3), dtype=torch.float32, device=self.device))
        
    def post_physics_step(self):
        super().post_physics_step()
        if hasattr(self, 'motion_frame'):
            self._motion_sync()
            done_percentage = (self.motion_frame.float().sum() / self.total_frames.float().sum()).item()
            print(f"done percentage: {done_percentage}")
            if self.save:
                save_ids = torch.where(self.motion_frame == self.total_frames - 1)[0]
                if len(save_ids) > 0:
                    for i in save_ids:
                        if not self.saved_flags[i]:
                            assert self.motion_frame[i] == self.to_save_list[i].shape[0] - 1
                            np.save(os.path.join(ASE_DIR, f"ase/poselib/data/g1_retarget_npy/{self.motion_names[i]}_key_bodies.npy"), self.to_save_list[i].cpu().numpy())
                            print(f"saved {self.motion_names[i]}")
                    self.saved_flags[save_ids] = True
                    
                if torch.all(self.saved_flags):
                    print("all saved")
                    exit()
            self.motion_frame[~self.saved_flags] += 1
            if not self.save:
                self.motion_frame[self.motion_frame >= self.total_frames] = 0

        return
    
    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def compute_reward(self):
        return
    
    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_fps = self._motion_lib.get_motion_fps(self._motion_ids)
        
        motion_times = 1.0 / motion_fps * self.motion_frame

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)


        # clamp dof_pos to be within joint limits
        dof_pos = torch.clamp(dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        if self.save:
            local_end_pos = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim, :3], self.root_states[:, :3])
            for i in range(self.num_envs):
                self.to_save_list[i][self.motion_frame[i], :, :] = local_end_pos[i, :]
        
        return
    
    
    def update_demo_obs(self):
        return