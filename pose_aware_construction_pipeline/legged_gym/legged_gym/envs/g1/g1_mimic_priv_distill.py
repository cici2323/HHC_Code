from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.utils.math import *
from legged_gym.envs.g1.g1_mimic_priv import G1MimicPriv, global_to_local, local_to_global
from isaacgym import gymtorch, gymapi, gymutil

import torch_utils

class G1MimicPrivDistill(G1MimicPriv):
    
    def compute_observations(self):
        super().compute_observations()
        self.extras["extra_hist_obs"] = self.get_extra_hist_obs()
        self.extras["decoder_demo_obs"] = self.get_decoder_demo_obs()  # root vel

    def get_decoder_demo_obs(self):
        return self.compute_obs_demo()
    
    def get_extra_hist_obs(self):
        return self.obs_extra_history_buf.view(self.num_envs, -1)