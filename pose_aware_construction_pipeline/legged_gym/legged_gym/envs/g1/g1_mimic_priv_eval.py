from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.utils.math import *
from legged_gym.envs.g1.g1_mimic_priv import G1MimicPriv, global_to_local, local_to_global
from isaacgym import gymtorch, gymapi, gymutil

import torch_utils

class G1MimicPrivEval(G1MimicPriv):
    
    def render_record(self, mode="rgb_array"):
        if self.global_counter % 2 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            imgs = []
            for i in range(self.num_envs):
                cam = self._rendering_camera_handles[i]
                root_pos = self.root_states[i, :3].cpu().numpy()
                cam_pos = root_pos + np.array([-1.5, 0, 0.4])
                self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
                
                img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
                w, h = img.shape
                imgs.append(img.reshape([w, h // 4, 4]))
            return imgs
        return None
    
    def _create_envs(self):
        super()._create_envs()
        if self.cfg.env.record_video or self.cfg.env.record_frame:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1080
            camera_props.height = 720
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))