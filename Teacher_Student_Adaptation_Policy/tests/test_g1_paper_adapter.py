"""Execute the real G1Paper adapter with isolated simulator dependency stubs.

The adapter, paper reward setup, recorded-data loader and coordinate context are
production code. Simulator stepping and the inherited environment are replaced
with lifecycle probes; these tests do not claim physics/simulator validation.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def quat_rotate(quaternion, vector):
    xyz = quaternion[:, :3]
    uv = torch.cross(xyz, vector, dim=-1)
    return vector + 2 * (quaternion[:, 3:4] * uv + torch.cross(xyz, uv, dim=-1))


def quat_rotate_inverse(quaternion, vector):
    inverse = quaternion.clone()
    inverse[:, :3] *= -1
    return quat_rotate(inverse, vector)


def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return (
        torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        torch.asin((2 * (w * y - z * x)).clamp(-1, 1)),
        torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )


class SimulatorBaseProbe:
    def _prepare_reward_function(self):
        self.events.append('base_prepare')
        self.reward_names = []
        self.reward_functions = []
        self.reward_scales = {}
        self.episode_sums = {}

    def post_physics_step(self):
        self.events.append('base_post')
        # Model the documented base ordering: reward, reset, then observations.
        self.events.append('reward')
        self.reward_snapshot = {
            'times': self._motion_times.clone(),
            'demo': self.demo_marker.clone(),
            'context': self.get_residual_context().clone(),
        }
        self.reset_idx(self.reset_ids)
        self.events.append('observations')
        self.observation_snapshot = {
            'times': self._motion_times.clone(),
            'demo': self.demo_marker.clone(),
            'yaw': self.yaw.clone(), 'target_yaw': self.target_yaw.clone(),
            'velocity': self.base_lin_vel.clone(),
            'angular_velocity': self.base_ang_vel.clone(),
            'gravity': self.projected_gravity.clone(),
            'context': self.get_residual_context().clone(),
        }


class MimicDistillProbe(SimulatorBaseProbe):
    def reset_idx(self, env_ids, init=False):
        self.events.append('parent_reset')
        if not len(env_ids):
            return
        self._motion_times[env_ids] = 0.45
        self.root_states[env_ids, :3] = self.env_origins[env_ids] + torch.tensor([2., 3., 1.])
        self.root_states[env_ids, 3:7] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        self.root_states[env_ids, 7:10] = torch.tensor([1., 0., 0.])
        self.root_states[env_ids, 10:13] = torch.tensor([0., 1., 0.])

    def update_demo_obs(self):
        self.events.append('demo')
        self.demo_marker = self._motion_times.clone()
        self._curr_demo_quat = torch.zeros(self.num_envs, 4)
        self._curr_demo_quat[:, 2] = torch.sin(self._motion_times / 2)
        self._curr_demo_quat[:, 3] = torch.cos(self._motion_times / 2)


def load_adapter():
    """Avoid importing legacy package initializers that require Isaac Gym."""
    names = ['paper_adapter_test', 'paper_adapter_test.g1_mimic_priv_distill',
             'paper_adapter_test.g1_mimic_priv', 'isaacgym', 'isaacgym.torch_utils',
             'legged_gym', 'legged_gym.envs', 'legged_gym.envs.base',
             'legged_gym.envs.base.legged_robot']
    modules = {name: ModuleType(name) for name in names}
    modules['paper_adapter_test'].__path__ = []
    modules['paper_adapter_test.g1_mimic_priv_distill'].G1MimicPrivDistill = MimicDistillProbe
    modules['paper_adapter_test.g1_mimic_priv'].local_to_global = lambda q, p, r: p + r[:, None]
    modules['isaacgym.torch_utils'].quat_rotate = quat_rotate
    modules['isaacgym.torch_utils'].quat_rotate_inverse = quat_rotate_inverse
    modules['legged_gym.envs.base.legged_robot'].LeggedRobot = SimulatorBaseProbe
    modules['legged_gym.envs.base.legged_robot'].euler_from_quaternion = euler_from_quaternion
    spec = importlib.util.spec_from_file_location(
        'paper_adapter_test.g1_paper', ROOT / 'legged_gym/legged_gym/envs/g1/g1_paper.py')
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.G1Paper


G1Paper = load_adapter()


class G1PaperAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data_path = Path(self.directory.name) / 'synthetic_test_only.json'
        self.geometry_path = Path(self.directory.name) / 'synthetic_geometry_test_only.json'
        self.data_path.write_text(json.dumps({
            'schema_version': 1, 'frame': 'environment',
            'provenance': 'Synthetic lifecycle test fixture, not construction training data',
            'samples': [
                {'support_points': [[0, 0, 1], [1, 1, 2]],
                 'human_workspace_boxes': [[[0, 0, 0], [1, 1, 1]]]},
                {'support_points': [[2, 3, 4], [4, 5, 6]],
                 'human_workspace_boxes': [[[3, 4, 5], [6, 7, 8]]]},
            ],
        }))
        self.geometry_path.write_text(json.dumps({
            'schema_version': 1,
            'support_points': [{'body': 'hand_left', 'offset': [0, 0, 0]},
                               {'body': 'hand_right', 'offset': [0, 0, 0]}],
            'robot_spheres': [{'body': 'hand_left', 'offset': [0, 0, 0], 'radius': .1}],
        }))

    def environment(self, construction=True):
        env = G1Paper.__new__(G1Paper)
        env.num_envs, env.device = 2, 'cpu'
        env.events = []
        env.cfg = SimpleNamespace(
            reward_profile=SimpleNamespace(motion_weights={'dof': 5., 'height': 2.}),
            construction=SimpleNamespace(enabled=construction, data_path=str(self.data_path),
                                         geometry_path=str(self.geometry_path)),
        )
        # These exist before _prepare_reward_function in the real constructor.
        env.envs, env.actor_handles = [0], [0]
        env.gym = SimpleNamespace(find_actor_rigid_body_handle=lambda e, a, name:
                                  {'hand_left': 0, 'hand_right': 1}.get(name, -1))
        env._prepare_reward_function()
        return env

    def add_motion_buffers(self, env):
        # The real constructor initializes motion buffers only after _prepare.
        env._motion_times = torch.tensor([.95, .25])
        env._motion_dt = .1
        env._motion_lengths = torch.ones(2)
        env.env_origins = torch.tensor([[10., 20., 0.], [100., 200., 0.]])
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, :3] = env.env_origins + torch.tensor([0., 0., 1.])
        env.root_states[:, 6] = 1.
        env.base_quat = env.root_states[:, 3:7]  # same view semantics as the source
        env.gravity_vec = torch.tensor([[0., 0., -1.], [0., 0., -1.]])
        for name in ('base_lin_vel', 'base_ang_vel', 'projected_gravity'):
            setattr(env, name, torch.full((2, 3), -7.))
        for name in ('roll', 'pitch', 'yaw', 'target_yaw'):
            setattr(env, name, torch.full((2,), -7.))
        env.viewer = None
        env.reset_ids = torch.tensor([0])
        if env.construction_dataset is not None:
            env._construction_ids.zero_()
        env.update_demo_obs()
        env.events.clear()
        return env

    def test_prepare_precedes_motion_buffers_and_binds_real_reward_and_context_setup(self):
        env = self.environment()
        self.assertEqual(env.events, ['base_prepare'])
        self.assertFalse(hasattr(env, '_motion_times'))
        self.assertEqual(set(env.episode_sums),
                         {'motion_dof', 'motion_height', 'support', 'workspace', 'stability', 'slip', 'paper_total'})
        self.assertEqual(env.residual_context_schema['context_dim'], 30)
        self.assertEqual(env.residual_context_schema['data_sha256'], hashlib.sha256(self.data_path.read_bytes()).hexdigest())
        self.assertTrue(torch.equal(env._support_body_ids, torch.tensor([0, 1])))
        self.assertEqual(env._construction_ids.shape, (2,))
        self.assertTrue(torch.equal(env._sphere_radii, torch.tensor([.1])))

    def test_post_physics_reward_then_reset_observation_uses_current_reference_and_context(self):
        env = self.add_motion_buffers(self.environment())
        with mock.patch.object(env.construction_dataset, 'sample', return_value=torch.tensor([1])) as sampler:
            env.post_physics_step()
        sampler.assert_called_once_with(1)
        self.assertEqual(env.events, ['demo', 'base_post', 'reward', 'parent_reset', 'demo', 'observations'])
        torch.testing.assert_close(env.reward_snapshot['times'], torch.tensor([0., .35]))
        torch.testing.assert_close(env.reward_snapshot['demo'], env.reward_snapshot['times'])
        observed = env.observation_snapshot
        torch.testing.assert_close(observed['times'], torch.tensor([.45, .35]))
        torch.testing.assert_close(observed['demo'], observed['times'])
        torch.testing.assert_close(observed['target_yaw'], torch.tensor([.45, -7.]))
        torch.testing.assert_close(observed['yaw'], torch.tensor([math.pi / 2, -7.]))
        torch.testing.assert_close(observed['velocity'][0], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(observed['angular_velocity'][0], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(observed['gravity'][0], torch.tensor([0., 0., -1.]))
        torch.testing.assert_close(observed['velocity'][1], torch.full((3,), -7.))
        torch.testing.assert_close(observed['context'][0, :6], torch.tensor([0., 0., 3., 2., -2., 5.]), atol=1e-6, rtol=0)
        self.assertFalse(torch.equal(observed['context'][0], env.reward_snapshot['context'][0]))
        torch.testing.assert_close(observed['context'][1], env.reward_snapshot['context'][1])
        self.assertTrue(torch.equal(env._construction_ids, torch.tensor([1, 0])))

    def test_empty_reset_does_not_change_reference_local_state_or_sample_task(self):
        env = self.add_motion_buffers(self.environment())
        old_times, old_velocity = env._motion_times.clone(), env.base_lin_vel.clone()
        with mock.patch.object(env.construction_dataset, 'sample') as sampler:
            env.reset_idx(torch.empty(0, dtype=torch.long))
        sampler.assert_not_called()
        self.assertEqual(env.events, ['parent_reset'])
        torch.testing.assert_close(env._motion_times, old_times)
        torch.testing.assert_close(env.base_lin_vel, old_velocity)

    def test_motion_only_prepare_and_initial_reset_need_no_construction_files(self):
        self.data_path.unlink()
        self.geometry_path.unlink()
        env = self.add_motion_buffers(self.environment(construction=False))
        self.assertIsNone(env.construction_dataset)
        self.assertEqual(env.residual_context_schema, 'none')
        # At initial reset the base has not computed Euler/target yaw attributes yet.
        for name in ('roll', 'pitch', 'yaw', 'target_yaw'):
            delattr(env, name)
        env.reset_idx(torch.tensor([0, 1]), init=True)
        torch.testing.assert_close(env.target_yaw, torch.full((2,), .45))
        torch.testing.assert_close(env.yaw, torch.full((2,), math.pi / 2))
        self.assertEqual(env.get_residual_context().shape, (2, 0))

    def test_invalid_robot_geometry_fails_during_prepare(self):
        geometry = json.loads(self.geometry_path.read_text())
        geometry['support_points'][0]['body'] = 'absent_body'
        self.geometry_path.write_text(json.dumps(geometry))
        with self.assertRaisesRegex(ValueError, 'Unknown robot body'):
            self.environment()


if __name__ == '__main__':
    unittest.main()
