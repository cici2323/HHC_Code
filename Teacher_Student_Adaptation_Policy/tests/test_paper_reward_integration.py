"""CPU integration checks for paper rewards and the legacy reward lifecycle.

The synthetic parent models the legacy environment's regularization setup and
compute-before-reset lifecycle. These checks do not replace an Isaac Gym run.
"""

import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.residual.motion_rewards import PAPER_MOTION_SCALES, PAPER_MOTION_WEIGHTS
from rsl_rl.residual.task_rewards import PaperRewardMixin


def reward_config():
    with (ROOT / "configs" / "rewards.json").open() as stream:
        return json.load(stream)


class SyntheticLegacyEnvironment:
    """Only the reward state and lifecycle needed by PaperRewardMixin."""

    def _prepare_reward_function(self):
        self.reward_scales = {
            name: scale * self.dt for name, scale in self.raw_regularization_scales.items() if scale != 0
        }
        self.reward_names = [name for name in self.reward_scales if name != "termination"]
        self.reward_functions = [getattr(self, "_reward_" + name) for name in self.reward_names]
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_scales
        }

    def _reward_alive(self):
        return self.alive

    def _reward_action_rate(self):
        return self.action_rate

    def _reward_termination(self):
        return self.terminations


class SyntheticPaperEnvironment(PaperRewardMixin, SyntheticLegacyEnvironment):
    def __init__(self, stage="teacher", construction=True, dt=0.02):
        config = reward_config()
        self.num_envs = 2
        self.device = "cpu"
        self.dt = dt
        self.cfg = SimpleNamespace(
            reward_profile=SimpleNamespace(
                stage=stage,
                tracking_weight=config["adaptation_tracking_weight"] if stage == "adaptation" else 1.0,
                motion_weights=copy.deepcopy(config["motion_weights"]),
                motion_scales=copy.deepcopy(config["motion_scales"]),
                collaboration=copy.deepcopy(config["collaboration"]),
            ),
            construction=SimpleNamespace(enabled=construction),
        )
        # Include a termination penalty to exercise the legacy special case.
        self.raw_regularization_scales = {
            "alive": config["regularization"]["alive"],
            "action_rate": config["regularization"]["action_rate"],
            "termination": -2.0,
        }
        self.alive = torch.ones(2)
        self.action_rate = torch.tensor([2.0, 4.0])
        self.terminations = torch.tensor([0.0, 1.0])
        self.rew_buf = torch.zeros(2)
        self.extras = {}
        self.events = []
        self.after_reset = False
        shapes = {
            "dof_position": (2, 23),
            "keypoint_position": (2, 3, 3),
            "root_velocity": (2, 3),
            "root_angular_velocity": (2, 3),
            "heading": (2,),
            "height": (2,),
            "body_rotation": (2, 3),
        }
        self.motion_inputs = {name: torch.zeros(shape) for name, shape in shapes.items()}
        self.motion_inputs.update({
            "reference_" + name: value.clone() for name, value in list(self.motion_inputs.items())
        })
        self.collaboration_inputs = {
            "current_support_points": torch.zeros(2, 2, 3),
            "target_support_points": torch.zeros(2, 2, 3),
            "workspace_overlap": torch.zeros(2, dtype=torch.bool),
            "base_angular_velocity": torch.zeros(2, 3),
            "foot_velocity": torch.zeros(2, 2, 3),
            "contacts": torch.zeros(2, 2, dtype=torch.bool),
        }
        self._prepare_reward_function()

    def motion_reward_inputs(self):
        self.events.append(("motion", self.after_reset))
        return self.motion_inputs

    def collaboration_reward_inputs(self):
        self.events.append(("collaboration", self.after_reset))
        if not self.cfg.construction.enabled:
            raise AssertionError("Motion-only mode must not request support/workspace measurements")
        return self.collaboration_inputs

    def stabilization_reward_inputs(self):
        self.events.append(("stabilization", self.after_reset))
        return {
            name: self.collaboration_inputs[name]
            for name in ("base_angular_velocity", "foot_velocity", "contacts")
        }

    def step_then_reset(self):
        self.compute_reward()
        reward = self.rew_buf.clone()
        self.events.append(("reset", self.after_reset))
        self.after_reset = True
        for value in self.motion_inputs.values():
            value.zero_()
        for value in self.collaboration_inputs.values():
            value.zero_()
        return reward


class PaperRewardIntegrationTests(unittest.TestCase):
    def assert_tensor_close(self, actual, expected):
        expected = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7), (actual, expected))

    def assert_log_sum_matches_reward(self, env):
        logs = env.extras["reward_terms"]
        component_sum = sum(value for name, value in logs.items() if name != "total")
        self.assert_tensor_close(component_sum, env.rew_buf)
        self.assert_tensor_close(logs["total"], env.rew_buf)

    def test_saved_profile_matches_paper_equations_and_legacy_regularization(self):
        config = reward_config()
        self.assertEqual(config["motion_weights"], PAPER_MOTION_WEIGHTS)
        self.assertEqual(config["motion_scales"], PAPER_MOTION_SCALES)
        self.assertEqual(config["adaptation_tracking_weight"], 0.2)
        self.assertEqual(config["collaboration"], {
            "support_weight": 5.0, "workspace_weight": 3.0,
            "stability_weight": 1.5, "slip_weight": 0.05,
            "support_tolerance": 0.02, "support_scale": 0.05,
        })
        self.assertEqual(config["regularization"], {
            "alive": 3.0, "orientation": -4.0, "dof_acc": -0.0000003,
            "action_rate": -0.1, "dof_error": -0.1, "feet_stumble": -2.0,
            "dof_pos_limits": -10.0, "feet_air_time": 10.0, "feet_force": -0.003,
            "ankle_action": -0.1, "waist_roll_pitch_error": -0.5,
        })

    def test_teacher_uses_all_motion_and_unchanged_regularization(self):
        env = SyntheticPaperEnvironment()
        env.compute_reward()
        self.assert_tensor_close(env.rew_buf, [(26 + 3 - 0.2) * 0.02, (26 + 3 - 0.4 - 2) * 0.02])
        logs = env.extras["reward_terms"]
        self.assert_tensor_close(logs["regularization"], [2.8 * 0.02, 0.6 * 0.02])
        self.assert_tensor_close(env.episode_sums["termination"], [0, -2 * 0.02])
        self.assertEqual(env.events, [("motion", False)])
        self.assertNotIn("support", logs)
        self.assert_log_sum_matches_reward(env)

    def test_adaptation_scales_only_motion_and_adds_collaboration_once(self):
        teacher = SyntheticPaperEnvironment()
        teacher.compute_reward()
        env = SyntheticPaperEnvironment(stage="adaptation")
        env.compute_reward()
        logs = env.extras["reward_terms"]
        self.assert_tensor_close(env.rew_buf, [(5.2 + 6.5 + 2.8) * 0.02, (5.2 + 6.5 + 0.6) * 0.02])
        self.assert_tensor_close(logs["regularization"], teacher.extras["reward_terms"]["regularization"])
        for name in PAPER_MOTION_WEIGHTS:
            self.assert_tensor_close(logs["motion_" + name], 0.2 * teacher.extras["reward_terms"]["motion_" + name])
        self.assert_tensor_close(logs["support"], [0.1, 0.1])
        self.assert_tensor_close(logs["stability"], [0.03, 0.03])
        self.assert_log_sum_matches_reward(env)

    def test_transition_support_overlap_and_contact_slip_are_snapshotted_before_reset(self):
        env = SyntheticPaperEnvironment(stage="adaptation")
        inputs = env.collaboration_inputs
        inputs["current_support_points"][0, :, 0] = torch.tensor([0.02, 0.07])
        inputs["workspace_overlap"][0] = True
        inputs["base_angular_velocity"][0] = torch.tensor([0.0, 3.0, 4.0])
        inputs["foot_velocity"][0] = torch.tensor([[3.0, 4.0, 0.0], [100.0, 0.0, 0.0]])
        inputs["contacts"][0, 0] = True
        reward = env.step_then_reset()
        logs = env.extras["reward_terms"]
        self.assertEqual(env.events, [("motion", False), ("collaboration", False), ("reset", False)])
        self.assert_tensor_close(logs["support"], [5 * math.exp(-0.5) * 0.02, 5 * 0.02])
        self.assert_tensor_close(logs["workspace"], [-3 * 0.02, 0])
        self.assert_tensor_close(logs["stability"], [1.5 * math.exp(-5) * 0.02, 1.5 * 0.02])
        self.assert_tensor_close(logs["slip"], [-0.05 * 25 * 0.02, 0])
        expected_first = (5.2 + 2.8 + 5 * math.exp(-0.5) - 3 + 1.5 * math.exp(-5) - 1.25) * 0.02
        self.assert_tensor_close(reward, [expected_first, (5.2 + 6.5 + 0.6) * 0.02])
        self.assertFalse(bool(inputs["workspace_overlap"].any()))
        self.assertFalse(bool(inputs["foot_velocity"].any()))
        self.assert_log_sum_matches_reward(env)

    def test_single_control_dt_scaling_for_every_component(self):
        slow = SyntheticPaperEnvironment(stage="adaptation", dt=0.02)
        fast = SyntheticPaperEnvironment(stage="adaptation", dt=0.01)
        for env in (slow, fast):
            env.collaboration_inputs["workspace_overlap"][:] = True
            env.collaboration_inputs["foot_velocity"][:, 0, 0] = 2
            env.collaboration_inputs["contacts"][:, 0] = True
            env.compute_reward()
        for name, value in slow.extras["reward_terms"].items():
            with self.subTest(term=name):
                self.assert_tensor_close(value, 2 * fast.extras["reward_terms"][name])
        self.assert_tensor_close(slow.extras["reward_terms"]["motion_dof"], [5 * 0.2 * 0.02] * 2)
        self.assert_tensor_close(slow.extras["reward_terms"]["support"], [5 * 0.02] * 2)

    def test_negative_total_is_not_clipped(self):
        env = SyntheticPaperEnvironment(stage="adaptation")
        env.collaboration_inputs["foot_velocity"][:, 0, 0] = 100
        env.collaboration_inputs["contacts"][:, 0] = True
        env.compute_reward()
        self.assertTrue(bool((env.rew_buf < 0).all()))
        self.assert_tensor_close(env.rew_buf, [(5.2 + 6.5 + 2.8 - 500) * 0.02, (5.2 + 6.5 + 0.6 - 500) * 0.02])
        self.assert_log_sum_matches_reward(env)

    def test_motion_only_has_explicit_zero_support_and_workspace(self):
        env = SyntheticPaperEnvironment(stage="adaptation", construction=False)
        # Nonzero placeholder geometry is deliberately inaccessible in this mode.
        env.collaboration_inputs["current_support_points"].fill_(100)
        env.collaboration_inputs["workspace_overlap"][:] = True
        env.collaboration_inputs["foot_velocity"][:, 0, 0] = 2
        env.collaboration_inputs["contacts"][:, 0] = True
        env.compute_reward()
        logs = env.extras["reward_terms"]
        self.assertEqual(env.events, [("motion", False), ("stabilization", False)])
        self.assert_tensor_close(logs["support"], [0, 0])
        self.assert_tensor_close(logs["workspace"], [0, 0])
        self.assert_tensor_close(logs["stability"], [1.5 * 0.02] * 2)
        self.assert_tensor_close(logs["slip"], [-0.05 * 4 * 0.02] * 2)
        self.assert_tensor_close(env.rew_buf, [(5.2 + 1.5 - 0.2 + 2.8) * 0.02, (5.2 + 1.5 - 0.2 + 0.6) * 0.02])
        self.assert_log_sum_matches_reward(env)

    def test_episode_logging_accumulates_without_aliasing_last_transition(self):
        env = SyntheticPaperEnvironment(stage="adaptation")
        env.compute_reward()
        first = {name: value.clone() for name, value in env.extras["reward_terms"].items()}
        saved_log_reference = env.extras["reward_terms"]
        env.collaboration_inputs["workspace_overlap"][:] = True
        env.compute_reward()
        second = env.extras["reward_terms"]
        for name, value in first.items():
            self.assert_tensor_close(saved_log_reference[name], value)
            if name not in ("total", "regularization"):
                self.assert_tensor_close(env.episode_sums[name], value + second[name])
        self.assert_tensor_close(env.episode_sums["paper_total"], first["total"] + second["total"])
        self.assert_tensor_close(env.episode_sums["alive"], [2 * 3 * 0.02] * 2)
        self.assert_log_sum_matches_reward(env)

    def test_changed_motion_and_collaboration_profile_weights_change_actual_reward(self):
        env = SyntheticPaperEnvironment(stage="adaptation")
        env.compute_reward()
        before = env.rew_buf.clone()
        env.cfg.reward_profile.motion_weights["dof"] = 7.0
        env.cfg.reward_profile.collaboration["support_weight"] = 8.0
        env.compute_reward()
        self.assert_tensor_close(env.rew_buf - before, [(0.2 * (7 - 5) + (8 - 5)) * 0.02] * 2)
        self.assert_log_sum_matches_reward(env)

    def test_changed_motion_exponent_profile_changes_actual_reward(self):
        env = SyntheticPaperEnvironment(stage="teacher")
        env.motion_inputs["height"].fill_(0.1)
        env.compute_reward()
        before = env.rew_buf.clone()
        env.cfg.reward_profile.motion_scales["height"] = 2.0
        env.compute_reward()
        expected_change = 2 * (math.exp(-0.2) - math.exp(-1)) * 0.02
        self.assert_tensor_close(env.rew_buf - before, [expected_change] * 2)

    def test_nonfinite_regularization_raises_instead_of_publishing_bad_reward(self):
        env = SyntheticPaperEnvironment()
        env.action_rate[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            env.compute_reward()
        self.assertTrue(bool(torch.isfinite(env.rew_buf).all()))


if __name__ == "__main__":
    unittest.main()
