"""Analytical CPU checks for the paper's seven motion-tracking equations."""

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
from rsl_rl.residual.motion_rewards import (
    PAPER_MOTION_SCALES,
    PAPER_MOTION_WEIGHTS,
    motion_tracking_reward,
    motion_tracking_terms,
)
from rsl_rl.residual.rewards import adaptation_reward


class MotionRewardTests(unittest.TestCase):
    def inputs(self, batch=2):
        shapes = {
            "dof_position": (batch, 4),
            "keypoint_position": (batch, 2, 3),
            "root_velocity": (batch, 3),
            "root_angular_velocity": (batch, 3),
            "heading": (batch,),
            "height": (batch,),
            "body_rotation": (batch, 3),
        }
        values = {name: torch.zeros(shape, dtype=torch.float64) for name, shape in shapes.items()}
        values.update({"reference_" + name: value.clone() for name, value in list(values.items())})
        return values

    def assert_tensor_close(self, value, expected):
        expected = torch.as_tensor(expected, dtype=value.dtype, device=value.device)
        self.assertTrue(torch.allclose(value, expected, rtol=1e-10, atol=1e-10), (value, expected))

    def test_perfect_tracking_has_paper_maximum(self):
        terms = motion_tracking_reward(**self.inputs())
        self.assertEqual(set(terms), set(PAPER_MOTION_WEIGHTS) | {"total"})
        for name, weight in PAPER_MOTION_WEIGHTS.items():
            self.assert_tensor_close(terms[name], [weight, weight])
        self.assert_tensor_close(terms["total"], [26, 26])

    def test_independent_analytical_values_use_unsquared_norms(self):
        values = self.inputs()
        values["dof_position"][0] = torch.tensor([1, -2, 3, -4], dtype=torch.float64)
        values["keypoint_position"][0] = torch.tensor([[3, 0, 0], [0, 4, 0]], dtype=torch.float64)
        values["root_velocity"][0] = torch.tensor([0, 0.3, 0.4], dtype=torch.float64)
        values["root_angular_velocity"][0] = torch.tensor([0, 0.6, 0.8], dtype=torch.float64)
        values["heading"][0] = -0.25
        values["height"][0] = -0.1
        values["body_rotation"][0] = torch.tensor([0.3, -0.4, 0], dtype=torch.float64)
        terms = motion_tracking_reward(**values)
        expected = {
            "dof": 5 * math.exp(-0.25 * 10),
            "keypoint": 4 * math.exp(-0.25 * 5),
            "root_velocity": 6 * math.exp(-4 * 0.5),
            "root_angular_velocity": 3 * math.exp(-2 * 1),
            "heading": 3 * math.exp(-2 * 0.25),
            "height": 2 * math.exp(-10 * 0.1),
            "body_rotation": 3 * math.exp(-2 * 0.5),
        }
        for name, value in expected.items():
            self.assert_tensor_close(terms[name], [value, PAPER_MOTION_WEIGHTS[name]])
        self.assert_tensor_close(terms["total"], [sum(expected.values()), 26])

    def test_heading_and_body_rotation_wrap_at_pi(self):
        values = self.inputs(batch=1)
        values["heading"].fill_(math.pi - 0.05)
        values["reference_heading"].fill_(-math.pi + 0.05)
        values["body_rotation"].fill_(math.pi - 0.05)
        values["reference_body_rotation"].fill_(-math.pi + 0.05)
        terms = motion_tracking_reward(**values)
        self.assert_tensor_close(terms["heading"], [3 * math.exp(-0.2)])
        self.assert_tensor_close(terms["body_rotation"], [3 * math.exp(-2 * math.sqrt(3 * 0.1**2))])
        values["heading"] += 2 * math.pi
        values["body_rotation"] -= 4 * math.pi
        after = motion_tracking_reward(**values)
        self.assert_tensor_close(after["heading"], terms["heading"])
        self.assert_tensor_close(after["body_rotation"], terms["body_rotation"])

    def test_joint_positions_are_not_angularly_wrapped(self):
        values = self.inputs(batch=1)
        values["dof_position"][0, 0] = 2 * math.pi
        self.assert_tensor_close(motion_tracking_reward(**values)["dof"], [5 * math.exp(-0.25 * 2 * math.pi)])

    def test_unweighted_terms_and_partial_coefficient_overrides(self):
        values = self.inputs()
        values["height"].fill_(0.1)
        scales_before = dict(PAPER_MOTION_SCALES)
        weights_before = dict(PAPER_MOTION_WEIGHTS)
        terms = motion_tracking_terms(**values, scales={"height": 2})
        self.assertEqual(set(terms), set(PAPER_MOTION_WEIGHTS))
        self.assert_tensor_close(terms["height"], [math.exp(-0.2)] * 2)
        weighted = motion_tracking_reward(**values, scales={"height": 2}, weights={"height": 4, "dof": 0})
        self.assert_tensor_close(weighted["height"], [4 * math.exp(-0.2)] * 2)
        self.assert_tensor_close(weighted["dof"], [0, 0])
        self.assert_tensor_close(weighted["total"], [19 + 4 * math.exp(-0.2)] * 2)
        self.assertEqual(PAPER_MOTION_SCALES, scales_before)
        self.assertEqual(PAPER_MOTION_WEIGHTS, weights_before)

    def test_motion_plus_collaboration_scales_only_tracking_by_point_two(self):
        tracking = motion_tracking_reward(**self.inputs())["total"]
        regularization = torch.tensor([-2, -3], dtype=torch.float64)
        terms = adaptation_reward(
            tracking, regularization,
            current_support_points=torch.zeros(2, 2, 3, dtype=torch.float64),
            target_support_points=torch.zeros(2, 2, 3, dtype=torch.float64),
            workspace_overlap=torch.tensor([False, True]),
            base_angular_velocity=torch.zeros(2, 3, dtype=torch.float64),
            foot_velocity=torch.zeros(2, 2, 3, dtype=torch.float64),
            contacts=torch.zeros(2, 2, dtype=torch.bool),
        )
        self.assert_tensor_close(terms["tracking"], [5.2, 5.2])
        self.assert_tensor_close(terms["regularization"], [-2, -3])
        self.assert_tensor_close(terms["total"], [9.7, 5.7])

    def test_mismatched_shapes_are_not_broadcast(self):
        for name in self.inputs():
            with self.subTest(name=name):
                values = self.inputs()
                values[name] = values[name][0:1]
                with self.assertRaises(ValueError):
                    motion_tracking_reward(**values)
        values = self.inputs()
        values["heading"] = values["heading"].unsqueeze(-1)
        with self.assertRaises(ValueError):
            motion_tracking_reward(**values)

    def test_empty_batch_joints_and_keypoints_are_rejected(self):
        with self.assertRaises(ValueError):
            motion_tracking_reward(**self.inputs(batch=0))
        for name, shape in (("dof_position", (2, 0)), ("keypoint_position", (2, 0, 3))):
            with self.subTest(name=name):
                values = self.inputs()
                values[name] = torch.zeros(shape, dtype=torch.float64)
                values["reference_" + name] = values[name].clone()
                with self.assertRaises(ValueError):
                    motion_tracking_reward(**values)

    def test_nonfinite_inputs_and_wrong_dtypes_are_rejected(self):
        for name in self.inputs():
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, invalid=invalid):
                    values = self.inputs()
                    values[name].flatten()[0] = invalid
                    with self.assertRaises(ValueError):
                        motion_tracking_reward(**values)
        values = self.inputs()
        values["reference_height"] = values["reference_height"].to(torch.float32)
        with self.assertRaises(ValueError):
            motion_tracking_reward(**values)
        values = self.inputs()
        values["root_velocity"] = values["root_velocity"].to(torch.int64)
        with self.assertRaises(TypeError):
            motion_tracking_reward(**values)

    def test_bad_overrides_are_rejected(self):
        for name in ("weights", "scales"):
            for value in (-1, float("inf"), float("nan")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        motion_tracking_reward(**self.inputs(), **{name: {"dof": value}})
            with self.assertRaises(ValueError):
                motion_tracking_reward(**self.inputs(), **{name: {"unknown": 1}})
            with self.assertRaises(TypeError):
                motion_tracking_reward(**self.inputs(), **{name: {"dof": True}})
            with self.assertRaises(TypeError):
                motion_tracking_reward(**self.inputs(), **{name: [1]})
        with self.assertRaises(ValueError):
            motion_tracking_reward(**self.inputs(), scales={"dof": 0})

    def test_common_keypoint_translation_preserves_reward(self):
        values = self.inputs()
        values["keypoint_position"][:, 0, 0] = 0.3
        before = motion_tracking_reward(**values)["keypoint"]
        offset = torch.tensor([2, -3, 4], dtype=torch.float64)
        values["keypoint_position"] += offset
        values["reference_keypoint_position"] += offset
        self.assert_tensor_close(motion_tracking_reward(**values)["keypoint"], before)


if __name__ == "__main__":
    unittest.main()
