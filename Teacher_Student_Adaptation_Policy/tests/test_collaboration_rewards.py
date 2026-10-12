"""CPU numerical checks for the paper's task reward equations."""

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
from rsl_rl.residual.rewards import adaptation_reward, collaboration_reward


class CollaborationRewardTests(unittest.TestCase):
    def inputs(self, batch=2):
        return {
            "current_support_points": torch.zeros(batch, 2, 3, dtype=torch.float64),
            "target_support_points": torch.zeros(batch, 2, 3, dtype=torch.float64),
            "workspace_overlap": torch.zeros(batch, dtype=torch.bool),
            "base_angular_velocity": torch.zeros(batch, 3, dtype=torch.float64),
            "foot_velocity": torch.zeros(batch, 2, 3, dtype=torch.float64),
            "contacts": torch.zeros(batch, 2, dtype=torch.bool),
        }

    def assert_tensor_close(self, value, expected):
        expected = torch.as_tensor(expected, dtype=value.dtype)
        self.assertTrue(torch.allclose(value, expected, rtol=1e-10, atol=1e-10), (value, expected))

    def test_perfect_alignment_rest_and_no_overlap(self):
        terms = collaboration_reward(**self.inputs())
        self.assertEqual(set(terms), {"support", "workspace", "stability", "slip", "total"})
        self.assert_tensor_close(terms["support"], [5, 5])
        self.assert_tensor_close(terms["stability"], [1.5, 1.5])
        self.assert_tensor_close(terms["workspace"], [0, 0])
        self.assert_tensor_close(terms["slip"], [0, 0])
        self.assert_tensor_close(terms["total"], [6.5, 6.5])

    def test_independent_numerical_terms(self):
        inputs = self.inputs()
        # First environment: one support point at tolerance, one 5 cm beyond it.
        # Mean squared normalized excess = (0**2 + 1**2)/2 = .5.
        inputs["current_support_points"][0, :, 0] = torch.tensor([0.02, 0.07], dtype=torch.float64)
        inputs["current_support_points"][1, :, 0] = 0.01
        inputs["workspace_overlap"][0] = True
        inputs["base_angular_velocity"][0] = torch.tensor([0.0, 3.0, 4.0])
        inputs["foot_velocity"][0] = torch.tensor([[3.0, 4.0, 0.0], [100.0, 0.0, 0.0]])
        inputs["contacts"][0, 0] = True
        terms = collaboration_reward(**inputs)
        self.assert_tensor_close(terms["support"], [5 * math.exp(-0.5), 5])
        self.assert_tensor_close(terms["workspace"], [-3, 0])
        self.assert_tensor_close(terms["stability"], [1.5 * math.exp(-5), 1.5])
        self.assert_tensor_close(terms["slip"], [-1.25, 0])
        self.assert_tensor_close(terms["total"], [5 * math.exp(-0.5) - 3 + 1.5 * math.exp(-5) - 1.25, 6.5])

    def test_adaptation_does_not_scale_regularization(self):
        tracking = torch.tensor([10.0, -5.0], dtype=torch.float64)
        regularization = torch.tensor([-2.0, -4.0], dtype=torch.float64)
        terms = adaptation_reward(tracking, regularization, **self.inputs())
        self.assert_tensor_close(terms["tracking"], [2, -1])
        self.assert_tensor_close(terms["regularization"], [-2, -4])
        self.assert_tensor_close(terms["collaboration_total"], [6.5, 6.5])
        self.assert_tensor_close(terms["total"], [6.5, 1.5])

    def test_points_are_invariant_to_a_common_translation(self):
        inputs = self.inputs()
        inputs["target_support_points"][:, :, 0] = 0.07
        before = collaboration_reward(**inputs)
        offset = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
        inputs["current_support_points"] += offset
        inputs["target_support_points"] += offset
        after = collaboration_reward(**inputs)
        self.assert_tensor_close(after["support"], before["support"])

    def test_slip_sums_all_contacting_feet(self):
        inputs = self.inputs(batch=1)
        inputs["foot_velocity"] = torch.tensor([[[1.0, 0, 0], [0, 2, 0], [0, 0, 3]]], dtype=torch.float64)
        inputs["contacts"] = torch.tensor([[True, False, True]])
        self.assert_tensor_close(collaboration_reward(**inputs)["slip"], [-0.5])

    def test_invalid_shapes_and_boolean_masks_are_rejected(self):
        cases = {
            "target_support_points": torch.zeros(2, 1, 3, dtype=torch.float64),
            "workspace_overlap": torch.zeros(2, 1, dtype=torch.bool),
            "base_angular_velocity": torch.zeros(3, 3, dtype=torch.float64),
            "foot_velocity": torch.zeros(2, 2, dtype=torch.float64),
            "contacts": torch.zeros(2, 3, dtype=torch.bool),
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                inputs = self.inputs()
                inputs[name] = value
                with self.assertRaises(ValueError):
                    collaboration_reward(**inputs)
        for name in ("workspace_overlap", "contacts"):
            with self.subTest(name=name):
                inputs = self.inputs()
                inputs[name] = inputs[name].to(torch.float64)
                with self.assertRaises(TypeError):
                    collaboration_reward(**inputs)

    def test_nonfinite_inputs_and_invalid_scale_are_rejected(self):
        for name in ("current_support_points", "target_support_points", "base_angular_velocity", "foot_velocity"):
            for invalid in (float("nan"), float("inf")):
                with self.subTest(name=name, invalid=invalid):
                    inputs = self.inputs()
                    inputs[name].flatten()[0] = invalid
                    with self.assertRaises(ValueError):
                        collaboration_reward(**inputs)
        for scale in (0.0, -0.1, float("nan")):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    collaboration_reward(**self.inputs(), support_scale=scale)

    def test_adaptation_rejects_broadcasted_or_nonfinite_rewards(self):
        regularization = torch.zeros(2, dtype=torch.float64)
        for tracking in (torch.zeros(2, 1, dtype=torch.float64), torch.tensor([float("nan"), 1], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                adaptation_reward(tracking, regularization, **self.inputs())


if __name__ == "__main__":
    unittest.main()
