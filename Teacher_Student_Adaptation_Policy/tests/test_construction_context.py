"""CPU contract checks using explicit synthetic fixtures, not training data."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
from rsl_rl.residual.construction import ConstructionDataset, robot_frame_context, robot_sphere_box_overlap


class ConstructionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic_test_fixture.json"
        self.data = {
            "schema_version": 1,
            "frame": "environment",
            "provenance": "synthetic unit-test fixture; not validated training data",
            "samples": [
                {"support_points": [[0, 1, 2], [3, 4, 5]],
                 "human_workspace_boxes": [[[-1, -2, -3], [1, 2, 3]]]},
                {"support_points": [[6, 7, 8], [9, 10, 11]],
                 "human_workspace_boxes": [[[3, 4, 5], [6, 7, 8]]]},
            ],
        }

    def load(self, data=None):
        self.path.write_text(json.dumps(self.data if data is None else data), encoding="utf-8")
        return ConstructionDataset(self.path, "cpu", torch.float64)

    def test_world_translation_and_context_contract(self):
        dataset = self.load()
        ids = torch.tensor([1, 0, 1])
        origins = torch.tensor([[1, 2, 3], [10, 20, 30], [-1, -2, -3]], dtype=torch.float64)
        context = dataset.context(ids)
        self.assertEqual(dataset.context_dim, 12)
        self.assertEqual(context.shape, (3, 12))
        self.assertTrue(torch.equal(context[0], context[2]))
        self.assertTrue(torch.equal(context[0], torch.tensor(
            [6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8], dtype=torch.float64)))
        support = dataset.support_points(ids, origins)
        boxes = dataset.workspace_boxes(ids, origins)
        self.assertTrue(torch.equal(support, context[:, :6].reshape(3, 2, 3) + origins[:, None]))
        self.assertTrue(torch.equal(boxes, context[:, 6:].reshape(3, 1, 2, 3) + origins[:, None, None]))
        # Mutating returned tensors must not change the loaded dataset.
        context.zero_()
        support.zero_()
        boxes.zero_()
        self.assertEqual(dataset.context(ids)[0, 0].item(), 6)

    def test_multiple_boxes_preserve_input_order(self):
        for sample in self.data["samples"]:
            sample["human_workspace_boxes"].append([[20, 30, 40], [21, 31, 41]])
        dataset = self.load()
        self.assertEqual(dataset.context_dim, 18)
        self.assertTrue(torch.equal(dataset.context(torch.tensor([0]))[0, -6:],
                                   torch.tensor([20, 30, 40, 21, 31, 41], dtype=torch.float64)))

    def test_sampling_follows_seed_and_accepts_empty_reset(self):
        dataset = self.load()
        torch.manual_seed(1701)
        before = dataset.sample(64)
        torch.manual_seed(1701)
        self.assertTrue(torch.equal(before, dataset.sample(64)))
        generator_one = torch.Generator().manual_seed(1702)
        generator_two = torch.Generator().manual_seed(1702)
        self.assertTrue(torch.equal(dataset.sample(64, generator_one), dataset.sample(64, generator_two)))
        self.assertEqual(set(before.tolist()), {0, 1})
        empty = dataset.sample(0)
        self.assertEqual(empty.dtype, torch.long)
        self.assertEqual(dataset.context(empty).shape, (0, 12))
        self.assertEqual(dataset.support_points(empty, torch.empty(0, 3, dtype=torch.float64)).shape, (0, 2, 3))

    def test_read_only_and_missing_file_behavior(self):
        self.load()
        previous = self.path.read_bytes()
        ConstructionDataset(self.path)
        self.assertEqual(previous, self.path.read_bytes())
        with self.assertRaises(FileNotFoundError):
            ConstructionDataset(self.path.parent / "missing.json")

    def test_invalid_schema_and_task_coordinates(self):
        mutations = [
            lambda d: d.update(schema_version=2),
            lambda d: d.update(schema_version=True),
            lambda d: d.pop("schema_version"),
            lambda d: d.update(frame="world"),
            lambda d: d.pop("frame"),
            lambda d: d.update(samples=[]),
            lambda d: d["samples"].__setitem__(0, []),
            lambda d: d["samples"][0].update(support_points=[[1, 2, 3]]),
            lambda d: d["samples"][0]["support_points"][0].__setitem__(0, True),
            lambda d: d["samples"][0]["support_points"][0].__setitem__(0, "1"),
            lambda d: d["samples"][0]["support_points"][0].__setitem__(0, float("nan")),
            lambda d: d["samples"][0].update(human_workspace_boxes=[]),
            lambda d: d["samples"][0]["human_workspace_boxes"].append([[0, 0, 0], [1, 1, 1]]),
            lambda d: d["samples"][0]["human_workspace_boxes"][0][0].__setitem__(0, 1),
            lambda d: d["samples"][0]["human_workspace_boxes"][0][1].__setitem__(0, -2),
            lambda d: d["samples"][0]["human_workspace_boxes"][0][1].__setitem__(0, float("inf")),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                invalid = copy.deepcopy(self.data)
                mutation(invalid)
                with self.assertRaises(ValueError):
                    self.load(invalid)

    def test_indices_origins_and_sampling_reject_invalid_inputs(self):
        dataset = self.load()
        for ids in (torch.tensor([2]), torch.tensor([-1]), torch.tensor([[0]])):
            with self.assertRaises(ValueError):
                dataset.context(ids)
        with self.assertRaises(TypeError):
            dataset.context(torch.tensor([0.0]))
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                dataset.sample(value)
        for origins in (torch.zeros(3, dtype=torch.float64), torch.zeros(1, 3),
                        torch.tensor([[float("nan"), 0, 0]], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                dataset.support_points(torch.tensor([0]), origins)


class RobotFrameContextTests(unittest.TestCase):
    def inputs(self, batch=1, boxes=1):
        return {
            "support_points_world": torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float64).repeat(batch, 1, 1),
            "workspace_boxes_world": torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.float64).repeat(batch, boxes, 1, 1),
            "root_position": torch.zeros(batch, 3, dtype=torch.float64),
            "root_quaternion_xyzw": torch.tensor([[0, 0, 0, 1]], dtype=torch.float64).repeat(batch, 1),
        }

    def test_identity_and_lexicographic_corner_order(self):
        inputs = self.inputs(batch=2, boxes=2)
        inputs["workspace_boxes_world"][:, 1] += 10
        result = robot_frame_context(**inputs)
        expected_corners = torch.tensor([
            [1, 2, 3], [1, 2, 6], [1, 5, 3], [1, 5, 6],
            [4, 2, 3], [4, 2, 6], [4, 5, 3], [4, 5, 6],
        ], dtype=torch.float64)
        self.assertEqual(result.shape, (2, 54))
        self.assertTrue(torch.equal(result[:, :6], inputs["support_points_world"].flatten(1)))
        self.assertTrue(torch.equal(result[0, 6:30], expected_corners.flatten()))
        self.assertTrue(torch.equal(result[0, 30:], (expected_corners + 10).flatten()))

    def test_common_world_translation_is_removed(self):
        inputs = self.inputs(batch=2)
        inputs["root_position"] += torch.tensor([[2, 3, 4], [7, 8, 9]], dtype=torch.float64)
        before = robot_frame_context(**inputs)
        offset = torch.tensor([[50, -25, 3], [-17, 11, -8]], dtype=torch.float64)
        inputs["support_points_world"] += offset[:, None]
        inputs["workspace_boxes_world"] += offset[:, None, None]
        inputs["root_position"] += offset
        self.assertTrue(torch.equal(before, robot_frame_context(**inputs)))

    def test_inverse_yaw_rotation_matches_analytic_coordinates(self):
        inputs = self.inputs()
        expected = robot_frame_context(**inputs).reshape(1, -1, 3)
        inputs["root_quaternion_xyzw"] = torch.tensor([[0, 0, math.sqrt(0.5), math.sqrt(0.5)]], dtype=torch.float64)
        actual = robot_frame_context(**inputs).reshape(1, -1, 3)
        # Robot yaw +90 degrees: world (x,y,z) becomes robot (y,-x,z).
        expected = torch.stack((expected[:, :, 1], -expected[:, :, 0], expected[:, :, 2]), dim=-1)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_nonunit_quaternions_are_normalized_and_inputs_unchanged(self):
        inputs = self.inputs(batch=3)
        inputs["root_quaternion_xyzw"] = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64).repeat(3, 1)
        expected = robot_frame_context(**inputs)
        inputs["root_quaternion_xyzw"] *= torch.tensor([[7], [-3], [1e200]], dtype=torch.float64)
        originals = {name: value.clone() for name, value in inputs.items()}
        self.assertTrue(torch.allclose(robot_frame_context(**inputs), expected, rtol=1e-12, atol=1e-12))
        for name, value in inputs.items():
            self.assertTrue(torch.equal(value, originals[name]), name)

    def test_robot_translation_changes_relative_features(self):
        inputs = self.inputs()
        before = robot_frame_context(**inputs).reshape(1, -1, 3)
        inputs["root_position"] += torch.tensor([[1, 2, 3]], dtype=torch.float64)
        after = robot_frame_context(**inputs).reshape(1, -1, 3)
        self.assertTrue(torch.equal(before - after, torch.tensor([1, 2, 3], dtype=torch.float64).expand_as(before)))

    def test_invalid_input_contracts(self):
        malformed = {
            "support_points_world": torch.zeros(1, 1, 3, dtype=torch.float64),
            "workspace_boxes_world": torch.zeros(1, 1, 3, dtype=torch.float64),
            "root_position": torch.zeros(1, 1, 3, dtype=torch.float64),
            "root_quaternion_xyzw": torch.zeros(1, 3, dtype=torch.float64),
        }
        for name, invalid in malformed.items():
            with self.subTest(name=name):
                inputs = self.inputs()
                inputs[name] = invalid
                with self.assertRaises(ValueError):
                    robot_frame_context(**inputs)
        for name in self.inputs():
            for invalid_kind in ("nan", "dtype"):
                with self.subTest(name=name, invalid_kind=invalid_kind):
                    inputs = self.inputs()
                    if invalid_kind == "nan":
                        inputs[name].flatten()[0] = float("nan")
                    else:
                        inputs[name] = inputs[name].to(torch.float32)
                    with self.assertRaises(ValueError):
                        robot_frame_context(**inputs)
        for invalid_quaternion in (torch.zeros(1, 4, dtype=torch.float64),
                                   torch.full((1, 4), 1e-320, dtype=torch.float64)):
            inputs = self.inputs()
            inputs["root_quaternion_xyzw"] = invalid_quaternion
            with self.assertRaises(ValueError):
                robot_frame_context(**inputs)
        inputs = self.inputs()
        inputs["workspace_boxes_world"][:, :, 1, 0] = inputs["workspace_boxes_world"][:, :, 0, 0]
        with self.assertRaises(ValueError):
            robot_frame_context(**inputs)

    def test_empty_environment_batch(self):
        self.assertEqual(robot_frame_context(**self.inputs(batch=0)).shape, (0, 30))


class WorkspaceOverlapTests(unittest.TestCase):
    def boxes(self, n=1):
        return torch.tensor([[[[0, 0, 0], [1, 1, 1]]]], dtype=torch.float64).repeat(n, 1, 1, 1)

    def test_inside_outside_and_face_tangency(self):
        centres = torch.tensor([[[0.5, 0.5, 0.5]], [[2, 0.5, 0.5]], [[2.01, 0.5, 0.5]]], dtype=torch.float64)
        self.assertEqual(robot_sphere_box_overlap(centres, 1.0, self.boxes(3)).tolist(), [True, True, False])

    def test_corner_distance_uses_sphere_not_expanded_box(self):
        centres = torch.tensor([[[1.8, 1.8, 0.5]], [[1.6, 1.6, 0.5]]], dtype=torch.float64)
        self.assertEqual(robot_sphere_box_overlap(centres, 1.0, self.boxes(2)).tolist(), [False, True])

    def test_any_sphere_any_box_and_individual_radii(self):
        centres = torch.tensor([[[10, 10, 10], [3, 3, 3]], [[10, 10, 10], [5, 5, 5]]], dtype=torch.float64)
        boxes = torch.cat((self.boxes(2), self.boxes(2) + 2), dim=1)
        self.assertEqual(robot_sphere_box_overlap(centres, [0.1, 0.0], boxes).tolist(), [True, False])

    def test_common_translation_does_not_change_overlap(self):
        centres = torch.tensor([[[1.6, 1.6, 0.5]]], dtype=torch.float64)
        offset = torch.tensor([100, -20, 5], dtype=torch.float64)
        self.assertTrue(torch.equal(robot_sphere_box_overlap(centres, 1.0, self.boxes()),
                                    robot_sphere_box_overlap(centres + offset, 1.0, self.boxes() + offset)))

    def test_invalid_radii_and_geometry_rejected(self):
        centres = torch.zeros(1, 1, 3, dtype=torch.float64)
        for radius in (-1.0, float("nan"), float("inf"), [0.1, 0.2], torch.zeros(1, 1, dtype=torch.float64)):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    robot_sphere_box_overlap(centres, radius, self.boxes())
        for radius in (True, [True], "1", torch.tensor(1)):
            with self.subTest(radius=radius):
                with self.assertRaises(TypeError):
                    robot_sphere_box_overlap(centres, radius, self.boxes())
        for invalid in (self.boxes(2), self.boxes().squeeze(0), torch.zeros_like(self.boxes()),
                        self.boxes().to(torch.float32), self.boxes() * float("nan")):
            with self.assertRaises(ValueError):
                robot_sphere_box_overlap(centres, 1.0, invalid)
        with self.assertRaises(ValueError):
            robot_sphere_box_overlap(torch.zeros(1, 0, 3, dtype=torch.float64), 1.0, self.boxes())

    def test_empty_environment_batch(self):
        actual = robot_sphere_box_overlap(torch.empty(0, 1, 3, dtype=torch.float64), 0.1, self.boxes(0))
        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
