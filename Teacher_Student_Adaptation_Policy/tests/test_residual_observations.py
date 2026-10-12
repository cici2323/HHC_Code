"""CPU tests for deployable observation isolation and PPO normalization."""

import io
import unittest

import torch

from rsl_rl.residual.observations import (
    RunningMeanStdNormalizer,
    StudentObservationLayout,
    build_residual_observations,
)


def g1_layout():
    return StudentObservationLayout(
        num_prop=78,
        num_teacher_priv=75,
        num_demo=64,
        num_scan=0,
        num_priv_latent=51,
        num_priv_explicit=3,
        num_hist=10,
        text_feat_input_dim=312,
        num_student_history=25,
    )


class StudentObservationTests(unittest.TestCase):
    def setUp(self):
        self.layout = g1_layout()
        self.full = torch.arange(1363, dtype=torch.float32).repeat(2, 1)
        self.history = torch.full((2, 1950), 2000.0)

    def test_g1_dimensions_and_retained_order(self):
        layout = self.layout
        self.assertEqual(layout.num_observations, 1363)
        self.assertEqual(layout.history_dim, 1950)
        self.assertEqual(layout.residual_dim(), 3187)
        self.assertEqual(layout.residual_dim(12), 3199)
        context = torch.full((2, 12), 3000.0)
        result = build_residual_observations(layout, self.full, self.history, context)
        expected = torch.cat(
            (self.full[:, :390], self.full[:, 465:532], self.full[:, 583:], self.history, context),
            dim=1,
        )
        self.assertEqual(result.shape, (2, 3199))
        self.assertTrue(torch.equal(result, expected))
        # Includes all 64 reference dimensions and the explicit-state slots.
        self.assertTrue(torch.equal(result[:, 390:457], self.full[:, 465:532]))

    def test_privileged_changes_do_not_affect_residual_input(self):
        baseline = self.layout.build(self.full, self.history)
        changed = self.full.clone()
        changed[:, 390:465] = float("nan")
        changed[:, 532:583] = float("inf")
        result = self.layout.build(changed, self.history)
        self.assertTrue(torch.equal(baseline, result))
        self.assertTrue(torch.equal(self.full[0], torch.arange(1363, dtype=torch.float32)))

    def test_deployable_nonfinite_values_rejected(self):
        for index in (0, 312, 465, 529, 583, 1362):
            bad = self.full.clone()
            bad[:, index] = float("nan")
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.layout.build(bad, self.history)
        bad_history = self.history.clone()
        bad_history[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.layout.build(self.full, bad_history)
        with self.assertRaises(ValueError):
            self.layout.build(self.full, self.history, torch.full((2, 3), float("nan")))

    def test_shapes_and_types_rejected(self):
        cases = (
            (self.full[:, :-1], self.history, None),
            (self.full[0], self.history, None),
            (self.full, self.history[:, :-1], None),
            (self.full, self.history.view(2, 25, 78), None),
            (self.full, self.history[:1], None),
            (self.full, self.history, torch.zeros(3, 2)),
            (self.full, self.history, torch.zeros(2)),
            (self.full, self.history.double(), None),
        )
        for full, history, context in cases:
            with self.subTest(shape=full.shape, history=history.shape), self.assertRaises(ValueError):
                self.layout.build(full, history, context)
        with self.assertRaises(TypeError):
            self.layout.build(self.full.long(), self.history)

    def test_invalid_layout_and_context_dimension_rejected(self):
        config = self.layout.__dict__.copy()
        for name, value in (("num_scan", 1), ("num_prop", 0), ("num_demo", -1), ("num_hist", 1.5)):
            invalid = dict(config, **{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                StudentObservationLayout(**invalid)
        with self.assertRaises(ValueError):
            self.layout.residual_dim(-1)


class RunningMeanStdTests(unittest.TestCase):
    def test_parallel_population_moments(self):
        normalizer = RunningMeanStdNormalizer(2)
        values = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], requires_grad=True)
        normalizer.update(values[:2])
        normalizer.update(values[2:])
        self.assertTrue(torch.allclose(normalizer.mean, torch.tensor([3.0, 14.0])))
        self.assertTrue(torch.allclose(normalizer.var, torch.tensor([8.0 / 3.0, 32.0 / 3.0])))
        self.assertEqual(normalizer.count.item(), 3.0)
        self.assertFalse(normalizer.mean.requires_grad)
        self.assertFalse(normalizer.var.requires_grad)
        self.assertIsNone(normalizer.mean.grad_fn)
        normalized = normalizer(values)
        self.assertTrue(torch.allclose(normalized.mean(0), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(normalized.var(0, unbiased=False), torch.ones(2)))

    def test_forward_keeps_rollout_moments_fixed_and_clips(self):
        normalizer = RunningMeanStdNormalizer(2)
        normalizer.update(torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
        before = {name: value.clone() for name, value in normalizer.state_dict().items()}
        normalizer.train()
        result = normalizer(torch.tensor([[100.0, -100.0]]))
        self.assertTrue(torch.equal(result, torch.tensor([[10.0, -10.0]])))
        for name, value in normalizer.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertEqual(list(normalizer.parameters()), [])

    def test_state_round_trip(self):
        normalizer = RunningMeanStdNormalizer(3)
        values = torch.tensor([[1.0, 2.0, 4.0], [3.0, 2.0, 8.0]])
        normalizer.update(values)
        checkpoint = io.BytesIO()
        torch.save(normalizer.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = RunningMeanStdNormalizer(3)
        restored.load_state_dict(torch.load(checkpoint))
        self.assertTrue(torch.equal(normalizer(values), restored(values)))
        restored.update(values)
        self.assertTrue(torch.equal(restored.mean, normalizer.mean))
        self.assertTrue(torch.equal(restored.var, normalizer.var))
        self.assertEqual(restored.count.item(), 4.0)

    def test_invalid_normalizer_inputs(self):
        normalizer = RunningMeanStdNormalizer(2)
        for batch in (torch.zeros(1, 3), torch.zeros(2), torch.full((1, 2), float("nan"))):
            with self.subTest(shape=batch.shape), self.assertRaises(ValueError):
                normalizer.update(batch)
            with self.assertRaises(ValueError):
                normalizer(batch)
        with self.assertRaises(ValueError):
            normalizer.update(torch.zeros(0, 2))
        with self.assertRaises(TypeError):
            normalizer(torch.zeros(1, 2, dtype=torch.long))
        for dimension in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                RunningMeanStdNormalizer(dimension)


if __name__ == "__main__":
    unittest.main()
