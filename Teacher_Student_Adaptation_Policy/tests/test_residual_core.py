"""CPU regression tests for residual control and timeout-safe PPO.

Run from the repository root with:
    PYTHONPATH=rsl_rl python -m unittest discover -s tests -p test_residual_core.py
"""

import copy
import math
import unittest

import torch
from torch import nn

from rsl_rl.residual import ResidualActorCritic, ResidualPPO


class DummyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.dropout = nn.Dropout(0.9)
        self.saw_history_encoding = False

    def forward(self, full_obs, history, hist_encoding=False):
        self.saw_history_encoding = hist_encoding
        return self.dropout(self.linear(full_obs[:, :3] + history[:, :3]))


def make_policy(**overrides):
    options = dict(
        student=DummyStudent(),
        residual_obs_dim=4,
        num_actions=3,
        residual_bounds=[0.1, 0.2, 0.3],
        action_scale=[0.5, 0.25, 0.1],
        actor_hidden_dims=(12, 8),
        critic_hidden_dims=(12, 8),
    )
    options.update(overrides)
    return ResidualActorCritic(**options)


def collect_rollout(policy, horizon=3, num_envs=5):
    fields = {key: [] for key in (
        "residual_obs", "latent_actions", "old_log_prob", "values", "rewards", "dones", "old_mean", "old_std"
    )}
    for step in range(horizon):
        residual_obs = torch.randn(num_envs, 4)
        residual_obs[:, 0] = torch.arange(step * num_envs, (step + 1) * num_envs)
        result = policy.act(torch.randn(num_envs, 8), torch.randn(num_envs, 6), residual_obs)
        fields["residual_obs"].append(residual_obs)
        for target, source in (
            ("latent_actions", "latent_actions"), ("old_log_prob", "log_prob"),
            ("values", "value"), ("old_mean", "mean"), ("old_std", "std")
        ):
            fields[target].append(result[source])
        fields["rewards"].append(torch.linspace(-0.5, 1.5, num_envs) + step)
        fields["dones"].append(torch.zeros(num_envs, dtype=torch.bool))
    return {name: torch.stack(values) for name, values in fields.items()}


class ResidualPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.policy = make_policy()
        self.obs = torch.randn(7, 8)
        self.history = torch.randn(7, 6)
        self.features = torch.randn(7, 4)

    def test_zero_initial_mean_exactly_preserves_deterministic_student(self):
        self.policy.train()
        baseline = self.policy.student(self.obs, self.history, hist_encoding=True)
        first = self.policy.act_inference(self.obs, self.history, self.features)
        second = self.policy.act_inference(self.obs, self.history, self.features)
        self.assertTrue(torch.equal(first, baseline))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(self.policy.student.training)
        self.assertFalse(self.policy.student.dropout.training)
        self.assertTrue(self.policy.student.saw_history_encoding)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.policy.student.parameters()))

    def test_action_corrections_are_radian_bounded_and_scaled_per_joint(self):
        with torch.no_grad():
            self.policy.actor[-1].bias.copy_(torch.tensor([100.0, -100.0, 0.5]))
        result = self.policy.act(self.obs, self.history, self.features)
        baseline = self.policy.student(self.obs, self.history, hist_encoding=True)
        torch.testing.assert_close(
            (result["actions"] - baseline) * self.policy.action_scale,
            result["residual_rad"],
        )
        self.assertTrue((result["residual_rad"].abs() <= self.policy.residual_bounds).all())
        torch.testing.assert_close(
            result["residual_rad"], self.policy.residual_bounds * result["latent_actions"].tanh()
        )

    def test_stored_latent_has_exactly_reproducible_log_probability(self):
        result = self.policy.act(self.obs, self.history, self.features)
        evaluated = self.policy.evaluate_latent(self.features, result["latent_actions"])
        torch.testing.assert_close(evaluated["log_prob"], result["log_prob"])
        torch.testing.assert_close(evaluated["std"], torch.full((7, 3), 0.1))
        self.assertEqual(evaluated["value"].shape, (7,))
        self.assertEqual(evaluated["entropy"].shape, (7,))
        self.assertFalse(result["actions"].requires_grad)
        self.assertTrue(evaluated["log_prob"].requires_grad)

    def test_residual_does_not_observe_full_student_input(self):
        with torch.no_grad():
            self.policy.actor[-1].weight.fill_(0.2)
        first = self.policy.act_inference(self.obs, self.history, self.features)
        first_base = self.policy.student(self.obs, self.history, hist_encoding=True)
        changed = self.obs + 5.0
        second = self.policy.act_inference(changed, self.history, self.features)
        second_base = self.policy.student(changed, self.history, hist_encoding=True)
        torch.testing.assert_close(first - first_base, second - second_base)

    def test_parameters_and_checkpoint_roundtrip(self):
        student_ids = {id(parameter) for parameter in self.policy.student.parameters()}
        self.assertTrue(all(id(parameter) not in student_ids for parameter in self.policy.trainable_parameters()))
        restored = ResidualActorCritic(DummyStudent(), **self.policy.export_config())
        restored.load_state_dict(self.policy.state_dict())
        torch.testing.assert_close(
            restored.act_inference(self.obs, self.history, self.features),
            self.policy.act_inference(self.obs, self.history, self.features),
        )

    def test_invalid_configuration_and_inputs_fail_explicitly(self):
        for kwargs in (
            {"residual_bounds": 0}, {"residual_bounds": [0.1, 0.2]},
            {"action_scale": -1}, {"action_scale": float("nan")},
            {"initial_std": 0}, {"initial_std": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_policy(**kwargs)
        bad_features = self.features.clone()
        bad_features[0, 0] = float("nan")
        for features in (bad_features, self.features[:, :3], self.features.double()):
            with self.assertRaises(ValueError):
                self.policy.act(self.obs, self.history, features)
        with self.assertRaises(ValueError):
            self.policy.act(self.obs[:-1], self.history, self.features)
        with self.assertRaises(ValueError):
            self.policy.evaluate_latent(self.features, torch.zeros(7, 2))

    def test_numerically_degenerate_likelihood_is_rejected(self):
        with torch.no_grad():
            self.policy.log_std.fill_(-100.0)
        with self.assertRaises(FloatingPointError):
            self.policy.evaluate_latent(self.features, torch.zeros(7, 3))


class ResidualPPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.policy = make_policy()
        self.algorithm = ResidualPPO(
            self.policy, num_learning_epochs=2, num_mini_batches=4, schedule="fixed"
        )

    def test_gae_distinguishes_terminal_timeout_and_continuing_transitions(self):
        algorithm = ResidualPPO(self.policy, gamma=0.9, lam=1.0)
        rollout = {
            "rewards": torch.tensor([[1.0], [2.0], [3.0]]),
            "values": torch.tensor([[0.5], [0.6], [0.7]]),
            "dones": torch.tensor([[False], [True], [False]]),
            "time_outs": torch.tensor([[False], [True], [False]]),
            "terminal_values": torch.tensor([[0.0], [5.0], [0.0]]),
        }
        returns, advantages = algorithm.compute_returns(rollout, torch.tensor([4.0]))
        torch.testing.assert_close(returns, torch.tensor([[6.85], [6.5], [6.6]]))
        torch.testing.assert_close(advantages, returns - rollout["values"])
        rollout["time_outs"].zero_()
        returns, _ = algorithm.compute_returns(rollout, torch.tensor([4.0]))
        torch.testing.assert_close(returns, torch.tensor([[2.8], [2.0], [6.6]]))

    def test_timeout_requires_actual_terminal_values(self):
        rollout = {
            "rewards": torch.ones(1, 1), "values": torch.zeros(1, 1),
            "dones": torch.ones(1, 1), "time_outs": torch.ones(1, 1),
        }
        with self.assertRaisesRegex(ValueError, "pre-reset terminal"):
            self.algorithm.compute_returns(rollout, torch.zeros(1))
        rollout["dones"].zero_()
        with self.assertRaisesRegex(ValueError, "also be marked done"):
            self.algorithm.compute_returns(rollout, torch.zeros(1))

    def test_update_freezes_student_and_visits_every_sample_each_epoch(self):
        rollout = collect_rollout(self.policy)
        before_student = copy.deepcopy(self.policy.student.state_dict())
        before_actor = self.policy.actor[-1].weight.detach().clone()
        before_critic = self.policy.critic[-1].weight.detach().clone()
        visited = []
        original_evaluate = self.policy.evaluate_latent

        def record_evaluate(features, latent_actions):
            visited.extend(features[:, 0].tolist())
            return original_evaluate(features, latent_actions)

        self.policy.evaluate_latent = record_evaluate
        metrics = self.algorithm.update(rollout, torch.zeros(5))
        self.assertEqual(metrics["samples"], 30)
        self.assertEqual(metrics["updates"], 8)
        self.assertEqual(sorted(visited[:15]), list(range(15)))
        self.assertEqual(sorted(visited[15:]), list(range(15)))
        self.assertNotEqual(visited[:15], visited[15:])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        for name, value in self.policy.student.state_dict().items():
            self.assertTrue(torch.equal(value, before_student[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.student.parameters()))
        self.assertFalse(self.policy.student.training)
        self.assertFalse(torch.equal(before_actor, self.policy.actor[-1].weight))
        self.assertFalse(torch.equal(before_critic, self.policy.critic[-1].weight))
        student_ids = {id(parameter) for parameter in self.policy.student.parameters()}
        self.assertTrue(all(id(parameter) not in student_ids for group in self.algorithm.optimizer.param_groups for parameter in group["params"]))

    def test_single_transition_does_not_produce_nan_normalized_advantages(self):
        rollout = collect_rollout(self.policy, horizon=1, num_envs=1)
        # Exercise the sampled KL fallback as well as the exact-KL path above.
        del rollout["old_mean"]
        del rollout["old_std"]
        metrics = self.algorithm.update(rollout, torch.zeros(1))
        self.assertEqual(metrics["updates"], 2)
        self.assertEqual(metrics["samples"], 2)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_nonfinite_rollout_rejected_before_parameters_change(self):
        rollout = collect_rollout(self.policy)
        rollout["residual_obs"][1, 2, 3] = float("inf")
        before = copy.deepcopy(self.policy.state_dict())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.algorithm.update(rollout, torch.zeros(5))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in self.policy.state_dict().items()))

    def test_scalar_rollout_trailing_dimensions_are_accepted(self):
        rollout = collect_rollout(self.policy, horizon=1, num_envs=2)
        for name in ("values", "rewards", "dones", "old_log_prob"):
            rollout[name] = rollout[name].unsqueeze(-1)
        metrics = self.algorithm.update(rollout, torch.zeros(2, 1))
        self.assertEqual(metrics["samples"], 4)


if __name__ == "__main__":
    unittest.main()
