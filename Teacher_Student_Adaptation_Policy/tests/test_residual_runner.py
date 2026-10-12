"""Exercise the REAL legacy student with a tiny deterministic vector environment."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import torch

from rsl_rl.residual.checkpoints import build_student, bounds_from_joint_names, load_student
from rsl_rl.residual.observations import StudentObservationLayout
from rsl_rl.residual.runner import ResidualRunner, default_config


def fixture():
    layout = StudentObservationLayout(3, 2, 2, 0, 2, 1, 10, 12, 5)
    spec = {"actor": {
        "num_prop": 3, "num_teacher_priv": 2, "num_demo": 2,
        "text_feat_input_dim": 12, "text_feat_output_dim": 4, "feat_hist_len": 4,
        "num_scan": 0, "num_actions": 2, "scan_encoder_dims": [],
        "actor_hidden_dims": [8], "priv_encoder_dims": [4],
        "num_priv_latent": 2, "num_priv_explicit": 1, "num_hist": 10,
        "activation": "elu", "tanh_encoder_output": False,
    }, "distill": {"activation": "elu", "num_demo": 2,
                   "student_actor_hidden_dims": [8], "num_student_history": 5}}
    config = default_config()
    config["num_steps_per_env"] = 3
    config["save_interval"] = 1
    config["policy"]["actor_hidden_dims"] = [8]
    config["policy"]["critic_hidden_dims"] = [8]
    config["algorithm"].update(num_learning_epochs=2, num_mini_batches=2)
    return layout, spec, config


class TinyEnvironment:
    device = "cpu"

    def __init__(self, layout, timeout=False, provide_terminal=True):
        self.num_envs = 2
        self.layout = layout
        self.obs = torch.zeros(2, layout.num_observations)
        self.history = torch.zeros(2, layout.history_dim)
        self.timeout = timeout
        self.provide_terminal = provide_terminal

    def get_observations(self):
        return self.obs

    def get_extra_hist_obs(self):
        return self.history

    def step(self, actions):
        self.obs.add_(0.01)  # Intentionally mutate buffers in place.
        self.history.add_(0.01)
        rewards = 1.0 - actions.square().sum(-1)
        dones = torch.tensor([self.timeout, False])
        infos = {"time_outs": dones.clone()}
        if self.timeout and self.provide_terminal:
            infos["terminal_residual_obs"] = self.layout.build(self.obs, self.history).clone()
        if self.timeout:
            self.obs[0] = 0
            self.history[0] = 0
        return self.obs, None, rewards, dones, infos


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.layout, self.spec, self.config = fixture()
        self.student_path = self.root / "student.pt"
        torch.save({"student_actor_state_dict": build_student(self.spec).state_dict()}, self.student_path)

    def runner(self, env=None, **kwargs):
        return ResidualRunner(
            env or TinyEnvironment(self.layout), self.spec, self.layout,
            ["left_shoulder_pitch_joint", "right_hip_pitch_joint"], 0.25,
            config=self.config, **kwargs,
        )

    def test_real_student_update_and_checkpoint_resume(self):
        runner = self.runner(student_checkpoint=self.student_path, log_dir=self.root / "run")
        frozen = {name: value.clone() for name, value in runner.policy.student.state_dict().items()}
        with redirect_stdout(StringIO()):
            runner.learn(2)
        for name, value in runner.policy.student.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        checkpoint = self.root / "run" / "residual_2.pt"
        resumed = self.runner(residual_checkpoint=checkpoint, env=runner.env)
        torch.testing.assert_close(runner.inference_actions(), resumed.inference_actions(), rtol=0, atol=0)
        self.assertEqual(resumed.iteration, 2)
        self.assertEqual(resumed.total_steps, 12)
        self.assertEqual(resumed.algorithm.learning_rate, runner.algorithm.learning_rate)
        self.assertTrue(resumed.algorithm.optimizer.state)
        torch.testing.assert_close(resumed.normalizer.mean, runner.normalizer.mean)
        with redirect_stdout(StringIO()):
            resumed.learn(1)
        self.assertEqual(resumed.iteration, 3)

    def test_teacher_only_checkpoint_is_rejected(self):
        path = self.root / "teacher.pt"
        torch.save({"model_state_dict": {}}, path)
        with self.assertRaisesRegex(ValueError, "distilled student"):
            load_student(build_student(self.spec), path)

    def test_incompatible_student_is_rejected(self):
        path = self.root / "bad_student.pt"
        torch.save({"student_actor_state_dict": {}}, path)
        with self.assertRaises(RuntimeError):
            self.runner(student_checkpoint=path)

    def test_real_student_ignores_privileged_fields(self):
        student = build_student(self.spec)
        obs = torch.randn(2, self.layout.num_observations)
        history = torch.randn(2, self.layout.history_dim)
        changed = obs.clone()
        changed[:, 15:17] = 1000  # Feature12 + proprio3 -> teacher-only2.
        changed[:, 20:22] = -1000  # Demo2 + explicit1 -> privileged latent2.
        torch.testing.assert_close(student(obs, history, True), student(changed, history, True), rtol=0, atol=0)

    def test_timeout_requires_terminal_observation(self):
        env = TinyEnvironment(self.layout, timeout=True, provide_terminal=False)
        runner = self.runner(env=env, student_checkpoint=self.student_path)
        with self.assertRaisesRegex(ValueError, "BEFORE reset"):
            runner.learn(1)

    def test_timeout_with_terminal_observation_trains(self):
        env = TinyEnvironment(self.layout, timeout=True)
        runner = self.runner(env=env, student_checkpoint=self.student_path)
        with redirect_stdout(StringIO()):
            runner.learn(1)
        self.assertEqual(runner.iteration, 1)

    def test_missing_construction_context_fails(self):
        self.config["context_dim"] = 6
        with self.assertRaisesRegex(ValueError, "get_residual_context"):
            self.runner(student_checkpoint=self.student_path)

    def test_joint_bounds_fail_closed(self):
        self.assertEqual(bounds_from_joint_names(["left_shoulder_joint", "right_ankle_joint"]), [0.35, 0.02])
        with self.assertRaises(ValueError):
            bounds_from_joint_names(["unknown_joint"])

    def test_resume_rejects_changed_action_scale(self):
        runner = self.runner(student_checkpoint=self.student_path)
        path = self.root / "residual.pt"
        runner.save(path)
        with self.assertRaisesRegex(ValueError, "action_scale"):
            ResidualRunner(runner.env, self.spec, self.layout, runner.joint_names,
                           0.5, config=self.config, residual_checkpoint=path)


if __name__ == "__main__":
    unittest.main()
