"""CPU integration tests for the real teacher PPO and distillation modules.

Only the simulator and optional W&B logger are substituted. Network forwards,
rollout storage, gradient updates, checkpoint I/O and resume use production code.
"""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

with mock.patch.dict(sys.modules, {"wandb": SimpleNamespace(run=None, log=lambda *a, **k: None)}):
    from rsl_rl.runners.on_policy_runner_mimic_priv import OnPolicyRunnerMimicPriv


class SmallVectorEnv:
    """Tensor-only stand-in for Isaac Gym, including mutable observation buffers."""
    def __init__(self):
        sizes = dict(n_proprio=3, n_teacher_priv=2, n_demo=1, n_scan=0,
                     n_priv_latent=2, n_priv=2, history_len=10, num_policy_actions=2)
        self.cfg = SimpleNamespace(env=SimpleNamespace(**sizes))
        self.num_obs = 43  # feature history + proprio + teacher/demo + priv + history
        self.num_privileged_obs = None
        self.num_envs = 4
        self.dof_names = ["joint_0", "joint_1"]
        self.obs = torch.randn(self.num_envs, self.num_obs)
        self.extra_hist = torch.randn(self.num_envs, 6)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.max_episode_length = 20

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return None

    def get_extra_hist_obs(self):
        return self.extra_hist

    def step(self, actions):
        self.obs.add_(0.01)
        self.extra_hist.add_(0.02)
        self.episode_length_buf += 1
        rewards = 1.0 - actions.square().sum(-1)
        dones = self.episode_length_buf >= self.max_episode_length
        return self.obs, None, rewards, dones, {"extra_hist_obs": self.extra_hist}


def small_config(distill=False, estimated=True):
    config = {
        "runner": dict(policy_class_name="ActorCriticMimicPriv", algorithm_class_name="PPOMimicPriv",
                       num_steps_per_env=2, save_interval=1),
        "policy": dict(actor_hidden_dims=[8], critic_hidden_dims=[8], activation="elu",
                       init_noise_std=0.3, priv_encoder_dims=[4], text_feat_input_dim=3,
                       text_feat_output_dim=3, feat_hist_len=1, tanh_encoder_output=False),
        "algorithm": dict(num_learning_epochs=1, num_mini_batches=2, dagger_update_freq=2,
                          priv_reg_coef_schedual=[0.1, 0.1, 0, 1],
                          grad_penalty_coef_schedule=[0.01, 0.01, 0, 1]),
        "estimator": dict(hidden_dims=[8], learning_rate=0.005, priv_states_dim=2, priv_start=9,
                          prop_dim=3, prop_start=3, train_with_estimated_states=estimated),
        "depth_encoder": dict(if_depth=False),
    }
    if distill:
        config["distill"] = dict(activation="elu", num_demo=1, student_actor_hidden_dims=[8],
                                 num_student_history=2, learning_rate=0.005, num_steps_per_env=2,
                                 num_pretrain_iter=1, num_mini_batches=2)
    return config


def make_runner(directory=None, distill=False, estimated=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return OnPolicyRunnerMimicPriv(SmallVectorEnv(), small_config(distill, estimated),
                                      log_dir=directory, init_wandb=False, device="cpu")


def collect(alg, env, hist):
    for _ in range(alg.storage.num_transitions_per_env):
        obs = env.get_observations()
        with torch.no_grad():
            actions = alg.act(obs, obs, {}, hist_encoding=hist)
            _, _, rewards, dones, infos = env.step(actions)
            alg.process_env_step(rewards, dones, infos)
    with torch.no_grad():
        alg.compute_returns(env.obs)


class TeacherStudentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(91)

    def test_ppo_replays_exact_features_and_mode_when_estimator_changes(self):
        for estimated in (False, True):
            for hist in (False, True):
                with self.subTest(estimated=estimated, hist=hist):
                    runner = make_runner(estimated=estimated)
                    alg = runner.alg
                    initial_obs = runner.env.obs.clone()
                    collect(alg, runner.env, hist)
                    torch.testing.assert_close(alg.storage.observations[0], initial_obs)
                    if not estimated:
                        torch.testing.assert_close(alg.storage.actor_observations, alg.storage.observations)
                    # This deliberately corrupts current predictions. PPO must
                    # still replay the original input distribution and ratio=1.
                    with torch.no_grad():
                        for parameter in alg.estimator.parameters():
                            parameter.add_(0.2)
                    for group in alg.optimizer.param_groups:
                        group['lr'] = 0.0
                    original_act = alg.actor_critic.act
                    seen = []
                    def check_act(obs, hist_encoding=False, **kwargs):
                        seen.append((obs.detach().clone(), hist_encoding))
                        return original_act(obs, hist_encoding=hist_encoding, **kwargs)
                    with mock.patch.object(alg.actor_critic, 'act', side_effect=check_act):
                        losses = alg.update()
                    self.assertEqual(len(seen), 2)
                    self.assertTrue(all(mode is hist for _, mode in seen))
                    all_inputs = alg.storage.actor_observations.flatten(0, 1)
                    for inputs, _ in seen:
                        self.assertTrue(all(any(torch.equal(row, saved) for saved in all_inputs) for row in inputs))
                    # With actor and critic fixed, normalized advantages sum to
                    # zero, so the unclipped on-policy surrogate averages to zero.
                    self.assertAlmostEqual(losses[1], 0.0, places=5)
                    self.assertEqual(alg.counter, 1)

    def test_history_mode_change_inside_rollout_fails(self):
        runner = make_runner()
        with torch.no_grad():
            runner.alg.act(runner.env.obs, runner.env.obs, {}, hist_encoding=True)
            runner.alg.process_env_step(torch.ones(4), torch.zeros(4), {})
            with self.assertRaisesRegex(ValueError, 'constant'):
                runner.alg.act(runner.env.obs, runner.env.obs, {}, hist_encoding=False)

    def test_teacher_checkpoint_restores_all_optimizer_states_and_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.learn_RL(2)
            self.assertEqual(runner.current_learning_iteration, 2)
            self.assertEqual(runner.alg.counter, 2)  # dagger must not double-count
            self.assertFalse((Path(directory) / 'model_0.pt').exists())
            path = Path(directory) / 'model_2.pt'
            self.assertTrue(path.is_file())
            resumed = make_runner()
            resumed.load(path)
            self.assertEqual(resumed.current_learning_iteration, 2)
            self.assertEqual(resumed.alg.counter, 2)
            for name in ('optimizer', 'estimator_optimizer', 'hist_encoder_optimizer'):
                old = getattr(runner.alg, name).state_dict()['state']
                new = getattr(resumed.alg, name).state_dict()['state']
                self.assertTrue(old)
                self.assertEqual(old.keys(), new.keys())
                for index in old:
                    for key in old[index]:
                        torch.testing.assert_close(old[index][key], new[index][key])

    def test_student_initialization_training_final_save_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = make_runner()
            teacher.current_learning_iteration = 9
            teacher.alg.counter = 12
            teacher_path = Path(directory) / 'teacher.pt'
            teacher.save(teacher_path)
            runner = make_runner(directory, distill=True)
            with self.assertRaisesRegex(RuntimeError, 'load_teacher'):
                runner.learn_distill(1)
            with self.assertRaisesRegex(ValueError, 'load_teacher'):
                runner.load(teacher_path)
            runner.load_teacher(teacher_path)
            runner.save_interval = 100  # final checkpoint must be saved off cadence
            self.assertEqual(runner.current_learning_iteration, 0)
            before_teacher = copy.deepcopy(runner.alg.actor_critic.state_dict())
            before_student = copy.deepcopy(runner.alg.student_actor.student_actor_backbone.state_dict())
            with contextlib.redirect_stdout(io.StringIO()):
                runner.learn_distill(2)
            self.assertEqual(runner.current_learning_iteration, 2)
            self.assertEqual(runner.alg.counter, 2)
            for name, tensor in before_teacher.items():
                torch.testing.assert_close(tensor, runner.alg.actor_critic.state_dict()[name])
            self.assertTrue(any(not torch.equal(tensor, runner.alg.student_actor.student_actor_backbone.state_dict()[name])
                                for name, tensor in before_student.items()))
            self.assertFalse((Path(directory) / 'model_0.pt').exists())
            resumed = make_runner(directory, distill=True)
            resumed.load(Path(directory) / 'model_2.pt')
            self.assertEqual(resumed.current_learning_iteration, 2)
            self.assertTrue(resumed.alg.student_actor_optimizer.state_dict()['state'])
            for name, tensor in runner.alg.student_actor.state_dict().items():
                torch.testing.assert_close(tensor, resumed.alg.student_actor.state_dict()[name])
            with contextlib.redirect_stdout(io.StringIO()):
                resumed.learn_distill(1)
            self.assertTrue((Path(directory) / 'model_3.pt').is_file())
            self.assertEqual(resumed.current_learning_iteration, 3)

    def test_student_resume_rejects_missing_optimizer_and_incompatible_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(distill=True)
            path = Path(directory) / 'student.pt'
            runner.save(path)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint['student_actor_optimizer_state_dict']
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'student_actor_optimizer_state_dict'):
                runner.load(path)
            del checkpoint['student_actor_state_dict']['student_actor_backbone.0.weight']
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                runner.load(path, load_optimizer=False)

    def test_resume_validates_pipeline_metadata_but_teacher_initialization_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = make_runner()
            teacher.pipeline_metadata = {'task': 'teacher', 'reward': 'v1'}
            path = Path(directory) / 'teacher.pt'
            teacher.save(path)
            resumed = make_runner()
            resumed.pipeline_metadata = {'task': 'teacher', 'reward': 'v2'}
            with self.assertRaisesRegex(ValueError, 'pipeline_metadata'):
                resumed.load(path)
            resumed.pipeline_metadata = teacher.pipeline_metadata.copy()
            resumed.load(path)
            student = make_runner(distill=True)
            student.pipeline_metadata = {'task': 'student', 'reward': 'v1'}
            student.load_teacher(path)
            self.assertTrue(student._teacher_initialized)

    def test_standard_deviation_remains_positive_without_changing_state_keys(self):
        runner = make_runner()
        with torch.no_grad():
            runner.alg.actor_critic.std.fill_(-1)
            runner.alg.act(runner.env.obs, runner.env.obs, {})
        self.assertTrue((runner.alg.actor_critic.action_std > 0).all())
        self.assertIn('std', runner.alg.actor_critic.state_dict())
        self.assertNotIn('log_std', runner.alg.actor_critic.state_dict())

    def test_legacy_storage_tuple_is_unchanged(self):
        runner = make_runner()
        collect(runner.alg, runner.env, False)
        self.assertEqual(len(next(runner.alg.storage.mini_batch_generator(1, 1))), 11)
        self.assertEqual(len(next(runner.alg.storage.mini_batch_generator(1, 1, True))), 12)


if __name__ == '__main__':
    unittest.main()
