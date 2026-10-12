"""Exercise teacher -> distilled student -> frozen-student residual checkpoints.

This is an integration test of the production learning code on a synthetic
vector environment. It does not validate Isaac Gym physics or construction-task
performance. The teacher and student network fixtures are small real modules.
"""
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_teacher_student_pipeline import SmallVectorEnv, make_runner, small_config
from rsl_rl.residual.checkpoints import load_checkpoint
from rsl_rl.residual.observations import StudentObservationLayout
from rsl_rl.residual.runner import ResidualRunner, default_config


def residual_fixture():
    cfg = small_config(distill=True, estimated=False)
    layout = StudentObservationLayout(
        num_prop=3, num_teacher_priv=2, num_demo=1, num_scan=0,
        num_priv_latent=2, num_priv_explicit=2, num_hist=10,
        text_feat_input_dim=3, num_student_history=2,
    )
    spec = {
        'actor': {
            'num_prop': layout.num_prop, 'num_teacher_priv': layout.num_teacher_priv,
            'num_demo': layout.num_demo, 'num_scan': layout.num_scan,
            'num_actions': 2, 'num_priv_latent': layout.num_priv_latent,
            'num_priv_explicit': layout.num_priv_explicit, 'num_hist': layout.num_hist,
            'scan_encoder_dims': [256, 256, 256],
            **{name: cfg['policy'][name] for name in (
                'text_feat_input_dim', 'text_feat_output_dim', 'feat_hist_len',
                'actor_hidden_dims', 'priv_encoder_dims', 'activation',
                'tanh_encoder_output')},
        },
        'distill': cfg['distill'],
    }
    config = default_config()
    config.update(num_steps_per_env=2, save_interval=100)
    config['policy'].update(actor_hidden_dims=[8], critic_hidden_dims=[8])
    config['algorithm'].update(num_learning_epochs=1, num_mini_batches=2)
    return layout, spec, config


class FullPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_load_teacher_rejects_student_checkpoints_including_legacy_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_runner(distill=True)
            checkpoint_path = Path(directory) / 'student.pt'
            source.save(checkpoint_path)
            destination = make_runner(distill=True)
            before = copy.deepcopy(destination.alg.actor_critic.state_dict())
            for legacy in (False, True):
                with self.subTest(legacy=legacy):
                    if legacy:
                        checkpoint = load_checkpoint(checkpoint_path)
                        checkpoint.pop('stage')
                        torch.save(checkpoint, checkpoint_path)
                    with self.assertRaisesRegex(ValueError, 'teacher'):
                        destination.load_teacher(checkpoint_path)
                    self.assertFalse(destination._teacher_initialized)
                    for name, value in destination.alg.actor_critic.state_dict().items():
                        self.assertTrue(torch.equal(value, before[name]), name)

    def test_production_teacher_student_residual_handoff_and_resume(self):
        torch.manual_seed(147)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            # Train and save the teacher through the actual PPO runner.
            teacher = make_runner(root / 'teacher', estimated=False)
            teacher.save_interval = 100
            teacher.learn(num_learning_iterations=2)
            teacher_path = root / 'teacher' / 'model_2.pt'
            teacher_state = load_checkpoint(teacher_path)
            self.assertEqual(teacher_state['stage'], 'teacher')
            self.assertEqual(teacher_state['algorithm_counter'], 2)

            # Initialize from that produced teacher, then learn and save the student.
            student = make_runner(root / 'student', distill=True, estimated=False)
            student.save_interval = 100
            student.load_teacher(teacher_path)
            for name, value in teacher.alg.actor_critic.actor.state_dict().items():
                self.assertTrue(torch.equal(value, student.alg.student_actor._orig_actor.state_dict()[name]), name)
            initial_student = copy.deepcopy(student.alg.student_actor.student_actor_backbone.state_dict())
            student.learn(num_learning_iterations=2)
            student_path = root / 'student' / 'model_2.pt'
            student_state = load_checkpoint(student_path)
            self.assertEqual(student_state['stage'], 'student')
            self.assertEqual(student_state['iter'], 2)
            self.assertTrue(student_state['student_actor_optimizer_state_dict']['state'])
            self.assertTrue(any(not torch.equal(value, student.alg.student_actor.student_actor_backbone.state_dict()[name])
                                for name, value in initial_student.items()))

            # Consume the unmodified student checkpoint via production loading.
            layout, spec, config = residual_fixture()
            env = SmallVectorEnv()
            residual = ResidualRunner(
                env, spec, layout, ['left_shoulder_pitch_joint', 'right_hip_pitch_joint'],
                0.25, config=config, log_dir=root / 'adaptation',
                student_checkpoint=student_path, device='cpu',
            )
            expected_student = student_state['student_actor_state_dict']
            for name, value in residual.policy.student.state_dict().items():
                self.assertTrue(torch.equal(value, expected_student[name]), name)
            self.assertTrue(all(not parameter.requires_grad for parameter in residual.policy.student.parameters()))
            with torch.no_grad():
                baseline = residual.policy.student(env.get_observations(), env.get_extra_hist_obs(), hist_encoding=True)
            torch.testing.assert_close(residual.inference_actions(), baseline, rtol=0, atol=0)
            before_residual = copy.deepcopy(residual.policy.actor.state_dict())
            residual.learn(1)
            self.assertEqual(residual.iteration, 1)
            self.assertEqual(residual.total_steps, 8)
            self.assertTrue(any(not torch.equal(value, residual.policy.actor.state_dict()[name])
                                for name, value in before_residual.items()))
            for name, value in residual.policy.student.state_dict().items():
                self.assertTrue(torch.equal(value, expected_student[name]), name)

            # The final checkpoint must exist even before the save interval.
            residual_path = root / 'adaptation' / 'residual_1.pt'
            saved_residual = load_checkpoint(residual_path)
            self.assertEqual(saved_residual['format'], 'pose_aware_residual_v1')
            self.assertEqual(saved_residual['iteration'], 1)
            self.assertTrue(saved_residual['optimizer_state_dict']['state'])
            metrics = json.loads((root / 'adaptation' / 'metrics.jsonl').read_text().strip())
            self.assertEqual(metrics['total_steps'], 8)
            self.assertTrue(torch.isfinite(torch.tensor(metrics['mean_step_reward'])))

            # Resume without access to a separate teacher/student checkpoint.
            resumed = ResidualRunner(
                env, spec, layout, residual.joint_names, 0.25, config=config,
                residual_checkpoint=residual_path, log_dir=root / 'resumed', device='cpu',
            )
            self.assertEqual(resumed.iteration, 1)
            torch.testing.assert_close(resumed.inference_actions(), residual.inference_actions(), rtol=0, atol=0)
            resumed.learn(1)
            self.assertEqual(resumed.iteration, 2)
            self.assertEqual(resumed.total_steps, 16)
            self.assertTrue((root / 'resumed' / 'residual_2.pt').is_file())
            for name, value in resumed.policy.student.state_dict().items():
                self.assertTrue(torch.equal(value, expected_student[name]), name)


if __name__ == '__main__':
    unittest.main()
