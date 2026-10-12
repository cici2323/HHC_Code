"""Validate the unified CLI before allocating a simulator."""

from contextlib import redirect_stderr
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_pipeline_cli", ROOT / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.checkpoint = self.path / "model.pt"
        self.checkpoint.touch()

    def parse(self, *arguments):
        return pipeline.parse_args([*arguments, "--output-dir", str(self.path / "run")])

    def rejected(self, *arguments):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.parse(*arguments)

    def test_teacher_defaults_and_local_checkout(self):
        args = self.parse("teacher")
        self.assertEqual(args.num_envs, 512)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.iterations, 50000)
        self.assertEqual(pipeline.ROOT, ROOT)

    def test_stage_handoffs_are_explicit(self):
        self.rejected("student")
        self.rejected("adaptation")
        self.rejected("teacher", "--student-checkpoint", str(self.checkpoint))
        self.assertEqual(self.parse("student", "--teacher-checkpoint", str(self.checkpoint)).stage, "student")

    def test_construction_requires_both_data_and_geometry(self):
        self.rejected("adaptation", "--student-checkpoint", str(self.checkpoint))
        args = self.parse("adaptation", "--student-checkpoint", str(self.checkpoint), "--motion-only")
        self.assertTrue(args.motion_only)
        self.rejected("adaptation", "--student-checkpoint", str(self.checkpoint),
                      "--motion-only", "--construction-data", str(self.checkpoint))
        args = self.parse("adaptation", "--student-checkpoint", str(self.checkpoint),
                          "--construction-data", str(self.checkpoint), "--robot-geometry", str(self.checkpoint))
        self.assertFalse(args.motion_only)

    def test_evaluation_and_positive_counts(self):
        self.rejected("teacher", "--evaluate")
        args = self.parse("teacher", "--evaluate", "--resume", str(self.checkpoint))
        self.assertEqual(args.num_envs, 1)
        for flag in ("--num-envs", "--iterations", "--save-interval", "--evaluation-steps"):
            self.rejected("teacher", flag, "0")

    def test_paper_reward_config_and_invalid_values(self):
        config = pipeline.load_reward_config(ROOT / "configs" / "rewards.json")
        self.assertEqual(sum(config["motion_weights"].values()), 26)
        self.assertEqual(config["adaptation_tracking_weight"], 0.2)
        self.assertEqual(config["collaboration"]["support_weight"], 5)
        config["collaboration"]["support_scale"] = 0
        path = self.path / "bad.json"
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            pipeline.load_reward_config(path)

    def test_new_run_does_not_overwrite_existing_results(self):
        (self.path / "run").mkdir()
        (self.path / "run" / "model.pt").touch()
        self.rejected("teacher")
        args = self.parse("teacher", "--resume", str(self.checkpoint))
        self.assertEqual(args.resume, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
