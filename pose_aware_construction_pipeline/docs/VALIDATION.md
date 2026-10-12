# Validation record — October 10, 2026

- Runtime: local macOS, Python 3.11.7, PyTorch 2.10.0, CPU.
- Command: `PYTHONPATH=rsl_rl python -m unittest discover -s tests -v`.
- Result: **101 tests passed**.
- Python 3.8 syntax parsing: 27 new/changed implementation and test files passed.
- Reward configuration and both data schemas parsed as valid JSON.
- Reference teacher SHA-256 matches the supplied original file.
- Main README and pipeline/reward/data/review documentation links resolve locally.

Coverage includes analytical paper reward values, configuration overrides,
single-dt scaling, unchanged regularization, occupied-volume geometry,
robot-relative context, actor privilege exclusion, frozen student invariance,
latent likelihood replay, GAE, checkpoint integrity and resume, stage argument
validation, and the complete teacher → student → residual checkpoint handoff.
Adapter tests execute the actual adapter with simulator dependencies stubbed.

The suite uses synthetic test observations and tiny randomly initialized test
networks. It does not produce trained research checkpoints, validate Isaac Gym
physics, establish training convergence, or reproduce the paper's reported
simulated/physical performance. Construction data and calibrated geometry are
still required for actual construction adaptation.
