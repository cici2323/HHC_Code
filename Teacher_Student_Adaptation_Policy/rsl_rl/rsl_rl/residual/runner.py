"""Simulation-independent runner for a frozen student plus residual PPO.

The environment owns task rewards and construction context. See
docs/residual_adaptation.md for the environment contract.
"""

from dataclasses import asdict
from pathlib import Path
import json
import os
import time

import torch

from .checkpoints import build_student, load_checkpoint, load_student, bounds_from_joint_names
from .observations import StudentObservationLayout, RunningMeanStdNormalizer
from .policy import ResidualActorCritic
from .ppo import ResidualPPO


def default_config():
    return {
        "num_steps_per_env": 24,
        "save_interval": 500,
        "context_dim": 0,
        "normalize_observations": True,
        "policy": {"actor_hidden_dims": [512, 256, 128],
                   "critic_hidden_dims": [512, 256, 128], "initial_std": 0.1},
        "algorithm": {"learning_rate": 1e-3, "clip_param": 0.2,
                      "gamma": 0.99, "lam": 0.95, "num_learning_epochs": 5,
                      "num_mini_batches": 4, "value_loss_coef": 1.0,
                      "entropy_coef": 0.005, "max_grad_norm": 1.0,
                      "use_clipped_value_loss": True, "desired_kl": 0.01,
                      "schedule": "adaptive"},
    }


class ResidualRunner:
    """Requires exactly one student checkpoint or self-contained residual resume.

    Checkpoint resume restores learning state; simulator state is reset, so it
    does not promise bitwise replay of the interrupted trajectory.
    """

    def __init__(self, env, student_spec, layout, joint_names, action_scale,
                 config=None, log_dir=None, device="cpu", student_checkpoint=None,
                 residual_checkpoint=None, environment_metadata=None):
        if (student_checkpoint is None) == (residual_checkpoint is None):
            raise ValueError("Provide exactly one student_checkpoint or residual_checkpoint")
        self.env = env
        self.device = torch.device(device)
        self.config = default_config() if config is None else config
        self.layout = layout
        self.student_spec = student_spec
        self.joint_names = list(joint_names)
        self.action_scale = action_scale
        self.environment_metadata = environment_metadata or {}
        self.context_dim = self.config["context_dim"]
        if not isinstance(self.context_dim, int) or self.context_dim < 0:
            raise ValueError("context_dim must be a nonnegative integer")
        if self.config["num_steps_per_env"] < 1 or self.config["save_interval"] < 1:
            raise ValueError("Rollout length and save interval must be positive")
        if self.context_dim and not callable(getattr(env, "get_residual_context", None)):
            raise ValueError("Construction context requires env.get_residual_context()")
        student = build_student(student_spec).to(self.device)
        if student_checkpoint is not None:
            load_student(student, student_checkpoint, self.device)
        self.policy = ResidualActorCritic(
            student, layout.residual_dim(self.context_dim), len(joint_names),
            bounds_from_joint_names(joint_names), action_scale,
            **self.config["policy"]
        ).to(self.device)
        self.normalizer = RunningMeanStdNormalizer(layout.residual_dim(self.context_dim)).to(self.device)
        self.algorithm = ResidualPPO(self.policy, **self.config["algorithm"])
        self.iteration = 0
        self.total_steps = 0
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if residual_checkpoint is not None:
            self.load(residual_checkpoint)
        self._read_inputs()  # Check the contract before creating a log directory.
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "config.json").open("w") as stream:
                json.dump(self.metadata(), stream, indent=2)

    def metadata(self):
        return {"config": self.config, "student_spec": self.student_spec,
                "layout": asdict(self.layout), "joint_names": self.joint_names,
                "action_scale": self.action_scale,
                "environment_metadata": self.environment_metadata}

    def _read_inputs(self):
        obs = self.env.get_observations().to(self.device)
        history = self.env.get_extra_hist_obs().to(self.device)
        context = None
        if self.context_dim:
            context = self.env.get_residual_context().to(self.device)
            if context.shape != (obs.shape[0], self.context_dim):
                raise ValueError("get_residual_context() returned the wrong shape")
        raw = self.layout.build(obs, history, context)
        return obs, history, raw

    def _normalize(self, observations):
        return self.normalizer(observations) if self.config["normalize_observations"] else observations

    def learn(self, num_learning_iterations):
        if num_learning_iterations < 1:
            raise ValueError("num_learning_iterations must be positive")
        self.policy.train()
        obs, history, raw = self._read_inputs()
        if self.config["normalize_observations"] and self.normalizer.count.item() == 0:
            self.normalizer.update(raw)
        for _ in range(num_learning_iterations):
            start = time.monotonic()
            rollout = {name: [] for name in (
                "residual_obs", "latent_actions", "old_log_prob", "values",
                "rewards", "dones", "time_outs", "terminal_values", "old_mean", "old_std")}
            raw_rollout = []
            residual_magnitude = []
            reward_term_totals = {}
            # Normalization statistics stay fixed through collection and PPO.
            for _step in range(self.config["num_steps_per_env"]):
                with torch.no_grad():
                    normalized = self._normalize(raw)
                    result = self.policy.act(obs, history, normalized)
                    # Copy inputs before step(): vector environments may reuse buffers.
                    raw_rollout.append(raw.detach().clone())
                    rollout["residual_obs"].append(normalized.detach().clone())
                    for key, result_key in (("latent_actions", "latent_actions"),
                                            ("old_log_prob", "log_prob"),
                                            ("values", "value"),
                                            ("old_mean", "mean"), ("old_std", "std")):
                        rollout[key].append(result[result_key].detach().clone())
                    residual_magnitude.append(result["residual_rad"].abs().mean().item())
                    env_device = getattr(self.env, "device", self.device)
                    _obs, _priv, rewards, dones, infos = self.env.step(result["actions"].to(env_device))
                    for name, term in infos.get("reward_terms", {}).items():
                        reward_term_totals[name] = reward_term_totals.get(name, 0.0) + term.float().mean().item()
                    rewards = rewards.to(self.device).reshape(-1)
                    dones = dones.to(self.device).reshape(-1).bool()
                    timeouts = infos.get("time_outs", torch.zeros_like(dones)).to(self.device).reshape(-1).bool()
                    terminal_values = torch.zeros_like(rewards)
                    if timeouts.any():
                        if "terminal_residual_obs" not in infos:
                            raise ValueError(
                                "Timeout bootstrap requires infos['terminal_residual_obs'] "
                                "from BEFORE reset; reset observations cannot be used."
                            )
                        terminal_raw = infos["terminal_residual_obs"].to(self.device)
                        if terminal_raw.shape != raw.shape:
                            raise ValueError("terminal_residual_obs must be [num_envs, residual_obs_dim]")
                        terminal_values[timeouts] = self.policy.value(self._normalize(terminal_raw[timeouts])).reshape(-1)
                    for key, tensor in (("rewards", rewards), ("dones", dones),
                                        ("time_outs", timeouts), ("terminal_values", terminal_values)):
                        rollout[key].append(tensor.detach().clone())
                    obs, history, raw = self._read_inputs()
            with torch.no_grad():
                last_values = self.policy.value(self._normalize(raw)).reshape(-1)
            rollout = {name: torch.stack(values) for name, values in rollout.items()}
            metrics = self.algorithm.update(rollout, last_values)
            if self.config["normalize_observations"]:
                self.normalizer.update(torch.cat(raw_rollout, dim=0))
            self.iteration += 1
            self.total_steps += rollout["rewards"].numel()
            metrics.update(iteration=self.iteration, total_steps=self.total_steps,
                           mean_step_reward=rollout["rewards"].mean().item(),
                           mean_abs_residual_rad=sum(residual_magnitude) / len(residual_magnitude),
                           elapsed_seconds=time.monotonic() - start)
            metrics.update({"reward/" + name: value / self.config["num_steps_per_env"]
                            for name, value in reward_term_totals.items()})
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if self.log_dir is not None:
                with (self.log_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                if self.iteration % self.config["save_interval"] == 0:
                    self.save(self.log_dir / "residual_{}.pt".format(self.iteration))
        if self.log_dir is not None:
            self.save(self.log_dir / "residual_{}.pt".format(self.iteration))

    @torch.no_grad()
    def inference_actions(self):
        self.policy.eval()
        obs, history, raw = self._read_inputs()
        return self.policy.act_inference(obs, history, self._normalize(raw))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"format": "pose_aware_residual_v1", **self.metadata(),
                      "model_state_dict": self.policy.state_dict(),
                      "normalizer_state_dict": self.normalizer.state_dict(),
                      "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                      "iteration": self.iteration, "total_steps": self.total_steps,
                      "torch_rng_state": torch.get_rng_state()}
        if self.device.type == "cuda":
            checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(str(temporary), str(path))

    def load(self, path):
        checkpoint = load_checkpoint(path, self.device)
        if checkpoint.get("format") != "pose_aware_residual_v1":
            raise ValueError("Expected a residual checkpoint, not a teacher/student checkpoint")
        for name, expected in self.metadata().items():
            if checkpoint.get(name) != expected:
                raise ValueError("Residual checkpoint {} does not match this run".format(name))
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.normalizer.load_state_dict(checkpoint["normalizer_state_dict"], strict=True)
        self.algorithm.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.algorithm.learning_rate = self.algorithm.optimizer.param_groups[0]["lr"]
        self.iteration = int(checkpoint["iteration"])
        self.total_steps = int(checkpoint["total_steps"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
