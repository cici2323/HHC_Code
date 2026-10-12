"""A bounded residual controller on top of a deterministic frozen student.

The student returns the environment's normalized joint actions. Residual bounds
are in radians, so their correction is divided by the environment action scale
before addition. The stochastic variable used by PPO is the *unsquashed* normal
sample. Its log probability is valid for the PPO importance ratio because the
fixed tanh transform and scaling have a parameter-independent Jacobian, which
cancels between old and new policies.

Only ``residual_obs`` enters the trainable actor and critic. Callers must select
deployable features and apply any observation normalization consistently during
rollout collection and PPO updates.
"""

import math

import torch
from torch import nn
from torch.distributions import Normal


def _positive_vector(value, size, name):
    result = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if result.ndim == 0:
        result = result.repeat(size)
    if result.shape != (size,):
        raise ValueError("{} must be scalar or have shape ({},)".format(name, size))
    if not torch.isfinite(result).all() or not (result > 0).all():
        raise ValueError("{} must contain finite, strictly positive values".format(name))
    return result


def _mlp(input_size, hidden_dims, output_size):
    layers = []
    previous = input_size
    for width in hidden_dims:
        if not isinstance(width, int) or width <= 0:
            raise ValueError("MLP hidden dimensions must be positive integers")
        layers.extend([nn.Linear(previous, width), nn.ELU()])
        previous = width
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class ResidualActorCritic(nn.Module):
    """Frozen ``ActorDistillPriv`` plus an independently trained residual MLP.

    ``student`` must implement ``student(full_obs, history, hist_encoding=True)``
    and return a deterministic action tensor. Its parameters and state are saved
    as part of this module, but its parameters never enter the PPO optimizer.
    ``residual_bounds`` and ``action_scale`` may be scalars or per-joint vectors.
    """

    def __init__(
        self,
        student,
        residual_obs_dim,
        num_actions,
        residual_bounds,
        action_scale,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        initial_std=0.1,
    ):
        super().__init__()
        if not isinstance(student, nn.Module):
            raise TypeError("student must be a torch.nn.Module")
        if not isinstance(residual_obs_dim, int) or residual_obs_dim <= 0:
            raise ValueError("residual_obs_dim must be a positive integer")
        if not isinstance(num_actions, int) or num_actions <= 0:
            raise ValueError("num_actions must be a positive integer")
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("initial_std must be finite and positive")
        self.student = student
        self.student.requires_grad_(False)
        self.student.eval()
        self.residual_obs_dim = residual_obs_dim
        self.num_actions = num_actions
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.initial_std = float(initial_std)
        self.actor = _mlp(residual_obs_dim, self.actor_hidden_dims, num_actions)
        self.critic = _mlp(residual_obs_dim, self.critic_hidden_dims, 1)
        # At initialization the deterministic composite exactly equals the student.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(initial_std)))
        self.register_buffer(
            "residual_bounds", _positive_vector(residual_bounds, num_actions, "residual_bounds")
        )
        self.register_buffer(
            "action_scale", _positive_vector(action_scale, num_actions, "action_scale")
        )

    def train(self, mode=True):
        """Keep the student deterministic, including dropout and batch norm."""
        super().train(mode)
        self.student.eval()
        return self

    def trainable_parameters(self):
        """Yield residual actor, critic and exploration parameters only."""
        yield self.log_std
        yield from self.actor.parameters()
        yield from self.critic.parameters()

    def export_config(self):
        """Return JSON-compatible constructor settings, excluding the student."""
        return {
            "residual_obs_dim": self.residual_obs_dim,
            "num_actions": self.num_actions,
            "residual_bounds": self.residual_bounds.detach().cpu().tolist(),
            "action_scale": self.action_scale.detach().cpu().tolist(),
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "critic_hidden_dims": list(self.critic_hidden_dims),
            "initial_std": self.initial_std,
        }

    def _check_residual_obs(self, residual_obs):
        if not isinstance(residual_obs, torch.Tensor):
            raise TypeError("residual_obs must be a tensor")
        if residual_obs.ndim != 2 or residual_obs.shape[1] != self.residual_obs_dim:
            raise ValueError("residual_obs must have shape [batch, residual_obs_dim]")
        if residual_obs.shape[0] == 0:
            raise ValueError("residual_obs batch must be nonempty")
        if residual_obs.device != self.log_std.device or residual_obs.dtype != self.log_std.dtype:
            raise ValueError("residual_obs must have the policy's device and dtype")
        if not torch.isfinite(residual_obs).all():
            raise ValueError("residual_obs contains nonfinite values")

    def _distribution(self, residual_obs):
        self._check_residual_obs(residual_obs)
        mean = self.actor(residual_obs)
        std = self.log_std.exp().expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise FloatingPointError("Residual action distribution became nonfinite or degenerate")
        return Normal(mean, std)

    def value(self, residual_obs):
        """Evaluate the trainable critic, returning shape [batch]."""
        self._check_residual_obs(residual_obs)
        value = self.critic(residual_obs).squeeze(-1)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Residual critic returned nonfinite values")
        return value

    @torch.no_grad()
    def _student_actions(self, full_obs, history, batch_size):
        for name, tensor in (("full_obs", full_obs), ("history", history)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[0] != batch_size:
                raise ValueError("{} must have the same batch size as residual_obs".format(name))
            if tensor.device != self.log_std.device or tensor.dtype != self.log_std.dtype:
                raise ValueError("{} must have the policy's device and dtype".format(name))
            if not torch.isfinite(tensor).all():
                raise ValueError("{} contains nonfinite values".format(name))
        self.student.eval()
        actions = self.student(full_obs, history, hist_encoding=True)
        if not isinstance(actions, torch.Tensor) or actions.shape != (batch_size, self.num_actions):
            raise ValueError("student must return actions with shape [batch, num_actions]")
        if actions.device != self.log_std.device or actions.dtype != self.log_std.dtype:
            raise ValueError("student actions must have the policy's device and dtype")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Student returned nonfinite actions")
        return actions

    def _compose(self, student_actions, latent_actions):
        if not torch.isfinite(latent_actions).all():
            raise FloatingPointError("Residual latent actions became nonfinite")
        residual_rad = self.residual_bounds * torch.tanh(latent_actions)
        actions = student_actions + residual_rad / self.action_scale
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Composite actions became nonfinite")
        return actions, residual_rad

    @torch.no_grad()
    def act(self, full_obs, history, residual_obs):
        """Sample actions for collection; retain latent actions for PPO."""
        distribution = self._distribution(residual_obs)
        latent_actions = distribution.sample()
        student_actions = self._student_actions(full_obs, history, residual_obs.shape[0])
        actions, residual_rad = self._compose(student_actions, latent_actions)
        log_prob = distribution.log_prob(latent_actions).sum(-1)
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("Residual log probabilities became nonfinite")
        return {
            "actions": actions,
            "latent_actions": latent_actions,
            "log_prob": log_prob,
            "value": self.value(residual_obs),
            "mean": distribution.mean,
            "std": distribution.stddev,
            "residual_rad": residual_rad,
        }

    @torch.no_grad()
    def act_inference(self, full_obs, history, residual_obs):
        """Compose the frozen student with the bounded residual mean."""
        distribution = self._distribution(residual_obs)
        student_actions = self._student_actions(full_obs, history, residual_obs.shape[0])
        actions, _ = self._compose(student_actions, distribution.mean)
        return actions

    def evaluate_latent(self, residual_obs, latent_actions):
        """Evaluate stored unsquashed actions with gradients for PPO.

        Entropy is that of the latent normal distribution, the exploration
        regularizer used here; it is not the entropy of the bounded actions.
        """
        distribution = self._distribution(residual_obs)
        if latent_actions.shape != distribution.mean.shape:
            raise ValueError("latent_actions must have shape [batch, num_actions]")
        if latent_actions.device != residual_obs.device or latent_actions.dtype != residual_obs.dtype:
            raise ValueError("latent_actions must have the policy's device and dtype")
        if not torch.isfinite(latent_actions).all():
            raise ValueError("latent_actions contains nonfinite values")
        log_prob = distribution.log_prob(latent_actions).sum(-1)
        entropy = distribution.entropy().sum(-1)
        if not torch.isfinite(log_prob).all() or not torch.isfinite(entropy).all():
            raise FloatingPointError("Residual likelihood or entropy became nonfinite")
        return {
            "log_prob": log_prob,
            "value": self.value(residual_obs),
            "entropy": entropy,
            "mean": distribution.mean,
            "std": distribution.stddev,
        }
