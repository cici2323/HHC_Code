"""Connect paper rewards to vector environments, with one control-dt scaling.

The environment implements motion_reward_inputs(), collaboration_reward_inputs()
and stabilization_reward_inputs(). All inputs describe the current transition
before automatic reset. The mixin has no Isaac Gym dependency and is CPU-testable.
"""

import torch

from .motion_rewards import motion_tracking_reward
from .rewards import collaboration_reward


class PaperRewardMixin:
    def _prepare_reward_function(self):
        # The inherited implementation binds only the regularization functions
        # configured in rewards.scales and multiplies those scales by dt.
        super()._prepare_reward_function()
        names = ["motion_" + name for name in self.cfg.reward_profile.motion_weights]
        names += ["support", "workspace", "stability", "slip", "paper_total"]
        for name in names:
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device)

    def compute_reward(self):
        profile = self.cfg.reward_profile
        motion = motion_tracking_reward(
            **self.motion_reward_inputs(),
            weights=profile.motion_weights, scales=profile.motion_scales,
        )
        # Regularization scales already include dt. Do not attenuate by alpha.
        regularization = torch.zeros_like(self.rew_buf)
        for name, function in zip(self.reward_names, self.reward_functions):
            contribution = function() * self.reward_scales[name]
            regularization += contribution
            self.episode_sums[name] += contribution
        if "termination" in self.reward_scales:
            contribution = self._reward_termination() * self.reward_scales["termination"]
            regularization += contribution
            self.episode_sums["termination"] += contribution
        terms = {}
        for name, contribution in motion.items():
            if name != "total":
                terms["motion_" + name] = profile.tracking_weight * contribution * self.dt
        total = profile.tracking_weight * motion["total"] * self.dt + regularization
        if profile.stage == "adaptation":
            if self.cfg.construction.enabled:
                collaboration = collaboration_reward(
                    **self.collaboration_reward_inputs(), **profile.collaboration,
                )
            else:
                # Explicit motion-only diagnostic: support/workspace have no data.
                inputs = self.stabilization_reward_inputs()
                collaboration = {
                    "support": torch.zeros_like(total),
                    "workspace": torch.zeros_like(total),
                    "stability": profile.collaboration["stability_weight"] * torch.exp(
                        -torch.linalg.vector_norm(inputs["base_angular_velocity"], dim=-1)),
                    "slip": -profile.collaboration["slip_weight"] * (
                        inputs["foot_velocity"].square().sum(-1) * inputs["contacts"]).sum(-1),
                }
                collaboration["total"] = sum(collaboration.values())
            for name in ("support", "workspace", "stability", "slip"):
                terms[name] = collaboration[name] * self.dt
            total += collaboration["total"] * self.dt
        if not torch.isfinite(total).all():
            raise FloatingPointError("Paper reward became nonfinite")
        # No aggregate clipping: the configured paper penalties retain their value.
        self.rew_buf.copy_(total)
        for name, contribution in terms.items():
            self.episode_sums[name] += contribution
        self.episode_sums["paper_total"] += total
        self.extras["reward_terms"] = {
            **{name: contribution.detach().clone() for name, contribution in terms.items()},
            "regularization": regularization.detach().clone(),
            "total": total.detach().clone(),
        }
