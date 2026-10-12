"""Deployable residual observations and explicitly updated normalization.

The legacy environment carries simulator-only values in the same tensor as
student observations.  Keep that tensor for the frozen student API, but remove
its privileged slices before constructing input to the residual policy.
"""

from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import Tensor, nn


def _nonnegative_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a nonnegative integer".format(name))


def _matrix(name: str, value: Tensor, width: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError("{} must have shape (batch, {})".format(name, width))
    if not value.is_floating_point():
        raise TypeError("{} must have a floating-point dtype".format(name))


@dataclass(frozen=True)
class StudentObservationLayout:
    """Dimensions of the legacy ``G1MimicPriv`` observation layout.

    Full observations are ordered as features, proprioception, teacher-only
    state, demonstration, scan, explicit state, privileged latent state, and
    proprioceptive history.  ``extra_hist`` is the separate flattened history
    supplied to ``ActorDistillPriv``.  The explicit state slot is retained: the
    supplied G1 environment fills it with zeros, and callers must keep it
    deployable (for example, zeros or an onboard estimate).
    """

    num_prop: int
    num_teacher_priv: int
    num_demo: int
    num_scan: int
    num_priv_latent: int
    num_priv_explicit: int
    num_hist: int
    text_feat_input_dim: int
    num_student_history: int

    def __post_init__(self) -> None:
        for field in fields(self):
            _nonnegative_dimension(field.name, getattr(self, field.name))
        if self.num_prop == 0:
            raise ValueError("num_prop must be positive")
        if self.num_scan != 0:
            raise ValueError("The legacy distilled student does not support scan observations")

    @property
    def num_observations(self) -> int:
        return (
            self.text_feat_input_dim
            + self.num_prop
            + self.num_teacher_priv
            + self.num_demo
            + self.num_scan
            + self.num_priv_explicit
            + self.num_priv_latent
            + self.num_hist * self.num_prop
        )

    @property
    def history_dim(self) -> int:
        """Width of the separate, flattened student history tensor."""
        return self.num_student_history * self.num_prop

    def residual_dim(self, context_dim: int = 0) -> int:
        _nonnegative_dimension("context_dim", context_dim)
        return (
            self.num_observations
            - self.num_teacher_priv
            - self.num_priv_latent
            + self.history_dim
            + context_dim
        )

    def build(
        self,
        full_obs: Tensor,
        extra_hist: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Build residual input without reading either privileged slice.

        Inputs must be floating-point matrices with a common batch size, dtype,
        and device.  Only retained, deployable values are checked for finiteness;
        placeholders such as NaNs in discarded privileged slots are allowed.
        The returned tensor is newly allocated, and no input is modified.
        """
        _matrix("full_obs", full_obs, self.num_observations)
        _matrix("extra_hist", extra_hist, self.history_dim)
        for name, value in (("extra_hist", extra_hist), ("context", context)):
            if value is None:
                continue
            if name == "context":
                if not isinstance(value, Tensor):
                    raise TypeError("context must be a torch.Tensor")
                if value.ndim != 2:
                    raise ValueError("context must have shape (batch, context_dim)")
                _matrix(name, value, value.shape[1])
            if value.shape[0] != full_obs.shape[0]:
                raise ValueError("{} batch size does not match full_obs".format(name))
            if value.device != full_obs.device or value.dtype != full_obs.dtype:
                raise ValueError("{} dtype and device must match full_obs".format(name))

        prop_end = self.text_feat_input_dim + self.num_prop
        demo_start = prop_end + self.num_teacher_priv
        explicit_end = demo_start + self.num_demo + self.num_priv_explicit
        history_start = explicit_end + self.num_priv_latent
        parts = [
            full_obs[:, :prop_end],
            full_obs[:, demo_start:explicit_end],
            full_obs[:, history_start:],
            extra_hist,
        ]
        if context is not None:
            parts.append(context)
        result = torch.cat(parts, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Residual observations contain non-finite deployable values")
        return result


def build_residual_observations(
    layout: StudentObservationLayout,
    full_obs: Tensor,
    extra_hist: Tensor,
    context: Optional[Tensor] = None,
) -> Tensor:
    """Functional equivalent of :meth:`StudentObservationLayout.build`."""
    return layout.build(full_obs, extra_hist, context)


class RunningMeanStdNormalizer(nn.Module):
    """Population moments with updates controlled by the rollout runner.

    ``forward`` never updates statistics, including in training mode.  Call
    ``update`` only at a rollout boundary, then hold moments fixed while
    collecting that rollout and optimizing its PPO likelihood ratios.  Buffers
    are included in ``state_dict`` and follow module device/dtype conversion.
    """

    def __init__(self, dimension: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        _nonnegative_dimension("dimension", dimension)
        if dimension == 0:
            raise ValueError("dimension must be positive")
        if not 0.0 < clip < float("inf"):
            raise ValueError("clip must be positive and finite")
        if not 0.0 < eps < float("inf"):
            raise ValueError("eps must be positive and finite")
        self.dimension = dimension
        self.clip = float(clip)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("var", torch.ones(dimension))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    def _validate(self, batch: Tensor) -> None:
        _matrix("batch", batch, self.dimension)
        if batch.device != self.mean.device:
            raise ValueError("batch and normalizer must be on the same device")
        if not torch.isfinite(batch).all():
            raise ValueError("Normalization input contains non-finite values")

    @torch.no_grad()
    def update(self, batch: Tensor) -> None:
        """Merge detached batch population moments using parallel variance."""
        self._validate(batch)
        if batch.shape[0] == 0:
            raise ValueError("Cannot update normalization from an empty batch")
        values = batch.detach().to(dtype=self.mean.dtype)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count)
        moment_sum = self.var * self.count + batch_var * batch_count
        moment_sum = moment_sum + delta.square() * (self.count * batch_count / total_count)
        self.mean.copy_(new_mean)
        self.var.copy_((moment_sum / total_count).clamp_min(0.0))
        self.count.copy_(total_count)

    def forward(self, batch: Tensor) -> Tensor:
        self._validate(batch)
        normalized = (batch - self.mean) / torch.sqrt(self.var + self.eps)
        return normalized.clamp(-self.clip, self.clip)
