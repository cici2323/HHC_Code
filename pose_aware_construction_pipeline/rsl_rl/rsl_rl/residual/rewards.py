"""Tensor reward terms for pose-aware collaboration and residual adaptation.

Support points must be expressed in metres in a common frame. Workspace overlap
is a per-environment Boolean computed by the caller from occupancy intersection;
this module does not infer workspace geometry or construct missing task data.
"""

import math
from typing import Dict

import torch


def _floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("{} must be a floating-point tensor".format(name))
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} must contain only finite values".format(name))


def _matching_tensor(
    name: str, value: torch.Tensor, reference: torch.Tensor, shape: tuple
) -> None:
    _floating_tensor(name, value)
    if tuple(value.shape) != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, tuple(value.shape)))
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError("{} must share the support points' device and dtype".format(name))


def _boolean_tensor(
    name: str, value: torch.Tensor, reference: torch.Tensor, shape: tuple
) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError("{} must be a Boolean tensor".format(name))
    if tuple(value.shape) != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, tuple(value.shape)))
    if value.device != reference.device:
        raise ValueError("{} must share the support points' device".format(name))


def _coefficient(name: str, value: float, strictly_positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be a real number".format(name))
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be a real number".format(name)) from exc
    if not math.isfinite(value) or value < 0 or (strictly_positive and value == 0):
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError("{} must be finite and {}".format(name, qualifier))
    return value


def collaboration_reward(
    current_support_points: torch.Tensor,
    target_support_points: torch.Tensor,
    workspace_overlap: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    foot_velocity: torch.Tensor,
    contacts: torch.Tensor,
    *,
    support_weight: float = 5.0,
    workspace_weight: float = 3.0,
    stability_weight: float = 1.5,
    slip_weight: float = 0.05,
    support_tolerance: float = 0.02,
    support_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Return weighted reward terms and their sum, each with shape ``[N]``.

    Args:
        current_support_points: Two measured support points, ``[N, 2, 3]``.
        target_support_points: Corresponding targets, ``[N, 2, 3]``.
        workspace_overlap: Whether occupied human/robot workspaces intersect,
            ``[N]``, Boolean. The caller must supply this geometric result.
        base_angular_velocity: Robot base angular velocity, ``[N, 3]``.
        foot_velocity: Foot linear velocities, ``[N, F, 3]``.
        contacts: Which feet contact the surface, ``[N, F]``, Boolean.

    Floating-point inputs must have the same dtype and device. Defaults implement
    the paper's collaboration terms. ``slip_weight`` is a nonnegative penalty
    magnitude; the returned workspace and slip terms are nonpositive.
    """
    _floating_tensor("current_support_points", current_support_points)
    if current_support_points.ndim != 3 or tuple(current_support_points.shape[1:]) != (2, 3):
        raise ValueError("current_support_points must have shape [N, 2, 3]")
    batch_size = current_support_points.shape[0]
    if batch_size < 1:
        raise ValueError("the environment batch must not be empty")
    _matching_tensor("target_support_points", target_support_points, current_support_points, (batch_size, 2, 3))
    _matching_tensor("base_angular_velocity", base_angular_velocity, current_support_points, (batch_size, 3))
    _floating_tensor("foot_velocity", foot_velocity)
    if foot_velocity.ndim != 3 or foot_velocity.shape[1] < 1 or foot_velocity.shape[2] != 3:
        raise ValueError("foot_velocity must have shape [N, F, 3] with at least one foot")
    _matching_tensor("foot_velocity", foot_velocity, current_support_points, (batch_size, foot_velocity.shape[1], 3))
    _boolean_tensor("workspace_overlap", workspace_overlap, current_support_points, (batch_size,))
    _boolean_tensor("contacts", contacts, current_support_points, (batch_size, foot_velocity.shape[1]))
    support_weight = _coefficient("support_weight", support_weight)
    workspace_weight = _coefficient("workspace_weight", workspace_weight)
    stability_weight = _coefficient("stability_weight", stability_weight)
    slip_weight = _coefficient("slip_weight", slip_weight)
    support_tolerance = _coefficient("support_tolerance", support_tolerance)
    support_scale = _coefficient("support_scale", support_scale, strictly_positive=True)

    distances = torch.linalg.vector_norm(current_support_points - target_support_points, dim=-1)
    normalized_error = torch.relu(distances - support_tolerance) / support_scale
    support = support_weight * torch.exp(-normalized_error.square().mean(dim=-1))
    workspace = -workspace_weight * workspace_overlap.to(dtype=current_support_points.dtype)
    stability = stability_weight * torch.exp(-torch.linalg.vector_norm(base_angular_velocity, dim=-1))
    slip = -slip_weight * (foot_velocity.square().sum(dim=-1) * contacts).sum(dim=-1)
    terms = {"support": support, "workspace": workspace, "stability": stability, "slip": slip}
    terms["total"] = support + workspace + stability + slip
    for name, value in terms.items():
        _floating_tensor("computed {} reward".format(name), value)
    return terms


def adaptation_reward(
    tracking_reward: torch.Tensor,
    regularization_reward: torch.Tensor,
    *,
    tracking_weight: float = 0.2,
    **collaboration_kwargs,
) -> Dict[str, torch.Tensor]:
    """Combine tracking, collaboration and unchanged regularization rewards.

    ``tracking_reward`` and ``regularization_reward`` are ``[N]`` floating-point
    tensors. Pass the arguments of :func:`collaboration_reward` by keyword. The
    returned ``tracking`` term is weighted, ``regularization`` is unchanged,
    ``collaboration_total`` sums the four collaboration terms, and ``total`` is
    their aggregate. Do not include these collaboration terms in regularization.
    """
    terms = collaboration_reward(**collaboration_kwargs)
    reference = terms["total"]
    shape = tuple(reference.shape)
    _matching_tensor("tracking_reward", tracking_reward, reference, shape)
    _matching_tensor("regularization_reward", regularization_reward, reference, shape)
    tracking_weight = _coefficient("tracking_weight", tracking_weight)
    terms["collaboration_total"] = terms["total"]
    terms["tracking"] = tracking_weight * tracking_reward
    terms["regularization"] = regularization_reward
    terms["total"] = terms["tracking"] + terms["collaboration_total"] + regularization_reward
    _floating_tensor("computed adaptation reward", terms["total"])
    return terms
