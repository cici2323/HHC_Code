"""Motion-tracking equations from Supplementary Table 2 of the paper.

All positions and velocities and their references must share coordinate frames
and units. Positions are in metres; joint positions, headings and body rotations
are in radians. Body rotation uses Euler angles in ``[roll, pitch, yaw]`` order
and a componentwise wrapped difference. Keypoints are flattened into one vector
per environment before computing their Euclidean norm. None of the L2 norms in
the tracking equations is squared or averaged over coordinates.
"""

import math
from collections.abc import Mapping
from typing import Dict, Optional

import torch


PAPER_MOTION_WEIGHTS = {
    "dof": 5.0,
    "keypoint": 4.0,
    "root_velocity": 6.0,
    "root_angular_velocity": 3.0,
    "heading": 3.0,
    "height": 2.0,
    "body_rotation": 3.0,
}

PAPER_MOTION_SCALES = {
    "dof": 0.25,
    "keypoint": 0.25,
    "root_velocity": 4.0,
    "root_angular_velocity": 2.0,
    "heading": 2.0,
    "height": 10.0,
    "body_rotation": 2.0,
}


def _floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("{} must be a floating-point tensor".format(name))
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} must contain only finite values".format(name))


def _matching_tensor(name, value, reference, shape) -> None:
    _floating_tensor(name, value)
    if tuple(value.shape) != tuple(shape):
        raise ValueError("{} must have shape {}, got {}".format(name, tuple(shape), tuple(value.shape)))
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError("{} must share dof_position's device and dtype".format(name))


def _coefficients(name, defaults, overrides, *, positive=False):
    result = dict(defaults)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("{} must be a mapping from term names to coefficients".format(name))
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError("unknown {} term(s): {}".format(name, ", ".join(map(str, unknown))))
        result.update(overrides)
    for term, value in result.items():
        if isinstance(value, bool):
            raise TypeError("{}[{}] must be a real number".format(name, term))
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("{}[{}] must be a real number".format(name, term)) from exc
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError("{}[{}] must be finite and {}".format(name, term, qualifier))
        result[term] = value
    return result


def _wrapped_difference(current, reference):
    return torch.remainder(current - reference + math.pi, 2.0 * math.pi) - math.pi


def motion_tracking_terms(
    dof_position: torch.Tensor,
    reference_dof_position: torch.Tensor,
    keypoint_position: torch.Tensor,
    reference_keypoint_position: torch.Tensor,
    root_velocity: torch.Tensor,
    reference_root_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    reference_root_angular_velocity: torch.Tensor,
    heading: torch.Tensor,
    reference_heading: torch.Tensor,
    height: torch.Tensor,
    reference_height: torch.Tensor,
    body_rotation: torch.Tensor,
    reference_body_rotation: torch.Tensor,
    *,
    scales: Optional[Mapping] = None,
) -> Dict[str, torch.Tensor]:
    """Return seven unweighted tracking terms, each with shape ``[N]``.

    Shapes are ``[N, D]`` for DoFs, ``[N, K, 3]`` for keypoints, ``[N, 3]``
    for root velocities and body Euler rotation, and ``[N]`` for heading and
    height. Each reference must match its current-state tensor exactly. All
    inputs must be finite floating tensors with a common dtype and device.

    ``scales`` optionally overrides individual positive exponent coefficients
    from :data:`PAPER_MOTION_SCALES`; unknown term names are rejected. Heading
    differences use the shortest wrapped angle to avoid a discontinuity at pi.
    """
    _floating_tensor("dof_position", dof_position)
    if dof_position.ndim != 2 or min(dof_position.shape) < 1:
        raise ValueError("dof_position must have shape [N, D] with N and D positive")
    batch_size = dof_position.shape[0]
    _matching_tensor("reference_dof_position", reference_dof_position, dof_position, dof_position.shape)
    _floating_tensor("keypoint_position", keypoint_position)
    if keypoint_position.ndim != 3 or keypoint_position.shape[1] < 1 or keypoint_position.shape[2] != 3:
        raise ValueError("keypoint_position must have shape [N, K, 3] with at least one keypoint")
    keypoint_shape = (batch_size, keypoint_position.shape[1], 3)
    _matching_tensor("keypoint_position", keypoint_position, dof_position, keypoint_shape)
    _matching_tensor("reference_keypoint_position", reference_keypoint_position, dof_position, keypoint_shape)
    for name, value in (
        ("root_velocity", root_velocity),
        ("reference_root_velocity", reference_root_velocity),
        ("root_angular_velocity", root_angular_velocity),
        ("reference_root_angular_velocity", reference_root_angular_velocity),
        ("body_rotation", body_rotation),
        ("reference_body_rotation", reference_body_rotation),
    ):
        _matching_tensor(name, value, dof_position, (batch_size, 3))
    for name, value in (
        ("heading", heading),
        ("reference_heading", reference_heading),
        ("height", height),
        ("reference_height", reference_height),
    ):
        _matching_tensor(name, value, dof_position, (batch_size,))
    scales = _coefficients("scales", PAPER_MOTION_SCALES, scales, positive=True)

    errors = {
        "dof": (dof_position - reference_dof_position).abs().sum(dim=-1),
        "keypoint": torch.linalg.vector_norm(
            (keypoint_position - reference_keypoint_position).flatten(start_dim=1), dim=-1
        ),
        "root_velocity": torch.linalg.vector_norm(root_velocity - reference_root_velocity, dim=-1),
        "root_angular_velocity": torch.linalg.vector_norm(
            root_angular_velocity - reference_root_angular_velocity, dim=-1
        ),
        "heading": _wrapped_difference(heading, reference_heading).abs(),
        "height": (height - reference_height).abs(),
        "body_rotation": torch.linalg.vector_norm(
            _wrapped_difference(body_rotation, reference_body_rotation), dim=-1
        ),
    }
    terms = {name: torch.exp(-scales[name] * error) for name, error in errors.items()}
    for name, value in terms.items():
        _floating_tensor("computed {} tracking reward".format(name), value)
    return terms


def motion_tracking_reward(
    dof_position: torch.Tensor,
    reference_dof_position: torch.Tensor,
    keypoint_position: torch.Tensor,
    reference_keypoint_position: torch.Tensor,
    root_velocity: torch.Tensor,
    reference_root_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    reference_root_angular_velocity: torch.Tensor,
    heading: torch.Tensor,
    reference_heading: torch.Tensor,
    height: torch.Tensor,
    reference_height: torch.Tensor,
    body_rotation: torch.Tensor,
    reference_body_rotation: torch.Tensor,
    *,
    weights: Optional[Mapping] = None,
    scales: Optional[Mapping] = None,
) -> Dict[str, torch.Tensor]:
    """Return weighted motion terms and ``total``, all with shape ``[N]``.

    Input conventions match :func:`motion_tracking_terms`. ``weights`` and
    ``scales`` may partially override the paper defaults. Weights must be finite
    and nonnegative. Perfect tracking yields a total of 26 under the defaults.
    This function does not apply simulation ``dt`` or the residual stage's 0.2
    tracking factor. Pass ``total`` to ``adaptation_reward`` once for that stage.
    """
    weights = _coefficients("weights", PAPER_MOTION_WEIGHTS, weights)
    terms = motion_tracking_terms(
        dof_position, reference_dof_position,
        keypoint_position, reference_keypoint_position,
        root_velocity, reference_root_velocity,
        root_angular_velocity, reference_root_angular_velocity,
        heading, reference_heading, height, reference_height,
        body_rotation, reference_body_rotation, scales=scales,
    )
    terms = {name: weights[name] * value for name, value in terms.items()}
    terms["total"] = sum(terms.values())
    _floating_tensor("computed total tracking reward", terms["total"])
    return terms
