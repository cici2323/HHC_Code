"""Recorded construction context and explicitly approximate workspace geometry.

The JSON input has this schema (all coordinates are metres)::

    {
      "schema_version": 1,
      "frame": "environment",
      "samples": [
        {
          "support_points": [[x, y, z], [x, y, z]],
          "human_workspace_boxes": [
            [[min_x, min_y, min_z], [max_x, max_y, max_z]]
          ]
        }
      ]
    }

Each sample contains two ordered support targets and the same nonzero number of
axis-aligned human workspace boxes with positive extents. Additional metadata
may describe provenance, but no configurations are generated or feasibility
certified here. Supply measured or otherwise validated task configurations.

The raw dataset context is the environment-frame support coordinates flattened
first, followed by box coordinates in input order (lower xyz, upper xyz for each
box). World-coordinate accessors add simulator environment origins. The separate
``robot_frame_context`` helper converts world coordinates into moving robot-frame
features: ordered support points followed by all eight corners of each box.
Deployment must recover the same relative geometry and corner ordering from
calibrated measurements of robot and human poses.
"""

import json
import numbers
from pathlib import Path
from typing import Optional, Union

import torch


def _finite_floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("{} must be a floating-point tensor".format(name))
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} must contain only finite values".format(name))


def _json_coordinates(name: str, value, shape: tuple, dtype, device) -> torch.Tensor:
    def validate(item, dimensions):
        if not dimensions:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("{} coordinates must be numbers".format(name))
            return
        if not isinstance(item, list) or len(item) != dimensions[0]:
            raise ValueError("{} must have shape {}".format(name, shape))
        for child in item:
            validate(child, dimensions[1:])

    validate(value, shape)
    try:
        tensor = torch.tensor(value, dtype=dtype, device=device)
    except (OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("{} coordinates cannot be represented in {}".format(name, dtype)) from exc
    _finite_floating_tensor(name, tensor)
    return tensor


class ConstructionDataset:
    """Load explicit construction configurations without inventing task data.

    Call ``sample`` for environments being reset, then retain those IDs through
    the episode. Sampling follows PyTorch's seeded random state unless an
    explicit same-device ``torch.Generator`` is passed to ``sample``.
    """

    def __init__(
        self,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch.dtype")
        self.path = Path(path)
        self.device = torch.device(device)
        self.dtype = dtype
        with self.path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("construction JSON must be an object")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("construction JSON requires schema_version: 1")
        if data.get("frame") != "environment":
            raise ValueError("construction JSON requires frame: 'environment'")
        samples = data.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("construction JSON requires a nonempty samples list")
        support, boxes = [], []
        box_count = None
        for index, sample in enumerate(samples):
            prefix = "samples[{}]".format(index)
            if not isinstance(sample, dict):
                raise ValueError("{} must be an object".format(prefix))
            support.append(_json_coordinates(
                prefix + ".support_points", sample.get("support_points"),
                (2, 3), dtype, self.device,
            ))
            workspace = sample.get("human_workspace_boxes")
            if not isinstance(workspace, list) or not workspace:
                raise ValueError("{}.human_workspace_boxes must be a nonempty list".format(prefix))
            if box_count is None:
                box_count = len(workspace)
            elif len(workspace) != box_count:
                raise ValueError("all samples must have the same number of human_workspace_boxes")
            workspace_tensor = _json_coordinates(
                prefix + ".human_workspace_boxes", workspace,
                (box_count, 2, 3), dtype, self.device,
            )
            if not bool((workspace_tensor[:, 1] > workspace_tensor[:, 0]).all()):
                raise ValueError("{}.human_workspace_boxes must have strictly positive extents".format(prefix))
            boxes.append(workspace_tensor)
        self._support = torch.stack(support)
        self._boxes = torch.stack(boxes)
        self.num_boxes = box_count
        self.sample_count = len(samples)

    @property
    def context_dim(self) -> int:
        return 6 + 6 * self.num_boxes

    def sample(self, num_envs: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Return ``[num_envs]`` int64 IDs, sampled uniformly with replacement."""
        if isinstance(num_envs, bool) or not isinstance(num_envs, numbers.Integral) or num_envs < 0:
            raise ValueError("num_envs must be a nonnegative integer")
        return torch.randint(self.sample_count, (int(num_envs),), device=self.device, generator=generator)

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long:
            raise TypeError("sample IDs must be an int64 tensor")
        if ids.ndim != 1 or ids.device != self._support.device:
            raise ValueError("sample IDs must have shape [N] on the dataset device")
        if bool(((ids < 0) | (ids >= self.sample_count)).any()):
            raise ValueError("sample IDs are out of range")

    def _validate_origins(self, ids: torch.Tensor, origins: torch.Tensor) -> None:
        self._validate_ids(ids)
        _finite_floating_tensor("origins", origins)
        if tuple(origins.shape) != (ids.numel(), 3):
            raise ValueError("origins must have shape [N, 3]")
        if origins.device != self._support.device or origins.dtype != self.dtype:
            raise ValueError("origins must share the dataset device and dtype")

    def support_points(self, ids: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
        """Return ordered world-frame support targets with shape ``[N, 2, 3]``."""
        self._validate_origins(ids, origins)
        return self._support[ids] + origins[:, None, :]

    def workspace_boxes(self, ids: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
        """Return world-frame human workspace bounds with shape ``[N, B, 2, 3]``."""
        self._validate_origins(ids, origins)
        return self._boxes[ids] + origins[:, None, None, :]

    def context(self, ids: torch.Tensor) -> torch.Tensor:
        """Return ``[N, 6 + 6B]`` context in the fixed environment frame.

        Environment origins are deliberately excluded, allowing independent
        simulator copies of the same scene to receive identical task context.
        """
        self._validate_ids(ids)
        return torch.cat((self._support[ids].flatten(1), self._boxes[ids].flatten(1)), dim=-1)


def robot_frame_context(
    support_points_world: torch.Tensor,
    workspace_boxes_world: torch.Tensor,
    root_position: torch.Tensor,
    root_quaternion_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Return relative construction geometry with shape ``[N, 6 + 24B]``.

    World inputs are two ordered support points ``[N, 2, 3]``, axis-aligned
    workspace bounds ``[N, B, 2, 3]`` (lower xyz, upper xyz), root positions
    ``[N, 3]``, and root orientations ``[N, 4]`` in xyzw convention. Quaternions
    rotate robot-frame vectors into the world frame; their inverses are applied
    here. All tensors must have a common floating dtype/device and finite values.
    Nonunit quaternions are normalized. Zero quaternions and those with every
    component at or below the dtype's smallest positive normal value are rejected.

    Output starts with support coordinates (6 entries), then each box's eight
    robot-frame corner coordinates (24 entries). Corner order is lexicographic
    in world-frame min/max bits: 000, 001, 010, 011, 100, 101, 110, 111, where
    a bit of zero chooses the lower bound for that x/y/z axis. Corners retain
    their identity after rotation; they are not re-bounded as an axis-aligned box.

    Simulation ground truth is used only to produce these relative perception
    features. Deployment needs corresponding relative target/corner positions
    from calibrated measured robot-human geometry; world-pose ground truth is
    not a policy input. Environment origins and absolute root position are not
    included in the output.
    """
    _finite_floating_tensor("support_points_world", support_points_world)
    if support_points_world.ndim != 3 or tuple(support_points_world.shape[1:]) != (2, 3):
        raise ValueError("support_points_world must have shape [N, 2, 3]")
    batch_size = support_points_world.shape[0]
    for name, value in (("workspace_boxes_world", workspace_boxes_world),
                        ("root_position", root_position),
                        ("root_quaternion_xyzw", root_quaternion_xyzw)):
        _finite_floating_tensor(name, value)
        if value.device != support_points_world.device or value.dtype != support_points_world.dtype:
            raise ValueError("{} must share support_points_world's device and dtype".format(name))
    if (workspace_boxes_world.ndim != 4 or workspace_boxes_world.shape[0] != batch_size
            or workspace_boxes_world.shape[1] < 1 or tuple(workspace_boxes_world.shape[2:]) != (2, 3)):
        raise ValueError("workspace_boxes_world must have shape [N, B, 2, 3] with B >= 1")
    if tuple(root_position.shape) != (batch_size, 3):
        raise ValueError("root_position must have shape [N, 3]")
    if tuple(root_quaternion_xyzw.shape) != (batch_size, 4):
        raise ValueError("root_quaternion_xyzw must have shape [N, 4]")
    lower, upper = workspace_boxes_world[:, :, 0], workspace_boxes_world[:, :, 1]
    if not bool((upper > lower).all()):
        raise ValueError("workspace_boxes_world must have strictly positive extents")

    # Scale first to avoid overflow/underflow when normalizing nonunit inputs.
    quaternion_scale = root_quaternion_xyzw.abs().amax(dim=-1, keepdim=True)
    if bool((quaternion_scale <= torch.finfo(root_quaternion_xyzw.dtype).tiny).any()):
        raise ValueError("root_quaternion_xyzw must be nonzero and not numerically vanishing")
    scaled_quaternion = root_quaternion_xyzw / quaternion_scale
    quaternion = scaled_quaternion / torch.linalg.vector_norm(scaled_quaternion, dim=-1, keepdim=True)

    corner_bits = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=torch.bool, device=support_points_world.device,
    )
    corners = torch.where(corner_bits[None, None], upper[:, :, None], lower[:, :, None])
    world_points = torch.cat((support_points_world, corners.flatten(1, 2)), dim=1)
    relative_points = world_points - root_position[:, None, :]
    inverse_xyz = -quaternion[:, None, :3].expand_as(relative_points)
    first_cross = torch.cross(inverse_xyz, relative_points, dim=-1)
    second_cross = torch.cross(inverse_xyz, first_cross, dim=-1)
    local_points = relative_points + 2 * (quaternion[:, None, 3:] * first_cross + second_cross)
    result = local_points.flatten(1)
    _finite_floating_tensor("computed robot-frame context", result)
    return result


def robot_sphere_box_overlap(
    robot_centres: torch.Tensor,
    robot_radii,
    human_boxes: torch.Tensor,
) -> torch.Tensor:
    """Return Boolean ``[N]`` intersection of robot spheres and human boxes.

    ``robot_centres`` has shape ``[N, R, 3]`` and ``human_boxes`` has shape
    ``[N, B, 2, 3]`` (lower xyz, upper xyz), in a common frame and in metres.
    ``robot_radii`` is a nonnegative scalar or a length-R vector. Sphere/box
    tangency counts as overlap. Positive-volume boxes and at least one sphere
    and box per environment are required; an empty environment batch is valid.

    The distance-to-AABB test is exact for these supplied primitive volumes.
    Spheres and boxes only approximate the actual robot and human occupancy;
    this helper is neither mesh intersection nor a safety guarantee.
    """
    _finite_floating_tensor("robot_centres", robot_centres)
    _finite_floating_tensor("human_boxes", human_boxes)
    if robot_centres.ndim != 3 or robot_centres.shape[1] < 1 or robot_centres.shape[2] != 3:
        raise ValueError("robot_centres must have shape [N, R, 3] with R >= 1")
    if (human_boxes.ndim != 4 or human_boxes.shape[0] != robot_centres.shape[0]
            or human_boxes.shape[1] < 1 or tuple(human_boxes.shape[2:]) != (2, 3)):
        raise ValueError("human_boxes must have shape [N, B, 2, 3] with B >= 1")
    if human_boxes.device != robot_centres.device or human_boxes.dtype != robot_centres.dtype:
        raise ValueError("human_boxes must share robot_centres' device and dtype")
    if not bool((human_boxes[:, :, 1] > human_boxes[:, :, 0]).all()):
        raise ValueError("human_boxes must have strictly positive extents")
    if isinstance(robot_radii, torch.Tensor):
        _finite_floating_tensor("robot_radii", robot_radii)
        if robot_radii.device != robot_centres.device or robot_radii.dtype != robot_centres.dtype:
            raise ValueError("robot_radii must share robot_centres' device and dtype")
        radii = robot_radii
    else:
        values = robot_radii if isinstance(robot_radii, (list, tuple)) else [robot_radii]
        if any(isinstance(value, bool) or not isinstance(value, numbers.Real) for value in values):
            raise TypeError("robot_radii must be real numbers")
        radii = torch.as_tensor(robot_radii, device=robot_centres.device, dtype=robot_centres.dtype)
        _finite_floating_tensor("robot_radii", radii)
    if radii.ndim > 1 or (radii.ndim == 1 and radii.shape[0] != robot_centres.shape[1]):
        raise ValueError("robot_radii must be a scalar or have shape [R]")
    if bool((radii < 0).any()):
        raise ValueError("robot_radii must be nonnegative")

    centres = robot_centres[:, :, None, :]
    lower = human_boxes[:, None, :, 0, :]
    upper = human_boxes[:, None, :, 1, :]
    separation = torch.maximum(lower - centres, centres - upper).clamp_min(0)
    radius_squared = radii.square() if radii.ndim == 0 else radii.square()[None, :, None]
    return (separation.square().sum(dim=-1) <= radius_squared).any(dim=2).any(dim=1)
