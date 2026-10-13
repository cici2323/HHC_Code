"""Strict loading of the existing distilled student and residual checkpoints."""

import torch


def load_checkpoint(path, device="cpu"):
    # weights_only was introduced after the upstream PyTorch 1.10 dependency.
    # Older PyTorch uses pickle: load only checkpoints from trusted sources.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary: {}".format(path))
    return checkpoint


def build_student(spec):
    from rsl_rl.modules.actor_critic_mimic_priv import Actor, ActorDistillPriv, get_activation

    actor_config = dict(spec["actor"])
    actor_config["activation"] = get_activation(actor_config["activation"])
    if actor_config["activation"] is None:
        raise ValueError("Unknown student activation")
    return ActorDistillPriv(Actor(**actor_config), spec["distill"])


def load_student(student, path, device="cpu"):
    checkpoint = load_checkpoint(path, device)
    if "student_actor_state_dict" not in checkpoint:
        raise ValueError(
            "A distilled student checkpoint with 'student_actor_state_dict' is required. "
            "A teacher-only model cannot initialize residual adaptation."
        )
    student.load_state_dict(checkpoint["student_actor_state_dict"], strict=True)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student.eval()


def bounds_from_joint_names(joint_names):
    """Paper bounds in radians, in the supplied *policy action* joint order."""
    categories = {"shoulder": 0.35, "elbow": 0.25, "wrist": 0.10,
                  "waist": 0.05, "hip": 0.03, "knee": 0.03, "ankle": 0.02}
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("Action joint names must be nonempty and unique")
    bounds = []
    for name in joint_names:
        matches = [value for category, value in categories.items() if category in name.lower()]
        if len(matches) != 1:
            raise ValueError("No unambiguous residual bound for joint {!r}".format(name))
        bounds.append(matches[0])
    return bounds
