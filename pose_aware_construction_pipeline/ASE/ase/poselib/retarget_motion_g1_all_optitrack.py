from isaacgym.torch_utils import *  # noqa: F401,F403
import torch
import json
import numpy as np  # noqa: F401
import sys
from poselib.core.rotation3d import *  # noqa: F401,F403
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
import yaml
import os
import multiprocessing
from tqdm import tqdm


"""
Retarget OptiTrack/CMU-style motions to Unitree G1 skeleton using PoseLib retargeting.

Key fix:
- Ground alignment is computed from *foot joints/links only*, not the global minimum over all joints.
  Using the global minimum can mistakenly treat a hand/knee/tool contact as "ground", causing the
  whole body to shift upward (feet floating).
- Applies a small clearance and a configurable root height offset.

Expected retarget config JSON keys (typical):
- source_tpose: path
- target_tpose: path
- source_motion: directory containing source motions (.npy)
- target_motion_path: output directory
- joint_mapping: dict
- rotation: list/array
- scale: bool
- trim_frame_beg / trim_frame_end: int (can be -1)
- root_height_offset: float (optional, default 0.0)
- foot_clearance: float (optional, default 0.02)
- foot_joints: list[str] (optional; if missing we will try common G1 names)
- align_feet_to_ground: bool (optional, default True)

Expected motions list YAML (typical):
- motions: { motion_name: {...}, ... , root: {...} }
"""

# ------------------------- helpers -------------------------


def _node_indices_dict(skeleton_tree):
    """
    PoseLib SkeletonTree exposes node indices in one of these:
    - skeleton_tree._node_indices (dict)
    - skeleton_tree.node_names + lookup
    This helper returns a dict-like mapping name -> index when possible.
    """
    if hasattr(skeleton_tree, "_node_indices") and isinstance(skeleton_tree._node_indices, dict):
        return skeleton_tree._node_indices
    if hasattr(skeleton_tree, "node_names"):
        return {name: i for i, name in enumerate(skeleton_tree.node_names)}
    return {}


def _pick_foot_indices(skeleton_tree, preferred_names=None):
    """
    Return a list of joint indices to be used for ground alignment.
    Tries preferred_names first, else tries common G1 foot/ankle/toe names.
    """
    name2idx = _node_indices_dict(skeleton_tree)

    # User-provided preferred names have highest priority
    if preferred_names:
        found = [name2idx[n] for n in preferred_names if n in name2idx]
        if len(found) >= 2:
            return found

    # Common candidates (include both "link" and non-link variants)
    candidates = [
        # feet / toes (best if available)
        "left_foot_link", "right_foot_link",
        "left_toe_link", "right_toe_link",
        "left_toe", "right_toe",
        "left_foot", "right_foot",
        # ankles (fallback)
        "left_ankle_link", "right_ankle_link",
        "left_ankle", "right_ankle",
        "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_ankle_pitch", "right_ankle_pitch",
    ]
    found = [name2idx[n] for n in candidates if n in name2idx]

    if len(found) >= 2:
        return found
    if len(found) == 1:
        return found

    return []


def _compute_ground_min_z(target_motion, frame_beg, frame_end, foot_joint_names=None):
    """
    Compute the minimum Z height used to align to ground.
    Prefer foot joints; fallback to global minimum if feet cannot be identified.
    """
    tar_global_pos = target_motion.global_translation[frame_beg:frame_end, ...]  # [T, J, 3]
    sk = target_motion.skeleton_tree

    foot_ids = _pick_foot_indices(sk, preferred_names=foot_joint_names)
    if foot_ids:
        foot_z = tar_global_pos[:, foot_ids, 2]  # [T, |F|]
        return torch.min(foot_z), True

    # Fallback: global minimum (less stable; can cause floating when hands touch ground)
    return torch.min(tar_global_pos[..., 2]), False


# ------------------------- main worker -------------------------


def process(i, motion_name, source_tpose_path, target_tpose_path,
            source_motion_dir, target_motion_dir, retarget_data):
    source_motion_path = os.path.join(source_motion_dir, motion_name + ".npy")
    target_motion_path = os.path.join(target_motion_dir, motion_name + ".npy")

    try:
        src_motion = SkeletonMotion.from_file(source_motion_path)
    except Exception:
        print("failed to load motion:", source_motion_path)
        return

    try:
        source_tpose = SkeletonState.from_file(source_tpose_path)
        target_tpose = SkeletonState.from_file(target_tpose_path)
    except Exception as e:
        print("failed to load tpose:", e)
        return

    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    print(f"run retargeting {i}: {motion_name}")

    # retarget
    tgt_motion = src_motion.retarget_to_by_tpose(
        joint_mapping=retarget_data["joint_mapping"],
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=retarget_data.get("scale", True),
    )

    # trim frames
    frame_beg = int(retarget_data.get("trim_frame_beg", -1))
    frame_end = int(retarget_data.get("trim_frame_end", -1))

    if frame_beg < 0:
        frame_beg = 0
    if frame_end < 0 or frame_end > tgt_motion.local_rotation.shape[0]:
        frame_end = tgt_motion.local_rotation.shape[0]
    if frame_end <= frame_beg:
        print(f"skip {motion_name}: invalid trim [{frame_beg}, {frame_end})")
        return

    local_rotation = tgt_motion.local_rotation[frame_beg:frame_end, ...]
    root_translation = tgt_motion.root_translation[frame_beg:frame_end, ...].clone()

    # ------------------------- ground alignment fix -------------------------
    align = bool(retarget_data.get("align_feet_to_ground", True))
    root_height_offset = float(retarget_data.get("root_height_offset", 0.0))
    foot_clearance = float(retarget_data.get("foot_clearance", 0.02))
    foot_joint_names = retarget_data.get("foot_joints", None)

    if align:
        min_h, used_feet = _compute_ground_min_z(
            tgt_motion, frame_beg, frame_end, foot_joint_names=foot_joint_names
        )
        # Shift so the lowest foot point touches z=0, then add clearance
        root_translation[:, 2] += (-min_h + foot_clearance)
        if not used_feet:
            print(f"[WARN] {motion_name}: could not identify foot joints; "
                  f"used global min height for alignment (may cause floating).")

    # Additional offset (URDF sole thickness / sim ground contact tuning)
    root_translation[:, 2] += root_height_offset
    # ----------------------------------------------------------------------

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        tgt_motion.skeleton_tree, local_rotation, root_translation, is_local=True
    )
    tgt_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=tgt_motion.fps)

    # save
    os.makedirs(target_motion_dir, exist_ok=True)
    tgt_motion.to_file(target_motion_path)


def save_all(
    retarget_data_path="data/configs/retarget_cmu_to_g1_prefixed_optitrack.json",
    motions_yaml_path="data/configs/motions_debug_optitrack.yaml",
):
    with open(retarget_data_path, "r") as f:
        retarget_data = json.load(f)

    source_tpose_path = retarget_data["source_tpose"]
    target_tpose_path = retarget_data["target_tpose"]
    source_motion_dir = retarget_data["source_motion"]
    target_motion_dir = retarget_data["target_motion_path"]

    with open(motions_yaml_path, "r") as f:
        motions_list = yaml.load(f, Loader=yaml.SafeLoader)["motions"]

    all_motion_names = []
    for motion_entry in motions_list.keys():
        if motion_entry == "root":
            continue
        out_file = os.path.join(target_motion_dir, motion_entry + ".npy")
        if os.path.exists(out_file):
            print("Already exists, skip:", motion_entry)
            continue
        all_motion_names.append(motion_entry)

    all_motion_names.sort()
    print("Motions to retarget:", len(all_motion_names))

    if not all_motion_names:
        return

    n_workers = max(1, min(multiprocessing.cpu_count(), len(all_motion_names)))

    args = [
        (i, motion_name, source_tpose_path, target_tpose_path,
         source_motion_dir, target_motion_dir, retarget_data)
        for i, motion_name in enumerate(all_motion_names)
    ]

    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm(pool.starmap(process, args), total=len(all_motion_names)))


if __name__ == "__main__":
    # Optional CLI usage:
    # python retarget_motion_g1_all_optitrack.py [retarget_json] [motions_yaml]
    if len(sys.argv) >= 3:
        save_all(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        save_all(sys.argv[1])
    else:
        save_all()

