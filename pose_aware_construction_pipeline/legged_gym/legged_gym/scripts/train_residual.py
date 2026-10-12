"""Train/evaluate residual adaptation using the existing distilled G1 student.

Run --help without Isaac Gym to see residual options. Other arguments are passed
to the existing legged_gym CLI (exptid, --motion_name, --device, --headless, ...).
"""

import argparse
from pathlib import Path
import sys


def parse_residual_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: train_residual.py residual_run --student_checkpoint student.pt "
               "--motion_name motions_dance_release.yaml --device cuda:0 --headless",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--student_checkpoint", type=Path, help="Distilled student checkpoint; starts residual training")
    source.add_argument("--residual_checkpoint", type=Path, help="Self-contained residual checkpoint; resumes or evaluates")
    parser.add_argument("--evaluate", action="store_true", help="Run the deterministic composite controller")
    parser.add_argument("--evaluation_steps", type=int, default=1000)
    parser.add_argument("--context_dim", type=int, default=None,
                        help="Custom task context width; requires env.get_residual_context()")
    parser.add_argument("--output_dir", type=Path, help="Default: legged_gym/logs/residual/<exptid>")
    residual, remaining = parser.parse_known_args()
    checkpoint = residual.student_checkpoint or residual.residual_checkpoint
    if not checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(checkpoint))
    if residual.context_dim is not None and residual.context_dim < 0:
        parser.error("--context_dim must be nonnegative")
    if residual.evaluation_steps < 1:
        parser.error("--evaluation_steps must be positive")
    if "--resume" in remaining or "--resumeid" in remaining:
        parser.error("Use --residual_checkpoint to resume this stage")
    if "--task" not in remaining and not any(arg.startswith("--task=") for arg in remaining):
        remaining += ["--task", "g1_mimic_priv_distill"]
    sys.argv = [sys.argv[0]] + remaining
    return residual


def main():
    residual = parse_residual_args()
    # Isaac Gym must be imported before torch with the upstream simulator build.
    import isaacgym  # noqa: F401
    import torch
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.residual.checkpoints import load_checkpoint
    from rsl_rl.residual.observations import StudentObservationLayout
    from rsl_rl.residual.runner import ResidualRunner, default_config

    args = get_args()
    checkpoint = load_checkpoint(residual.student_checkpoint or residual.residual_checkpoint)
    if residual.student_checkpoint and "student_actor_state_dict" not in checkpoint:
        raise ValueError("This is not a distilled student checkpoint; train the student first")
    if residual.residual_checkpoint and checkpoint.get("format") != "pose_aware_residual_v1":
        raise ValueError("--residual_checkpoint requires a residual checkpoint")
    if args.seed is None:
        args.seed = 42
    if args.num_envs is None:
        args.num_envs = 1 if residual.evaluate else 512
    if args.max_iterations is None:
        args.max_iterations = 20000
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    # Keep the student dynamics and reference-motion representation unchanged.
    env_cfg.seed = args.seed
    env, env_cfg = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    if env_cfg.control.control_type != "P":
        raise ValueError("Residual joint-position targets require control_type='P'")
    dims = env_cfg.env
    if not callable(getattr(env, "get_extra_hist_obs", None)):
        raise ValueError("Choose a task exposing the distilled student's extra history")
    action_count = dims.num_policy_actions
    probe = torch.arange(action_count, dtype=torch.float32, device=env.device).unsqueeze(0)
    if len(env.dof_names) != action_count or not torch.equal(env.reindex(probe), probe):
        raise ValueError("Nonidentity action order requires an explicit policy joint-name mapping")
    layout = StudentObservationLayout(
        num_prop=dims.n_proprio, num_teacher_priv=dims.n_teacher_priv,
        num_demo=dims.n_demo, num_scan=dims.n_scan,
        num_priv_latent=dims.n_priv_latent, num_priv_explicit=dims.n_priv,
        num_hist=dims.history_len, text_feat_input_dim=dims.n_feature,
        num_student_history=dims.extra_history_len,
    )
    policy_cfg = class_to_dict(train_cfg.policy)
    student_spec = {
        "actor": {
            "num_prop": dims.n_proprio, "num_teacher_priv": dims.n_teacher_priv,
            "num_demo": dims.n_demo, "text_feat_input_dim": dims.n_feature,
            "text_feat_output_dim": policy_cfg["text_feat_output_dim"],
            "feat_hist_len": policy_cfg["feat_hist_len"], "num_scan": dims.n_scan,
            "num_actions": action_count, "scan_encoder_dims": policy_cfg["scan_encoder_dims"],
            "actor_hidden_dims": policy_cfg["actor_hidden_dims"],
            "priv_encoder_dims": policy_cfg["priv_encoder_dims"],
            "num_priv_latent": dims.n_priv_latent, "num_priv_explicit": dims.n_priv,
            "num_hist": dims.history_len, "activation": policy_cfg["activation"],
            "tanh_encoder_output": policy_cfg["tanh_encoder_output"],
        },
        "distill": class_to_dict(train_cfg.distill),
    }
    config = checkpoint["config"] if residual.residual_checkpoint else default_config()
    if residual.context_dim is not None:
        # Resume loader will reject a context dimension differing from checkpoint.
        config = dict(config, context_dim=residual.context_dim)
    environment_metadata = {
        "task": args.task, "motion_name": env_cfg.motion.motion_name,
        "motion_type": env_cfg.motion.motion_type, "sim_dt": env_cfg.sim.dt,
        "decimation": env_cfg.control.decimation, "asset": env_cfg.asset.file,
        "reward_config": class_to_dict(env_cfg.rewards),
        "context_schema": getattr(env, "residual_context_schema", "none"),
    }
    output_dir = residual.output_dir or Path(LEGGED_GYM_ROOT_DIR) / "logs" / "residual" / args.exptid
    if not residual.evaluate and output_dir.exists() and any(output_dir.iterdir()):
        if residual.residual_checkpoint is None:
            raise ValueError("Output directory is not empty; use a new experiment name")
    runner = ResidualRunner(
        env, student_spec, layout, env.dof_names, env_cfg.control.action_scale,
        config=config, log_dir=None if residual.evaluate else output_dir,
        device=args.rl_device, student_checkpoint=residual.student_checkpoint,
        residual_checkpoint=residual.residual_checkpoint,
        environment_metadata=environment_metadata,
    )
    if config["context_dim"] == 0:
        print("Motion-only residual baseline: using the supplied environment reward. "
              "Construction targets/workspace rewards require a custom task.")
    if residual.evaluate:
        for _ in range(residual.evaluation_steps):
            env.step(runner.inference_actions().to(env.device))
    else:
        runner.learn(args.max_iterations)


if __name__ == "__main__":
    main()
