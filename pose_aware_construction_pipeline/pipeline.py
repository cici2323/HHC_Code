#!/usr/bin/env python3
"""One entry point for teacher PPO, student distillation, and residual adaptation."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
# Always load this copy, even if another Paper_1_try checkout was installed -e.
for directory in (ROOT / "rsl_rl", ROOT / "legged_gym"):
    sys.path.insert(0, str(directory))


def load_reward_config(path):
    with Path(path).open() as stream:
        config = json.load(stream)
    required = {"motion_weights", "motion_scales", "adaptation_tracking_weight", "collaboration", "regularization"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Reward config must contain exactly {}".format(sorted(required)))
    motion_names = {"dof", "keypoint", "root_velocity", "root_angular_velocity", "heading", "height", "body_rotation"}
    collaboration_names = {"support_weight", "workspace_weight", "stability_weight", "slip_weight", "support_tolerance", "support_scale"}
    regularization_names = {"alive", "orientation", "dof_acc", "action_rate", "dof_error", "feet_stumble", "dof_pos_limits", "feet_air_time", "feet_force", "ankle_action", "waist_roll_pitch_error"}
    for section, names in (("motion_weights", motion_names), ("motion_scales", motion_names),
                           ("collaboration", collaboration_names), ("regularization", regularization_names)):
        if not isinstance(config[section], dict) or set(config[section]) != names:
            raise ValueError("Unexpected or missing reward terms in {}".format(section))
        for name, value in config[section].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Reward coefficients must be finite numbers")
            if section != "regularization" and value < 0:
                raise ValueError("{} coefficients must be nonnegative".format(section))
    if any(value <= 0 for value in config["motion_scales"].values()) or config["collaboration"]["support_scale"] <= 0:
        raise ValueError("Reward error scales must be positive")
    alpha = config["adaptation_tracking_weight"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("adaptation_tracking_weight must be finite and nonnegative")
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "student", "adaptation"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--motion-name", default="motions_dance_release.yaml")
    parser.add_argument("--iterations", type=int, default=None, help="Additional iterations; defaults teacher50000, student/adaptation20000")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Show the simulator viewer; headless by default")
    parser.add_argument("--headless", action="store_true", help="Explicit headless mode (default)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--teacher-checkpoint", type=Path, help="Initialize a new student from a teacher")
    source.add_argument("--student-checkpoint", type=Path, help="Initialize a new residual from a student")
    source.add_argument("--resume", type=Path, help="Resume/evaluate a checkpoint from this stage")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluation-steps", type=int, default=2000)
    parser.add_argument("--construction-data", type=Path, help="Measured support targets and human workspace boxes JSON")
    parser.add_argument("--robot-geometry", type=Path, help="Support point offsets and occupied robot spheres JSON")
    parser.add_argument("--motion-only", action="store_true", help="Adaptation diagnostic with support/workspace terms omitted")
    parser.add_argument("--reward-config", type=Path, default=ROOT / "configs" / "rewards.json")
    parser.add_argument("--save-interval", type=int, default=500)
    args = parser.parse_args(argv)
    args.run_name = args.run_name or args.stage
    args.iterations = args.iterations if args.iterations is not None else (50000 if args.stage == "teacher" else 20000)
    args.num_envs = args.num_envs if args.num_envs is not None else (1 if args.evaluate else 512)
    if min(args.iterations, args.num_envs, args.evaluation_steps, args.save_interval) < 1:
        parser.error("Iterations, environment count, evaluation steps and save interval must be positive")
    if args.render and args.headless:
        parser.error("Choose --render or --headless")
    if args.evaluate and args.resume is None:
        parser.error("--evaluate requires --resume with a checkpoint from the chosen stage")
    if args.stage == "teacher" and (args.teacher_checkpoint or args.student_checkpoint):
        parser.error("Teacher training starts from scratch or --resume")
    if args.stage == "student" and (args.student_checkpoint or not (args.teacher_checkpoint or args.resume)):
        parser.error("Student stage requires --teacher-checkpoint or --resume")
    if args.stage == "adaptation" and (args.teacher_checkpoint or not (args.student_checkpoint or args.resume)):
        parser.error("Adaptation requires --student-checkpoint or --resume")
    if args.stage != "adaptation" and (args.motion_only or args.construction_data or args.robot_geometry):
        parser.error("Construction inputs and --motion-only apply only to adaptation")
    if args.stage == "adaptation":
        if args.motion_only and (args.construction_data or args.robot_geometry):
            parser.error("--motion-only cannot be combined with construction inputs")
        if not args.motion_only and not (args.construction_data and args.robot_geometry):
            parser.error("Construction adaptation requires --construction-data and --robot-geometry; use --motion-only for the diagnostic baseline")
    for name in ("teacher_checkpoint", "student_checkpoint", "resume", "construction_data", "robot_geometry", "reward_config"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error("{} does not exist: {}".format(name, value))
    args.output_dir = args.output_dir or ROOT / "outputs" / args.run_name
    if not args.evaluate and args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        parser.error("Output directory is not empty; choose another --run-name or --output-dir")
    return args


def make_student_description(env_cfg, train_cfg, class_to_dict):
    from rsl_rl.residual.observations import StudentObservationLayout
    dims = env_cfg.env
    policy = class_to_dict(train_cfg.policy)
    layout = StudentObservationLayout(
        num_prop=dims.n_proprio, num_teacher_priv=dims.n_teacher_priv,
        num_demo=dims.n_demo, num_scan=dims.n_scan, num_priv_latent=dims.n_priv_latent,
        num_priv_explicit=dims.n_priv, num_hist=dims.history_len,
        text_feat_input_dim=dims.n_feature, num_student_history=dims.extra_history_len,
    )
    actor = {name: policy[name] for name in ("text_feat_output_dim", "feat_hist_len", "scan_encoder_dims", "actor_hidden_dims", "priv_encoder_dims", "activation", "tanh_encoder_output")}
    actor.update(num_prop=dims.n_proprio, num_teacher_priv=dims.n_teacher_priv,
                 num_demo=dims.n_demo, text_feat_input_dim=dims.n_feature,
                 num_scan=dims.n_scan, num_actions=dims.num_policy_actions,
                 num_priv_latent=dims.n_priv_latent, num_priv_explicit=dims.n_priv,
                 num_hist=dims.history_len)
    return layout, {"actor": actor, "distill": class_to_dict(train_cfg.distill)}


def main(argv=None):
    args = parse_args(argv)
    rewards = load_reward_config(args.reward_config)
    # NVIDIA Isaac Gym requires this import before torch.
    import isaacgym  # noqa: F401
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.residual.checkpoints import load_checkpoint

    source = args.resume or args.teacher_checkpoint or args.student_checkpoint
    checkpoint = load_checkpoint(source) if source is not None else None
    if args.student_checkpoint and "student_actor_state_dict" not in checkpoint:
        raise ValueError("Adaptation needs a distilled student, not a teacher-only checkpoint")
    if args.construction_data:
        from rsl_rl.residual.construction import ConstructionDataset
        ConstructionDataset(args.construction_data)  # Validate measured context before simulator allocation.
    task = "g1_paper_" + args.stage
    legacy_argv = [sys.argv[0], args.run_name, "--task", task,
                   "--motion_name", args.motion_name, "--device", args.device,
                   "--num_envs", str(args.num_envs), "--seed", str(args.seed)]
    if not args.render:
        legacy_argv.append("--headless")
    previous_argv, sys.argv = sys.argv, legacy_argv
    try:
        legacy_args = get_args()
    finally:
        sys.argv = previous_argv
    env_cfg, train_cfg = task_registry.get_cfgs(task)
    profile = env_cfg.reward_profile
    profile.stage = args.stage
    profile.tracking_weight = rewards["adaptation_tracking_weight"] if args.stage == "adaptation" else 1.0
    profile.motion_weights = rewards["motion_weights"]
    profile.motion_scales = rewards["motion_scales"]
    profile.collaboration = rewards["collaboration"]
    env_cfg.rewards.scales = SimpleNamespace(**rewards["regularization"])
    env_cfg.construction.enabled = args.stage == "adaptation" and not args.motion_only
    env_cfg.construction.data_path = str(args.construction_data.resolve()) if args.construction_data else ""
    env_cfg.construction.geometry_path = str(args.robot_geometry.resolve()) if args.robot_geometry else ""
    train_cfg.seed = args.seed
    train_cfg.runner.save_interval = args.save_interval
    env, env_cfg = task_registry.make_env(task, args=legacy_args, env_cfg=env_cfg)
    metadata = {
        "task": task, "motion_name": args.motion_name, "rewards": rewards,
        "sim_dt": env_cfg.sim.dt, "decimation": env_cfg.control.decimation,
        "joint_names": list(env.dof_names), "action_scale": env_cfg.control.action_scale,
        "asset": env_cfg.asset.file, "context_schema": env.residual_context_schema,
        "motion_only": args.motion_only,
    }
    if args.stage == "adaptation":
        from rsl_rl.residual.runner import ResidualRunner, default_config
        layout, student_spec = make_student_description(env_cfg, train_cfg, class_to_dict)
        config = checkpoint["config"] if args.resume else default_config()
        config = dict(config)
        config["context_dim"] = env.get_residual_context().shape[1]
        if not args.resume:
            config["save_interval"] = args.save_interval
        probe = torch.arange(env_cfg.env.num_policy_actions, device=env.device).unsqueeze(0)
        if len(env.dof_names) != env_cfg.env.num_policy_actions or not torch.equal(env.reindex(probe), probe):
            raise ValueError("This pipeline requires identity action joint order")
        if env_cfg.control.control_type != "P":
            raise ValueError("Residual targets require a position controller")
        runner = ResidualRunner(
            env, student_spec, layout, env.dof_names, env_cfg.control.action_scale,
            config=config, log_dir=None if args.evaluate else args.output_dir,
            device=args.device, student_checkpoint=args.student_checkpoint,
            residual_checkpoint=args.resume, environment_metadata=metadata,
        )
        if args.motion_only:
            print("Motion-only adaptation diagnostic: support and workspace rewards are omitted.")
        inference = runner.inference_actions
    else:
        from rsl_rl.runners import OnPolicyRunnerMimicPriv
        runner = OnPolicyRunnerMimicPriv(
            env, class_to_dict(train_cfg), log_dir=None if args.evaluate else str(args.output_dir),
            init_wandb=False, device=args.device,
        )
        runner.pipeline_metadata = metadata
        if args.resume:
            runner.load(args.resume, load_optimizer=not args.evaluate)
        elif args.teacher_checkpoint:
            runner.load_teacher(args.teacher_checkpoint)
        if args.stage == "teacher":
            def inference():
                return runner.alg.actor_critic.act_inference(env.get_observations().to(args.device), hist_encoding=False)
        else:
            def inference():
                return runner.alg.student_actor(env.get_observations().to(args.device), env.get_extra_hist_obs().to(args.device), hist_encoding=True)
    if not args.evaluate:
        # Only record a run after strict initialization/resume has succeeded.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "pipeline_config.json").open("w") as stream:
            json.dump({**metadata, "seed": args.seed, "stage": args.stage,
                       "num_envs": args.num_envs, "source_checkpoint": str(source) if source else None}, stream, indent=2)
    if args.evaluate:
        if args.stage != "adaptation":
            runner.alg.actor_critic.eval()
            if args.stage == "student":
                runner.alg.student_actor.eval()
        with torch.no_grad():
            for _ in range(args.evaluation_steps):
                env.step(inference().to(env.device))
    elif args.stage == "adaptation":
        runner.learn(args.iterations)
    else:
        runner.learn(args.iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
