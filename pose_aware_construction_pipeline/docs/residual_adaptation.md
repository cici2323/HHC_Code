> This is the earlier standalone residual-entry reference. For the full pipeline and updated paper rewards in this folder, use [PIPELINE.md](PIPELINE.md) and [REWARDS.md](REWARDS.md). Its original validation count describes the earlier extension.

# Student-to-residual adaptation

This extension implements the frozen-student residual stage described in
*Pose-aware Human-Humanoid Collaboration in Construction*. It builds on this
repository's `ActorDistillPriv` student. It is a new implementation, with CPU
validation; it is not a release of the paper's trained construction policy or
experimental data.

## Controller and training objective

The inherited environment maps policy actions to joint targets as
`q_target = q_default + action_scale * action`. Its G1 action scale is `0.25`.
For the frozen student's deterministic action `a_student`, the residual is:

```text
u ~ Normal(mu(residual_observation), exp(log_std))
delta_q = joint_bounds_rad * tanh(u)
action = a_student + delta_q / action_scale
q_target = q_default + action_scale * a_student + delta_q
```

The student and all of its encoders stay frozen and in evaluation mode.
Only the residual actor, critic, and log standard deviation are optimized.
PPO stores `u` and its Gaussian log probability; the fixed transformation's
Jacobian cancels in the importance ratio. Entropy regularization uses the latent
Gaussian entropy. The final actor layer starts at zero and the latent standard
deviation at `0.10`. Deterministic inference initially reproduces the student
exactly; stochastic training intentionally explores small corrections.

| Joint family | Correction bound (rad) |
| --- | ---: |
| Shoulder | 0.35 |
| Elbow | 0.25 |
| Wrist | 0.10 |
| Waist | 0.05 |
| Hip / knee | 0.03 |
| Ankle | 0.02 |

Bounds are matched to actual policy joint names. Unknown names fail explicitly.
The CLI validates the existing identity action order and position controller;
a robot with another action permutation needs an explicit mapping. The current
23-action G1 asset has no actuated wrists. Bounds limit corrections, and the
environment retains its existing action clipping and torque controller.

## What is supplied

| File | Role |
| --- | --- |
| `rsl_rl/rsl_rl/residual/policy.py` | Frozen student plus residual actor and critic |
| `rsl_rl/rsl_rl/residual/ppo.py` | Latent-action PPO and terminal-aware GAE |
| `rsl_rl/rsl_rl/residual/observations.py` | Deployable feature selection and normalization |
| `rsl_rl/rsl_rl/residual/checkpoints.py` | Strict student loading and joint bounds |
| `rsl_rl/rsl_rl/residual/runner.py` | Rollout collection, logs, resume, inference |
| `rsl_rl/rsl_rl/residual/rewards.py` | Support, workspace, stability, slip, adaptation reward |
| `legged_gym/legged_gym/scripts/train_residual.py` | Training and evaluation CLI |
| `tests/` | CPU policy, reward, observation, and runner tests |

## Training and evaluation

Use the existing Linux/CUDA/Isaac Gym environment described in
[the upstream setup](upstream_exbody2.md). Install this checkout's packages:

```bash
pip install -e ./rsl_rl
pip install -e ./legged_gym
```

First obtain a **distilled student checkpoint** from the existing student stage.
It must contain `student_actor_state_dict`, including the frozen original
actor/encoders and `student_actor_backbone`. The bundled
`legged_gym/logs/g1/000-00-dance_teacher/model_50000.pt` contains only the teacher
and is deliberately rejected. A malformed student checkpoint is also rejected.

From the repository root, run the supplied motion-only residual baseline:

```bash
python legged_gym/legged_gym/scripts/train_residual.py residual_run \
  --student_checkpoint /path/to/distilled_student/model_10000.pt \
  --motion_name motions_dance_release.yaml \
  --num_envs 512 --max_iterations 20000 --device cuda:0 --headless
```

Use the **same motion representation and student architecture** as distillation.
The example motion name resolves to the included cached dance `.pkl` through
the upstream loader, even though its corresponding YAML is absent. If the cache
is unavailable, provide the motion YAML and referenced data. Do not request
cache regeneration without the source motion files.

Resume with all embedded student weights, residual weights, normalization,
optimizer state, adaptive learning rate, iteration, and Torch RNG state:

```bash
python legged_gym/legged_gym/scripts/train_residual.py residual_run \
  --residual_checkpoint legged_gym/logs/residual/residual_run/residual_500.pt \
  --motion_name motions_dance_release.yaml \
  --num_envs 512 --max_iterations 500 --device cuda:0 --headless
```

`--max_iterations` means additional updates, including on resume. Simulator
state is reset on resume; this is not bitwise trajectory replay. Checkpoints
are saved atomically every 500 updates and at the end. Use a new output directory
when branching from an older checkpoint to preserve later checkpoints/logs.
Architecture, observation layout, action scale/order, motion, reward config,
and context schema must agree with the saved metadata. Load only trusted
checkpoints with upstream PyTorch 1.10, whose loader uses pickle.

Evaluate deterministically without updating normalization:

```bash
python legged_gym/legged_gym/scripts/train_residual.py residual_eval \
  --residual_checkpoint legged_gym/logs/residual/residual_run/residual_500.pt \
  --motion_name motions_dance_release.yaml \
  --evaluate --evaluation_steps 2000 --num_envs 1 --device cuda:0
```

Pass `--evaluate --student_checkpoint ...` to evaluate the initial zero-residual
student baseline. The CLI writes JSON configuration and JSONL metrics; it does
not start a new W&B run.

## Construction-task integration hooks

The default task is `g1_mimic_priv_distill`. It uses its **existing motion reward**
unchanged. Construction adaptation requires a custom registered environment;
passing `--context_dim` alone does not turn the dance environment into a
collaboration task.

Your custom task must provide:

1. `get_observations() -> [N,1363]` and `get_extra_hist_obs() -> [N,1950]`,
   matching the student's training layout, or a corresponding explicit layout
   and architecture configuration.
2. `get_residual_context() -> [N,C]` for the current observation: two requested
   support positions (6 values) and a documented, fixed-width encoding of the
   measured/estimated human workspace. Use metres in a common frame; decide and
   record the reference frame. Add `residual_context_schema` to identify the
   feature order, frame, units, and workspace encoding. Context must be available
   at deployment, and reset at the same time as the environment.
3. A collaboration reward computed **before automatic reset**. Supply current
   robot support positions, requested positions, actual occupied-volume overlap,
   base angular velocity, foot velocities and contact flags to the helper below.
4. Correct terminations. For time-limit truncations, return both `dones=True`
   and `infos['time_outs']=True`, plus raw, pre-reset
   `infos['terminal_residual_obs']` of shape `[N,3187+C]`. Build those features
   from terminal observations, terminal extra history, and terminal context
   before resetting history. Entries outside the timeout mask are unused.
   Genuine falls/collisions are terminal, with `time_outs=False`.

Reward integration inside your environment's pre-reset reward computation:

```python
from rsl_rl.residual.rewards import adaptation_reward

# All input terms refer to this same physical transition. tracking_reward and
# regularization_reward must be separate, consistently scaled [N] tensors.
terms = adaptation_reward(
    tracking_reward, regularization_reward,
    current_support_points=robot_support_points,  # [N,2,3], metres
    target_support_points=requested_support_points,  # [N,2,3]
    workspace_overlap=occupied_volume_overlap,  # [N], bool
    base_angular_velocity=base_ang_vel,  # [N,3]
    foot_velocity=foot_vel,  # [N,F,3]
    contacts=foot_contact,  # [N,F], bool
)
self.rew_buf[:] = terms['total']
```

The result implements:

```text
support   = 5 * exp(-mean((relu(distance - 0.02) / 0.05)^2))
workspace = -3 * indicator(occupied human and robot volumes intersect)
stability = 1.5 * exp(-L2(base_angular_velocity))
slip      = -0.05 * sum(contact * squared_L2(foot_velocity))
total     = 0.2 * tracking + support + workspace + stability + slip + regularization
```

Distances are Euclidean over the two support points; `0.02 m` is a reward
plateau tolerance. The helper does not compute workspace geometry. Do not
substitute a proximity guess for occupancy intersection without defining that
change in the method. Do not pass the entire inherited reward as `tracking`:
that would also scale regularization. The helpers return raw terms without
time-step scaling or clipping. The existing environment scales its reward
weights by control `dt`; use one consistent convention and apply it once to
all new terms.

Register your subclass in `legged_gym/legged_gym/envs/__init__.py` with the
student-compatible configurations, then select it:

```bash
python legged_gym/legged_gym/scripts/train_residual.py construction_run \
  --task YOUR_REGISTERED_TASK --context_dim YOUR_CONTEXT_WIDTH \
  --student_checkpoint /path/to/student.pt \
  --motion_name YOUR_MOTION.yaml --device cuda:0 --headless
```

The existing G1 task leaves time-limit termination disabled and loops its
reference motions. The runner rejects ambiguous timeout bootstrapping rather
than using a reset observation as a terminal state. If enabling time limits,
implement the pre-reset hook above.

## Observations and normalization

The source observation layout is:

```text
feature history312 | proprio78 | teacher-only75 | demo64 | explicit3
| privileged latent51 | proprio history780 = 1363
extra student history: 25 * 78 = 1950
```

The frozen student is always called with `hist_encoding=True`. Its source
encoders use 10 history steps; its separate student backbone also consumes 25
steps, as required by the existing checkpoint architecture. This preserves
the supplied implementation rather than assuming the paper's history notation
fully specifies the checkpoint.

The residual actor and critic both receive the raw deployable student features
and extra history, concatenated with context: `3187+C` values. They exclude
teacher-only state and privileged latent state. The explicit three-value slot
is zero in the supplied environment; a custom environment must keep it zero or
use deployable estimates. Scan observations are currently unsupported by the
legacy student and are rejected. The use of an unprivileged critic and this
specific feature serialization are implementation choices.

Only residual observations are empirically normalized. The frozen student sees
its original input scaling. Statistics are initialized from the starting batch,
then held fixed through an entire rollout and its PPO update; the exact
normalized features are stored in the rollout. Statistics update at the next
boundary and are frozen for evaluation. Mean, population variance, and count
are checkpointed. Standardized observations are clipped to `[-10,10]`.

## Defaults and reproduction limits

Paper-derived defaults: actor/critic `[512,256,128]`, ELU, latent std `0.10`,
learning rate `1e-3`, adaptive KL target `0.01`, PPO clip `0.2`, gamma `0.99`,
GAE lambda `0.95`, 5 epochs, 4 minibatches, entropy coefficient `0.005`, clipped
value loss coefficient `1`, gradient norm `1`, rollout24, seed42, 512 environments,
20,000 updates, save500. These are sourced from the supplied final manuscript
(main Eqs.22–24) and supplementary Tables2–3 / Note1 / Algorithm1.

For checkpoint compatibility the CLI retains the source simulator timing:
`0.002 s * 10 = 0.02 s` control interval. The paper lists `0.005 s * 4`.
These have equal control frequency but different physics substeps; changing
timing needs simulator validation. Existing motion tracking reward weights and
formulas also differ from the paper's table; this extension does not silently
replace them. Implement the paper's tracking reward in the construction task
before claiming experimental reproduction.

Missing research inputs include construction motion recordings, requested
support configurations, workspace geometry/encoding and its estimator, the
configuration sampling distribution, construction environment/reset logic,
and trained student/residual checkpoints. Adaptation is intended for simulation;
both policies stay fixed during physical evaluation.

## Validation

```bash
PYTHONPATH=rsl_rl python -m unittest discover -s tests -v
```

Tests cover numerical reward values, privilege isolation, frozen-student
invariance, exact initial deterministic behavior, radian bounds/action scaling,
latent likelihood replay, GAE terminal/timeout cases, complete minibatch
coverage, strict loading, checkpoint resume, and integration with the real
legacy student architecture in a synthetic vector environment.
