# Running the full pipeline

Use `pipeline.py` in this folder. It prepends this copy's package directories so
an editable installation of the outer `Paper_1_try` does not silently select
older source. Install the packages from this folder too.

## Environment

For full simulation, follow the existing Linux/NVIDIA CUDA/Isaac Gym setup in
[upstream_exbody2.md](upstream_exbody2.md), then install:

```bash
pip install -e ./rsl_rl
pip install -e ./legged_gym
```

The inherited setup uses Python 3.8 / PyTorch 1.10 with Isaac Gym. New files use
compatible Python 3.8 syntax and intended compatible Torch APIs. Local CPU tests
use Python 3.11 / PyTorch 2.10; the legacy simulator runtime has not been run here.
CPU tests require only Torch and NumPy. Tests replace simulator/logging imports
where necessary and do not claim simulator coverage.

`python pipeline.py --help` works without Isaac Gym. The training entry point
imports Isaac Gym before Torch, as required by the inherited simulator build.
Default motion name `motions_dance_release.yaml` resolves through the included
cached `ASE/ase/poselib/data/pkl/motions_dance_release.pkl`. That example YAML is
not supplied. If regenerating the cache, you must provide its source motion
YAML and motion files. Use construction motions for research experiments; the
dance cache only exercises the inherited example task.

## Teacher

```bash
python pipeline.py teacher --run-name teacher --iterations 50000 --num-envs 512
```

This trains `ActorCriticMimicPriv` with the new paper motion reward and retained
upstream non-tracking regularization. The source privileged/history adaptation
mechanism is preserved. Collection and PPO now use the same effective actor
features and history-encoding mode, including when the estimator is enabled.
Raw observations remain available for supervised estimator and latent targets.

Output: `outputs/teacher/model_50000.pt`. The default iteration counts are
editable run budgets; they do not assert that these budgets reproduce results.

## Student

```bash
python pipeline.py student --run-name student --iterations 20000 \
  --teacher-checkpoint outputs/teacher/model_50000.pt
```

This initializes the frozen original actor/encoders from the teacher and trains
`ActorDistillPriv.student_actor_backbone` with action distillation on student
rollouts. Student learning is supervised; the paper reward is computed for
rollout evaluation, not substituted for the distillation loss.

To skip new teacher training for a smoke run, use
`--teacher-checkpoint checkpoints/reference_teacher.pt`. That file is the
supplied upstream dance teacher. It was not trained with this copy's new reward,
so using it is a warm-start experiment, not a newly reproduced paper teacher.

The final student file includes `student_actor_state_dict` and its optimizer.
Missing or incompatible student weights cannot silently produce a random
student. Teacher-only input is accepted only through `--teacher-checkpoint`.

## Residual adaptation

```bash
python pipeline.py adaptation --run-name adaptation --iterations 20000 \
  --student-checkpoint outputs/student/model_20000.pt \
  --construction-data data/construction.json \
  --robot-geometry data/robot_geometry.json
```

This runs the paper reward in `G1Paper` and trains `ResidualActorCritic`. The
student remains frozen and deterministic. Both residual actor and critic use
only deployable selected inputs, extra history and robot-relative support /
workspace context. Privileged teacher features and latent parameters are
excluded. The 3-value explicit-state slot remains zero as in the source task.

The stochastic variable stored for PPO is the latent Gaussian sample `u`:

```text
u ~ Normal(mu(obs), exp(log_std))
delta_q = bounds_rad * tanh(u)
action = student(obs, history) + delta_q / 0.25
```

The last residual mean layer starts at zero and latent standard deviation at
0.10. At evaluation use `u=mu(obs)`; the initial deterministic policy exactly
matches the student. Gaussian likelihoods use stored latent samples, without
inverting clipped/saturated actions. Normalization statistics are fixed through
collection and PPO, then updated at the iteration boundary; the student's own
input scaling is unchanged.

Per-joint bounds in radians: shoulders0.35, elbows0.25, wrists0.10, waist0.05,
hips/knees0.03, ankles0.02. The current G1 asset has 23 actuated joints and fixed
wrists. Unknown joint families and nonidentity action order fail explicitly.
Existing final action clipping and torque control remain in the simulator.

Without construction recordings, explicitly choose the diagnostic baseline:

```bash
python pipeline.py adaptation --run-name adaptation_motion --motion-only \
  --student-checkpoint outputs/student/model_20000.pt
```

This applies 0.2-weighted motion tracking, stability, slip and regularization;
support/workspace terms are zero and no synthetic measurements are inserted.
It is not construction-task adaptation.

## Resume and evaluate

```bash
python pipeline.py student --run-name student --iterations 1000 \
  --resume outputs/student/model_20000.pt

python pipeline.py adaptation --run-name adaptation --iterations 1000 \
  --resume outputs/adaptation/residual_20000.pt \
  --construction-data data/construction.json --robot-geometry data/robot_geometry.json

python pipeline.py adaptation --evaluate --render --num-envs 1 \
  --resume outputs/adaptation/residual_20000.pt \
  --construction-data data/construction.json --robot-geometry data/robot_geometry.json
```

For teacher/student evaluation, choose that stage and its `--resume` file.
For a motion-only residual checkpoint, specify `--motion-only` during resume or
evaluation as well. `--iterations` always means additional updates. Save interval
is 500 by default; every stage also saves its final checkpoint. A new run cannot
overwrite a nonempty output folder. When branching from an older checkpoint,
choose a new output folder to preserve later checkpoints.

Metadata checks include motion, joint order, reward configuration, simulator
step size, action scale, and construction/context hashes. Residual checkpoints
are self-contained and do not need the original student file. Teacher/student
checkpoints retain main, estimator/history, and student optimizer states where
applicable. Simulator state is not saved; resume restarts the environments and
does not promise bitwise replay. Residual resume also restores Torch RNG state;
teacher/student checkpoints do not capture simulator or external RNG state.

Outputs contain `pipeline_config.json`; residual output additionally contains
normalizer/config metadata and `metrics.jsonl` with each reward component.
Teacher/student report training losses and episode reward components to stdout.
The new launcher disables W&B logging by default.

## Remaining reproduction decisions

- Existing student architecture has a 10-step encoder history plus 25-step extra
  history. Preserving those dimensions is necessary for checkpoint compatibility.
- Physics retains source `0.002 s × 10` substeps (50 Hz control). The paper lists
  `0.005 s × 4`; equal control frequency does not imply equal physics behavior.
- The new task advances reference motion before reward and observations, and
  refreshes reference/local robot state on resets to avoid previous-episode input.
- Time-limit termination remains disabled as in the supplied G1 source. Genuine
  falls/collisions terminate. Custom truncations require pre-reset terminal
  observations; the residual runner rejects missing terminal bootstrap inputs.
- The teacher/student implementation retains the upstream asymmetric controller
  and losses; only residual observations use the newly added empirical normalizer.
- Workspace representation and feasible-configuration sampling are not specified
  fully in the paper. This adapter uses user-calibrated robot spheres, recorded
  human AABBs, uniform reset sampling, and robot-relative corner observations.
- Measured construction data, calibrated full-volume geometry, source construction
  motions and trained construction checkpoints remain external inputs.
