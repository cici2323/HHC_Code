# Paper reward implementation

The new `G1Paper` environment calls the paper functions directly during reward
computation, before resets. `pipeline.py` loads all coefficients from
[`configs/rewards.json`](../configs/rewards.json). These are connected training
rewards, not unused helper functions.

## Motion tracking

| Term | Weighted reward |
| --- | --- |
| Joint positions | `5 exp(-0.25 ||q-q_ref||_1)` |
| Keypoint positions | `4 exp(-0.25 ||p-p_ref||_2)` |
| Root linear velocity | `6 exp(-4 ||v-v_ref||_2)` |
| Root angular velocity | `3 exp(-2 ||omega-omega_ref||_2)` |
| Heading | `3 exp(-2 abs(wrap(yaw-yaw_ref)))` |
| Root height | `2 exp(-10 abs(z-z_ref))` |
| Body rotation | `3 exp(-2 ||wrap(rpy-rpy_ref)||_2)` |

Implementation: `rsl_rl/rsl_rl/residual/motion_rewards.py`. Keypoints are flattened
per environment for the L2 norm; these are unsquared norms. Angles are wrapped
to the shortest signed difference at ±pi. Joint positions use the stated L1
error without wrapping. Body rotation uses Euler roll/pitch/yaw, with a separate
heading term retained. This serialization makes the paper's rotation notation
explicit; it does not claim a geodesic quaternion distance.

Velocity targets use the source's local-root velocity convention. Keypoints
share world coordinates with horizontally aligned references, preserving the
source motion alignment. Height includes each simulation environment's origin.
No reference-velocity tensor is modified in place for stand-still commands.

Perfect motion tracking totals **26 before regularization and dt scaling**.
The teacher uses `tracking + regularization`; the student sees the same reward
for rollout reporting while optimizing its distillation objective.

## Adaptation

```text
r = 0.2 * tracking + support + workspace + stability + slip + regularization
support   = 5 * exp(-mean_i((relu(||p_i-p_i*||_2 - 0.02)/0.05)^2))
workspace = -3 * indicator(robot occupancy intersects human occupancy)
stability = 1.5 * exp(-||base_angular_velocity||_2)
slip      = -0.05 * sum_feet(contact * ||foot_velocity||_2^2)
```

There are two support points in metres; 2 cm is a plateau tolerance, not a hard
constraint. Contact uses positive normal force. Foot velocity is read from the
rigid-body linear velocity components, and contact flags select the slipping
feet. The overlap indicator comes from configured robot spheres and recorded
human boxes. See [the data guide](CONSTRUCTION_DATA.md) for this approximation.

Implementation: `residual/rewards.py` provides analytical collaboration terms;
`residual/task_rewards.py` connects them to the simulator. In motion-only
adaptation, support/workspace terms are explicitly omitted; stability and slip
still apply. No baseline is presented as construction training.

## Regularization, scaling and clipping

The original non-tracking terms are retained separately: alive, orientation,
joint acceleration, action rate, joint error, stumbling, joint limits, airtime,
foot force, ankle action and waist roll/pitch error. Their original weights are
listed explicitly in the JSON file, including positive alive/airtime terms.
These are inherited implementation details where the paper delegates
regularization to the original controller; they are not newly estimated paper
parameters.

The regularization functions receive the source's `weight * dt`. Motion and
collaboration helpers return raw rewards and receive one `dt` multiplication in
the mixin. The factor 0.2 applies only to tracking. There is no final positive
clamp or inherited `-0.5` floor in the new paper task, so configured workspace
and regularization penalties are preserved. The old `g1_mimic_*` tasks remain
available for comparison with their original reward functions.

Change coefficients in a copy of the JSON and pass `--reward-config path.json`.
Resume rejects a different saved reward configuration; start a new experiment
when changing the objective.
