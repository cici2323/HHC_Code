# Review and fixes in this pipeline copy

This folder is a separate copy of `Paper_1_try`. The outer folder is unchanged
by this pipeline assembly. The following repairs support a connected three-stage
run; they preserve the original actor/student state-dictionary keys.

| Issue in the supplied source | Treatment in this copy |
| --- | --- |
| PPO updated a distribution with different estimator inputs/history mode from rollout collection | Store actor inputs separately from raw observations and preserve one history mode through each rollout/update |
| Student-load errors were caught and training could continue with random weights | Separate explicit teacher initialization from strict same-stage student resume; validate shapes and required optimizer state |
| Distillation optimizer and accurate iteration state were not saved | Save/restore all relevant optimizers, counters, adaptive learning rate and the next iteration; always save final checkpoints |
| Directly optimized Gaussian std could become nonpositive | Clamp inherited std after update/load while retaining its checkpoint key; residual uses log_std |
| References were advanced after reward/observation computation and could be stale after reset | New G1Paper task advances references first and refreshes reference/local robot state after reset |
| Residual construction context lacked current robot-relative geometry under randomized starts | Transform requested supports and workspace corners into the current robot-base frame |
| Reward helpers previously existed without construction task integration | New G1Paper task calls paper tracking/collaboration functions before reset and logs individual terms |
| Aggregate source reward clipping could weaken paper penalties | New task retains signed totals, applies 0.2 only to tracking and multiplies raw rewards by control dt once |
| Unused vision/GUI imports blocked CPU student loading | Removed the previously identified unused torchvision and turtle imports |

The tests run real teacher/student/residual network classes and production PPO
and runner code using small synthetic vector environments. The three-stage
handoff test trains, saves, loads and resumes through all stages. Other tests
check numerical reward values, geometry, privileged-input exclusion, frozen
weights, likelihood replay, reset ordering and CLI validation.

## Limits still applicable

- No full Isaac Gym or physical robot training was performed on this macOS host.
- Source scan / privileged branches of ActorDistillPriv remain unsuitable for
  nondefault use. The residual path enforces scan=0 and hist_encoding=True.
- Time-limit termination is disabled in the source G1 task. Residual custom
  truncations must supply actual pre-reset terminal features. Legacy teacher
  timeout bootstrapping remains an inherited approximation if timeouts are
  enabled independently; validate that extension before use.
- The new task preserves the source 19 selected demonstration DoFs and 12
  keypoints, 23 policy actions, observation layout and checkpoint architecture.
- The supplied reference teacher was trained upstream, with different reward
  definitions. It is provided as a warm-start/reference asset only.
- Primitive workspace volumes, uniform configuration sampling, frame conventions
  and Euler error serialization are explicit implementation choices where the
  paper does not specify a complete executable data/interface contract.
- Construction recordings, calibrated support offsets/full-volume geometry and
  trained student/adaptation checkpoints are not fabricated or supplied.

See [REWARDS.md](REWARDS.md), [CONSTRUCTION_DATA.md](CONSTRUCTION_DATA.md) and
[PIPELINE.md](PIPELINE.md) for the connected implementation and reproduction limits.
