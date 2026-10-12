# Construction context and occupied-volume geometry

No construction training data is fabricated in this folder. The adapter is
implemented and requires measured or deliberately designed configurations.
Schemas are supplied for preparing the two inputs.

## Recorded collaboration configurations

`--construction-data` points to JSON with:

```text
schema_version: 1
frame: "environment"
samples: a nonempty list of configurations
  support_points: two [x,y,z] target points, in metres
  human_workspace_boxes: B boxes, each [[min_x,min_y,min_z],[max_x,max_y,max_z]]
```

See [data/construction.schema.json](../data/construction.schema.json). Every
sample must use the same B>=1. Boxes have positive extent and all numbers must
be finite. Each sample is expressed in a common environment frame with the
robot initialization and motion setup. The simulator translates that frame by
its environment origin to compute rewards. Sampling is uniform over recorded
configurations at resets using the seeded Torch generator. Sampling weights,
dynamic human trajectories, configuration feasibility checks and estimator
training are not invented here.

## Robot support points and occupied volume

`--robot-geometry` points to JSON with:

```text
schema_version: 1
support_points: exactly two records
  body: retained rigid-body name in the simulated robot
  offset: [x,y,z] support/contact point in that body's local frame, metres
robot_spheres: a nonempty list of records
  body: retained rigid-body name
  offset: [x,y,z] local sphere centre, metres
  radius: positive radius, metres
```

See [configs/robot_geometry.schema.json](../configs/robot_geometry.schema.json).
The G1 source asset includes `left_rubber_hand` and `right_rubber_hand`, but
local contact offsets and sphere radii must be calibrated for your application.
Fixed-joint collapsing can change available names; unknown body names fail.
Cover the intended full occupied robot volume with the configured spheres.
The code cannot verify that your set of spheres adequately covers the robot.

At every reward step, support points and sphere centres are rotated by each
body's current quaternion and translated to world position. Overlap is true if
any configured sphere intersects any human box, including tangency. This tests
the selected primitive volumes exactly; it approximates the physical occupied
volume. It is not robot mesh collision checking or a validated human-safety
system. The paper does not provide the geometry implementation or calibration,
so this choice is recorded explicitly.

## What the residual policy observes

The dataset loader can return environment-frame coordinates for data inspection.
The simulator adapter instead builds **robot-relative** context:

```text
2 support positions: 6 values
8 corners of each human box: 24 * B values
context width: 6 + 24 * B
```

The corner order uses lexicographic min/max bits: 000,001,010,011,100,101,110,111.
Every point is transformed by the inverse robot-base orientation after
subtracting the base position. Eight corners retain box orientation under this
transform. Supplying relative geometry avoids identical inputs requiring
opposite corrections when the initial robot position changes.

Simulation uses state to construct these relative measurements. Deployment
must supply equivalent relative targets/workspace corners from calibrated
robot–human perception/localization; the code does not supply that estimator.
Only the relative positions/corners enter the residual actor, not simulator
root pose or hidden teacher state. Requested support-point order, workspace
order, units, and body frame must match training. No policy updates occur during
formal physical evaluation.

Checkpoint metadata records the observation schema plus SHA-256 hashes of the
recorded data and robot geometry. Resuming with changed content is rejected.
Changing the data or volume approximation should be treated as a new experiment.
