<!--
Copyright (C) 2026 Dexmate Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Which joint-position call do I use?

A plugin-managed component (in the shipped Vega profiles: the arms, the head
and the torso) accepts a position target over three paths. The verb says the
cadence, the return type says whether the robot reports back.

| Call | Where it goes | Returns | Guards | Call rate | Use for |
| --- | --- | --- | --- | --- | --- |
| `set_joint_pos(q, ...)` | joint controller directly; no planning, no smoothing | `None` | E-stop, freshness, joint limits, **step guard** (0.3 rad default) | any; one setpoint per call | host-side control loops that shape their own motion; small corrections |
| `stream_joint_pos(q, ...)` | motion plugin, **untracked**: the plugin smooths towards the newest goal with its own velocity, acceleration and jerk bounds | `None` | E-stop, freshness for relative goals, joint limits; refused while a tracked motion of the component is running | a loop, typically 50–500 Hz, one goal per tick | teleoperation, visual servoing, any loop that moves the goal every tick |
| `move_to_joint_pos(q, ...)`, `move_joint_trajectory(points, ...)`, `go_to_pose(name, ...)` | motion plugin, **tracked**: planned on the robot, status published per motion | `MotionHandle` | E-stop, freshness for relative goals, joint limits | discrete; one call per move | moves you may wait on, cancel or supervise |

Robot-level fan-outs take `{component: q}` and mirror the rows:
`robot.set_joint_pos`, `robot.stream_joint_pos`, `robot.move_to_joint_pos`
(returns a `MultiMotionHandle`). `robot.execute_trajectory` is the host-paced
counterpart of `set_joint_pos` for a whole timed trajectory.

A direct setpoint is checked against the measured position, not the previous
command. A loop that replays a precomputed trajectory with `set_joint_pos`
therefore works only while the arm keeps up with the host clock and fails the
moment it lags by the step ceiling; replay a trajectory with
`robot.execute_trajectory`, which checks waypoint spacing once up front, or
plan it on the robot with `move_joint_trajectory`.

Hands, grippers and the chassis are direct-only. They have no motion-plugin
endpoint, so streamed and planned operations are unsupported even when the
shared handle type exposes these methods. Use direct commands; Python
`robot.component_info(name)` and `robot.supports(name, capability)` expose
configured support. A direct setpoint on a `pv_mode` component (head, torso,
the DexS/DexD grippers) without an explicit `joint_vel` carries the model's
velocity limit towards the target, which the gripper firmware requires to move
at all; gripper commands also carry the configured grasp torque.

## Why three paths

Industrial controllers expose the same split. ABB's Externally Guided Motion
has a raw setpoint write (our direct path) and a controller-filtered mode
(our streamed path), and RAPID's `MoveAbsJ` is a planned move with completion
(our tracked path). ros2_control's command interfaces are streaming; MoveIt's
planned moves report completion. Collapsing the paths hides the one thing
that differs: whether anything comes back.

## Streamed goals in detail

- **No wait.** `stream_joint_pos` returns as soon as the goal is published.
  There is no `wait=` argument, no timeout and no handle, because the plugin
  publishes no status for an untracked goal. The caller's loop owns
  convergence.
- **No step guard.** The plugin bounds velocity, acceleration and jerk, so a
  far goal is a smooth ramp, not a jump. This is why a teleop loop should
  stream rather than call `set_joint_pos` with a raised `max_step`.
- **Best-effort delivery.** A dropped tick is repaired by the next one.
  Tracked motions use reliable delivery instead, because nothing follows a
  dropped one-shot.
- **Settling.** When the loop stops, the plugin holds the last goal. To know
  the joints have arrived, poll `is_joint_pos_reached(q)` or hand off with
  one `move_to_joint_pos(q, wait=True)`.
- **Mixing with tracked motions.** A streamed goal while a tracked motion is
  running is refused with the running motion's id. Wait for it or cancel it
  first. Admission is serialized within this connection; separate clients
  still require server-side arbitration.
- **On the wire.** A streamed goal is the same motion-target message as a
  tracked one with `motion_id = 0` and `cancel_motion_id = 0`, which the
  firmware's motion plugin documents as its untracked branch.

## Deprecated spellings

Each of these is a one-line wrapper over the canonical call. It emits a
`DeprecationWarning` naming the replacement. Removal is evaluated per API,
no earlier than 1.0, after a released warning period. The complete list of
deprecated spellings in every language is in [`API-PARITY.md`](API-PARITY.md).

| Deprecated | Use |
| --- | --- |
| `move_joint_pos(q, relative=, velocity_scale=)` (component and robot) | `stream_joint_pos(q, ...)` |
| `command_position(q, max_steps=)` | `set_joint_pos(q, max_steps=...)` |
| `set_joint_pos_vel(q, qd, ...)` | `set_joint_pos(q, joint_vel=qd, ...)` |
| `start_position(q)` | `move_to_joint_pos(q)` |

Python hides `DeprecationWarning` outside `__main__` by default. Library and
application code sees the warnings under pytest or with
`python -W default::DeprecationWarning`.

`dexcontrol.compat.v05` keeps the full 0.5 spelling set without warnings; its
`move_joint_pos` streams, as 0.5's did.

## Customer Rust SDK

The packaged `dexcontrol` crate uses `Robot::connect`, `MotionOptions`, and
synchronous methods. For example:

```rust,no_run
use dexcontrol::{ConnectOptions, MotionOptions, Robot};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let robot = Robot::connect(ConnectOptions::default())?;
let arm = robot.joints("left_arm")?;
arm.stream_joint_pos(&[0.0; 7], MotionOptions::default())?;
robot.close()?;
# Ok(()) }
```

Robot fan-outs are `set_joint_pos`, `stream_joint_pos`, and
`move_to_joint_pos`, the same names as on a component. Use `start_joint_motions` when per-component delivery
and cancellation reports are needed; a returned report does not mean success.
A tracked start (`move_to_joint_pos_with`, `go_to_pose_with`,
`move_joint_trajectory`, the `move_to_joint_pos` fan-out) fails with a
`MotionStartError` that retains the underlying `Error` and, when the command
may have reached the robot, the owning handle or group; `?` into `Error`
drops the handle. Branch on `error.kind()` rather than numeric error codes.

The native engine crate in the source workspace has the same crate name but a
different, async API (`Robot::builder`, `StreamOptions`, `DeliveryReport`). Its
examples are not drop-in examples for the packaged Rust facade.

## C and C++

`dex_component_stream_joint_pos(component, values, count, options, error)` and
`dex_robot_stream_joint_pos_multi(robot, names, values, counts, n, options,
error)` take the existing `dex_motion_options_t` (null for the defaults;
`wait_ceiling_ms` is ignored because there is no handle). They are additive
within ABI 5. The C++ wrapper exposes `JointComponent::stream_joint_pos(values,
options)` and `Robot::stream_joint_pos(targets, options)`.

## Completion, recovery and stopping

Prefer `wait_success` when later work requires a successfully completed move.
Python groups expose `status`, `wait_success`, and `cancel_and_wait`, matching
single handles. Existing `wait` behavior is preserved: Python/C++ return terminal
states, while the customer Rust facade's `wait` requires success. A timeout ends
observation; it does not cancel the motion.

Fan-out validation happens before delivery. Delivery is neither atomic nor
synchronized. Python failures carry `delivery_report`, and tracked partial
starts additionally carry `motion_handle` (the recovery group). The supported
`start_joint_motions` API returns `(group, report)` without raising for delivery
failures, so callers must inspect `all_delivered`. C callers must release a
non-null output group even on failure; C++ `MotionGroupStartError::group()` owns
that recovery reference.

Ending a streamed loop leaves its last goal in effect. After a streamed goal,
`stop()` sends a replacement goal at the fresh measured position through the
motion plugin, rather than relying on a raw controller hold that the planner
can overwrite. It attempts tracked cancellations too. Missing/stale feedback
or failed delivery is reported; a successful call means a stop request was
published, not that cancellation was acknowledged or the robot is measured
stationary. `Robot.stop_all()` and the heartbeat safety stop do the same for
every component whose plugin goal was last streamed, alongside cancelling
tracked motions. Publishing a tracked motion hands the plugin goal back to the
tracked path, and `stop()` returns to its usual policy (cancel, direct hold,
degrade on unusable feedback) until the next streamed goal. None of this
provides command expiry after process/network loss. Use the configured safety
supervision and validate the behavior on supported firmware before relying on
it in an application.
