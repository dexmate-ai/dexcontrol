<!--
Copyright (C) 2026 Dexmate Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Migrating from dexcontrol 0.5

For agent-assisted direct migration, give your coding agent
[these migration instructions](MIGRATING-FROM-0.5-AGENTS.md). This document is
the human guide: what changed, how to choose between pinning, the
compatibility module and a rewrite, how to carry out and validate the
migration, and the reference of known API changes to keep open while editing.

It is for anyone with Python code written against dexcontrol 0.5.x who needs
to run it on the native SDK. The compatibility adapter described here ships
from **dexcontrol 0.8.1** on; the 0.8.0 wheels do not have it. Install 0.8.1
or newer and verify with `python -c "import dexcontrol.compat.v05"`.

## What happened

dexcontrol 0.5 was a pure-Python package: connection, decoding, safety
monitoring and motion tracking all ran in the interpreter. dexcontrol 0.7
moved that core into a native (Rust) runtime that is shared by the Python,
C++ and Rust SDKs, and it changed the Python API at the same time. There was
no published 0.6 line or general migration period for the rewrite. The first
native prereleases were 0.7.7 release candidates; the published stable native
baseline for this guide is 0.8.1 (0.8.0, the first stable native release,
lacks the adapter and the fixes listed under 0.8.1 in `CHANGELOG.md`).

Three kinds of change matter to a 0.5 user:

1. **Names and shapes.** Modules were deleted, methods renamed, return
   shapes changed, sensors moved under `robot.sensors`.
2. **Behaviour.** The client now refuses commands 0.5 accepted: targets
   outside joint limits (0.5 clipped silently), steps exceeding the
   component-specific limit, commands while the E-stop is engaged, or joint
   state exceeding the configured freshness limit. **The process no longer
   exits on its own**: 0.5 called `os._exit(1)` when an E-stop button was
   pressed or the heartbeat timed out; 0.8 raises `EStopActiveError` on
   commands while stopped, runs the profile's heartbeat action (stop the
   robot, close the connection) and keeps your program running unless you
   opt in with `Robot(..., exit_on_safety_termination=True)`. No signal
   handlers are installed at import. Logging goes to loguru when it is
   installed and to the standard `logging` module otherwise. A watchdog
   process starts by default on real robots.
3. **Robot software.** The 0.8 client requires a robot SOC software build of
   1200 or newer and honours the server's advertised minimum client version.
   It does not follow the 0.5 README's "firmware minor version equals client
   minor version" rule.

## Which path

| Situation | Path |
| --- | --- |
| The robot must stay on software the 0.5 client supports | **Pin.** `dexcontrol>=0.5,<0.6`. Nothing else changes. The 0.5 line receives no new features. |
| Working application, limited time, robot software is current | **Compatibility module.** Update imports, then check the covered and refused calls below. Migration effort depends on the APIs used. |
| New feature work, or C++/Rust alongside Python | **Rewrite to the modern API.** Use the [reference tables](#what-changed-by-task) below and verify the changed behavior. |

Use `compat.v05.Robot.wrap(modern_robot)` to share an existing connection.
Closing that borrowed view releases its camera subscriptions without closing
the modern robot or interrupting other legacy views or modern managed camera
subscriptions; closed views cannot create new subscriptions. Independently
constructed robots do not automatically share a connection.

Confirm with Dexmate support which robot software your client version
supports before upgrading a deployed robot. A client upgrade on an
unsupported robot fails at connect with a version error; it does not move
the robot.

## The compatibility module: `dexcontrol.compat.v05`

`dexcontrol.compat.v05` is a thin Python wrapper over the current runtime
that presents the 0.5 surface: constructor and sensor-enable configuration,
`wait_time`/`exit_on_reach` waiting, `move_joint_pos` (streamed, as in 0.5), scalar joint getters,
the 0.5 state dictionary keys and motion-state names, 0.5 sensor names
(`head_imu`, `lidar`), camera getters, wrench and temperature sub-sensors,
and the `supported_models` decorator. It adds nothing to the modern API; the
two can be mixed over one robot with `compat.v05.Robot.wrap(robot)`.

For existing sensor-to-control loops, keep the supported 0.5 calls on that
adapter: camera getters return image arrays (or `None` before a frame),
`include_timestamp=True` preserves the legacy `data`/`timestamp` wrapper,
and joint getters preserve scalar/indexed results. `set_joint_pos`,
`set_joint_pos_vel`, chassis velocity commands, and `move_joint_pos` retain
their documented legacy call shapes; `move_to_joint_pos` returns a legacy
motion handle. Native safety checks still apply.

Legacy camera getters and modern `robot.sensors.subscribe_camera(...)`
contexts share subscription ownership on the same robot. Closing one view
does not interrupt another owner, and legacy reads preserve an existing
modern history policy. Only the last owner releases the stream. A stream
subscribed manually through a native handle remains the caller's
responsibility; do not manually unsubscribe a stream while other code uses it.

Regression tests exercise old-style sensor reads and control commands together
in simulation for `vega_1`, `vega_1p`, and `vega_1u`, using each profile's
available components. This validates SDK behavior, not hardware timing or
firmware compatibility.

### Step by step

1. **Install dexcontrol 0.8.1 or newer in a fresh environment**
   (`python -m pip install 'dexcontrol==0.8.1'`) and verify it with
   `python -c "import dexcontrol.compat.v05"`; the 0.8.0 wheel does not include
   the adapter. The wheel
   is a compiled extension; no Rust or C++ toolchain is needed. `dexbot-utils`, `dexcomm` and `jaxtyping` are no longer dependencies, and
   `loguru` is optional (`dexcontrol[examples]` installs it); if your code
   imports them directly, that is a separate change.
2. **Swap the imports.** Replace

   ```python
   from dexcontrol.robot import Robot
   from dexcontrol.core.config import get_robot_config
   from dexcontrol.utils.compat import supported_models
   ```

   with

   ```python
   from dexcontrol.compat.v05 import Robot, get_robot_config, supported_models
   ```

   Other `dexcontrol.core.*`, `dexcontrol.utils.*`, `dexcontrol.sensors.*`
   and `dexcontrol.apps.*` imports have no replacement; see
   [Imports that no longer exist](#imports-that-no-longer-exist).
3. **Run in simulation first.** `Robot(..., simulation=True)` connects to a
   deterministic in-process robot. Simulation covers many command and state
   paths, but it does not reproduce hardware timing or every firmware service.
   Some service queries fail in the mock, and cameras deliver no frames. Use
   populated fake-frame tests for image contracts and hardware validation for
   changed motion paths.
4. **Read the errors.** Every refused call raises with a message that names
   the replacement and this guide. The refused set is small and deliberate;
   the table under [What it refuses](#what-it-refuses) lists each call, why,
   and what to do instead. Also expect commands that 0.5 accepted to be
   rejected by the guards in "What happened" above. The exception says which
   guard fired. Fix the command, do not disable the guard.
5. **Check observable changes**, including different return formats and guard failures:

   | Symptom | Cause | Fix |
   | --- | --- | --- |
   | `systemctl stop` / `docker stop` no longer stops the robot | 0.5 installed SIGTERM/SIGHUP handlers at import | `compat.v05.Robot(auto_shutdown=True)` (the default) installs them for signals nothing else handles; the modern API needs `dexcontrol.install_signal_handlers()` |
   | no dexcontrol log lines, or they look different | records go to loguru when it is installed, else to the standard `logging` module; native verbosity is filtered by `DEXCONTROL_LOG` | keep loguru installed if your sinks expect it; `dexcontrol.configure_logging()` adds a styled stderr sink; `DEXCONTROL_LOG=dexcontrol=debug` for more |
   | the program keeps running after an E-stop press or a lost heartbeat | 0.5 exited the process from its monitor threads; 0.8 does not | `Robot(..., exit_on_safety_termination=True)` to opt back in, or react to `robot.wait_safety_event()` / `robot.health()`; supervisors that waited for the exit need one of the two |
   | `estop.deactivate()` takes up to 2 s or raises `DexTimeoutError` | it now waits until the robot's state reports the software E-stop released, so the next command is not refused | nothing, unless the robot never reports the release: then check its E-stop wiring and software |
   | the arm keeps moving after a `move_joint_pos` loop ends | a streamed goal stays in effect; the plugin converges on the last one | end the loop with `stop()`, a final goal, or one `move_to_joint_pos(q, wait=True)` |
   | `ValueError: tracked motion ... is still running` from `move_joint_pos` / `stream_joint_pos` | a streamed goal is refused while a planned move of the same component runs | wait on or cancel the motion handle before streaming |
   | `DeprecationWarning` for `move_joint_pos`, `set_joint_pos_vel` | modern-API aliases of `stream_joint_pos` and `set_joint_pos(joint_vel=)` | rename when convenient; adapter calls do not warn |
   | `ConfigurationError` naming `ROBOT_CONFIG` at connect | `ROBOT_NAME` or `ROBOT_CONFIG` holds a value the client does not recognise; 0.5 fell back to `vega_1` | set `ROBOT_CONFIG` to the deployment's profile name, or fix the identity |
   | a far target is rejected by `set_joint_pos` | configured step guard | `move_to_joint_pos` (planned on the robot) or a `wait_time` ramp |
   | `ValueError: ... exceeding max_step 0.3` part-way through a loop that replays a planned or recorded trajectory with `set_joint_pos` | each setpoint is checked against the *measured* position; the loop runs only while the arm keeps up with the host clock. 0.5 compared against the previous command | replace the loop, not the call: `robot.execute_trajectory({component: waypoints}, control_hz=...)` keeps the host timing and checks waypoint spacing up front; `arm.move_joint_trajectory([(q, t), ...], wait=True)` lets the robot own the timing. Do not pass `max_step=None` |
   | code expects a scalar from `arm.get_joint_pos(3)` | modern getter returns a 1-element array | the compatibility module returns scalars; modern code indexes `[0]` |
   | an extra `dexcontrol-watchdog` process appears | dedicated safety watchdog | expected; `watchdog=False` disables it |
   | the head does not move to "home" at connect | automatic head homing removed | call `robot.head.go_to_pose("home")` if you relied on it |
   | `Robot()` raises `ComponentError: component "right_arm" failed to start ... enable step timed out` | a connection enables the arms first; a motor that does not answer fails a required arm, where 0.5 logged and carried on | fix the arm (clear its error, reboot its board, check the joint the mode vector shows as `0`); recovery scripts connect with `Robot(initialize_arm_modes=False)` so they reach the arm |
   | `open_hand()` / `close_hand()` on a DexS/DexD gripper does nothing | the gripper firmware ignores a position command without a velocity; the 0.8.0 model gave the gripper a velocity limit of zero | use 0.8.1 or newer (its `robot_config()["robot"]["asset_version"]` is `dexmate-urdf 0.9.1`); it sends the model velocity (0.9 rad/s) and the profile's grasp torque, adjustable with `set_grasp_torque` |

6. **Validate on hardware** with the robot clear of people and the E-stop in
   reach. Run the application's normal start-up, one representative motion
   per component, one E-stop engagement during a motion followed by a
   release and the next command, the end of any streaming loop (the arm
   must stop where you expect), and a clean shutdown. Watch for guard
   rejections in the log: each one is a command that 0.5 would have sent and
   the current client did not.

### A complete example

This runs in simulation without a connected robot:

```python
from dexcontrol.compat.v05 import Robot, get_robot_config

configs = get_robot_config("vega_1")
configs.enable_sensor("head_camera")

with Robot(configs=configs, simulation=True) as robot:
    # Existing 0.5 getters retain their return shapes and key names.
    print("First arm joint:", robot.left_arm.get_joint_pos(0))  # scalar
    print("Head acceleration:", robot.sensors.head_imu.get_obs()["acc"])

    camera = robot.sensors.head_camera
    image = camera.get_left_rgb()  # subscribes lazily; ndarray or None
    if image is not None:
        print("Image shape:", image.shape)
    info = camera.get_camera_info()  # cached metadata/calibration, or None
    if info is not None:
        print("Camera dimensions:", info.get("actual"))

    target = robot.left_arm.get_joint_pos().copy()
    target[0] += 0.01  # radians
    robot.left_arm.set_joint_pos(target, wait_time=0.25)
```

Simulation supplies joint and IMU state, but no camera images or camera-info
service by default, so the camera getters above return `None`. On hardware,
select the appropriate profile and connection settings and use
`simulation=False`. A `Robot` connection can initialize motors; it is not a
read-only connection. The `with` block closes the connection and releases its
subscriptions. `camera.get_camera_info(force_refresh=True)` queries again; an
unavailable service returns `None`.

### What it refuses

Calls the adapter deliberately refuses, each with an error that names the
replacement:

| 0.5 call | Why | Do instead |
| --- | --- | --- |
| `Robot(configs=cfg)` with anything but `cfg.sensors[name].enabled` | the 0.5 config classes no longer exist | `Robot(profile=..., enable_sensors=[...])` or a `config_file` |
| `configs.sensors[name].transport = "rtc"` | RTC streams are off in the default build | use Zenoh streams |
| `query_hand_type()` | asked by the connection itself; a Vega's hands are adopted from the robot's answer | `robot.has_component("left_hand")`; the connect log's `end effectors:` line and `display_robot_info` show, per side, the hand in use and what the robot reported. `Robot(adopt_detected_hardware=False)` keeps the profile's hands as configured |
| `release_brake(enable)` without `joints` | `None` meant "all joints"; an arm can drop | `release_brake(enable, joints=[...])` or `release_all_brakes(enable)` |
| `set_idle_timeout`, `get_subscription_status`, `Robot.set_subscription_policy` | state subscriptions are always on | camera streams: `subscribe()`/`unsubscribe()` |
| `execute_trajectory(..., relative=True)` | 0.5 added each offset to that tick's measured position, so offsets compounded | add the offsets to one start position and pass absolute waypoints |
| `query_ntp(device="jetson")` | only the SOC clock is queried | `query_ntp(sample_count)` |
| `move_joint_trajectory(positions, time_from_start, ...)` | the current API takes point pairs and does not expose the old feedforward/tracking arguments | migrate explicitly using the recipe below |

Behaviour the adapter does not restore, under any flag: silent clipping of
targets to joint limits (out-of-range targets now raise), automatic head
homing at startup, implicit all-joint brake release, or any bypass of the
E-stop, state-freshness, step and limit guards. Commands 0.5 accepted may be
rejected now; the error says which guard fired.

## Mix old and new code on one connection

The compatibility module is a Python adapter, not a second installed SDK.
Use an explicit borrowed view to migrate one part of an application at a time:

```python
from dexcontrol import Robot
from dexcontrol.compat.v05 import Robot as LegacyRobot

with Robot(profile="vega_1", simulation=True, enable_sensors=["head_camera"]) as modern:
    with LegacyRobot.wrap(modern) as legacy:
        old_position = legacy.left_arm.get_joint_pos(0)  # scalar
        new_position = modern.left_arm.get_joint_pos(0)  # one-element array
        print(old_position, new_position[0])

        legacy.sensors.head_camera.get_left_rgb()  # lazy legacy subscription
        with modern.sensors.subscribe_camera("head_camera", "left_rgb") as camera:
            print(camera.latest_frame("left_rgb"))  # None in simulation

        target = legacy.left_arm.get_joint_pos().copy()
        target[0] += 0.01
        legacy.left_arm.set_joint_pos(target, wait_time=0.25)

    # The borrowed view is closed; the shared connection is still usable.
    print(modern.left_arm.get_joint_pos())
```

Keep legacy calls on `legacy` and migrated calls on `modern`. Both use the same
state, motion ownership, and safety guards; coordinate control of each component
rather than starting competing command loops. The adapter's `.native` property
also exposes the underlying modern robot or component when needed.

Legacy camera getters return arrays or `None`; modern frame getters return a
frame dictionary or `None`. Closing a legacy view releases only that view's
managed camera ownership. Other legacy views and modern `subscribe_camera`
contexts keep their subscriptions. Manual native `unsubscribe()` still stops
the shared stream, so do not call it while another reader needs that stream.

For camera calibration, the legacy `get_camera_info(force_refresh=False)`
caches successful replies and returns `None` on service failure. The modern
`get_camera_info(timeout=2.0)` queries on each call and raises on failure.
Neither requires an image subscription or invents missing calibration fields.

## Rewriting to the modern API

Work through the [reference tables](#what-changed-by-task) in order: connecting,
moving joints, hands and grippers, stopping and safety, reading joints,
sensors, queries and maintenance. The patterns that account for most edits:

- `set_joint_pos(q, wait_time=t)`: keep the adapter to preserve the arm's
  host-side ramp. For a modern planned move, use `move_to_joint_pos` and choose
  a completion timeout explicitly; `timeout` does not specify motion duration.
- `move_joint_pos(q)` → `stream_joint_pos(q)` (untracked, plugin-smoothed, as
  in 0.5; the old name still works with a `DeprecationWarning`). Not
  `move_to_joint_pos(q, wait=False)`: called every tick that replans each time
  and lags.
- `robot.sensors.<name>.get_obs()` on a camera → `camera.subscribe(stream)`
  once, then `camera.latest_frame(stream)["data"]` (the frame is a dict with
  `data`, `timestamp_ns` and `frame_id`, or `None` before the first image).
- `cfg = get_robot_config(); cfg.enable_sensor(n); Robot(configs=cfg)` →
  `Robot(enable_sensors=[n])`.
- `get_version_info()` → `robot.version_info`; `get_component_status()` → `robot.component_status()`.
- `get_joint_pos(i)` scalar → `[0]`; flat `get_joint_pos_dict([...])` → nested.
- `robot.set_joint_pos({...})`, `robot.move_joint_pos({...})`,
  `robot.move_to_joint_pos({...})` keep their dictionary forms
  (`robot.stream_joint_pos` for the streamed one). Validation is all-or-none;
  a delivery failure raises with `component` and `delivery_report` attributes.
- Code that reacted to an E-stop or heartbeat failure by letting the process
  die needs a new reaction (see "What happened"); the adapter does not bring
  the exit back.
- Gripper code that passed its own `vel`/`torque` or called
  `DexGripper.set_mode(...)`: `open_hand`/`close_hand` and `set_joint_pos`
  take the velocity from the model and the grasp torque from the profile
  (`set_grasp_torque` to change it); the 0.5 mode strings have no mapping.
- Maintenance scripts (`clear_error`, `reboot`, brake and E-stop
  tools) connect with `Robot(initialize_arm_modes=False)`, so an arm that
  cannot enable does not stop the script that is meant to fix it.

These movement replacements change semantics: a completion timeout is not a
requested motion duration, and expiration may raise rather than return normally.
Modern `execute_trajectory(relative=True)` uses a fixed starting position;
only the compatibility adapter refuses that option. See the reference before
changing old timing or relative commands.

Stopping changed shape too. `component.stop()` cancels that component's
planned motions and holds at the measured position; `robot.stop_all()` does it
for every component and zeroes the base. After `stream_joint_pos`, `stop()`
replaces the streamed goal through the motion plugin instead and raises
`StaleStateError` when no fresh measurement exists to hold at. A stop returns
when the request was published, not when the joints are measured stationary.
[`JOINT-CONTROL-PATHS.md`](JOINT-CONTROL-PATHS.md) has the full comparison of
the direct, streamed and planned paths; [`SAFETY.md`](SAFETY.md) and
[`ESTOP-WRITE-GUARD.md`](ESTOP-WRITE-GUARD.md) describe the E-stop and
heartbeat behaviour.

The modern API also offers what 0.5 did not: `DiagnosticClient` for read-only
tooling, `Robot.health()`, `wait_safety_event()` / `take_safety_events()`,
typed exceptions (`EStopActiveError`, `StaleStateError`, `MotionTimeoutError`,
`MotionStoppedError`), `RateLimiter` for control loops, and C++/Rust bindings.
Check the language-specific reference for differences in available methods,
return types, and waiting behavior.

## What changed, by task

The tables use the Python names. The same operations have the same names in
C++ and Rust; [`API-PARITY.md`](API-PARITY.md) is the generated table of every
operation across the four surfaces and of the spellings that still work but
warn.

### Connecting

| 0.5 | Now |
| --- | --- |
| `Robot()` resolves `ROBOT_NAME` | same, and an unknown `ROBOT_NAME` raises instead of defaulting to `vega_1` |
| `ROBOT_CONFIG` selects the model | same variable, same precedence over `ROBOT_NAME`; it takes a profile name (`vega_1p_gripper`) or a robot identity, and an unknown value raises instead of defaulting |
| `Robot(configs=cfg)` | `Robot(profile="vega_1_f5d6", enable_sensors=["head_camera"])` or `Robot(config_file=...)` |
| hand type detected at connect | still detected: a base profile (`vega_1`, `vega_1p`, `vega_1u`) adopts the hands the robot reports, a `_f5d6`/`_gripper` variant is corrected to them, and a side the robot reports empty is disabled. `adopt_detected_hardware=False` makes the profile authoritative; a `config_file` always is |
| sensors enabled via `cfg.sensors[name].enabled = True` | `enable_sensors=[...]`; nothing is enabled by default, as before |
| `auto_shutdown=True` installed SIGTERM/SIGHUP handlers | no handlers at import; call `dexcontrol.install_signal_handlers()` if nothing else handles them |
| loguru logging | records go to loguru when it is installed (your sinks keep working; `loguru` is now optional), otherwise to standard `logging`; `dexcontrol.configure_logging()` adds a styled stderr sink; `DEXCONTROL_LOG=dexcontrol=debug` sets native verbosity |
| in-process safety monitors | plus a watchdog child process on real robots (`watchdog=False` to disable) |
| arms set to position mode, head enabled and homed | arms set to position mode, head enabled, **no** home target sent |
| a failed arm mode change at startup was logged and the robot stayed usable | a required arm that cannot enable fails the connection with `ComponentError`; connect with `Robot(initialize_arm_modes=False)` to skip arm enabling, e.g. before `clear_errors()` |

### Moving joints

| 0.5 | Now |
| --- | --- |
| `arm.set_joint_pos(q, wait_time=3.0)` (host-side ramp) | `arm.move_to_joint_pos(q, wait=True, timeout=3.0)` (planned on the robot) |
| `head.set_joint_pos(q, wait_time=2.0, exit_on_reach=True)` | `head.set_joint_pos(q, wait=True, timeout=2.0, tolerance=0.05)` |
| `set_joint_pos(..., wait_kwargs=...)` | removed; `wait_kwargs` was never applied outside the arm |
| `move_joint_pos(q)` | `stream_joint_pos(q)`: the same untracked, plugin-smoothed goal 0.5 sent. `move_joint_pos` still works and warns; removal is evaluated per API, no earlier than 1.0. Do **not** substitute `move_to_joint_pos(q, wait=False)` in a loop: each call replans and the arm lags |
| `set_joint_pos_vel(q, qd)` | `set_joint_pos(q, joint_vel=qd)`; the old name still works and warns |
| `robot.set_joint_pos({name: q})`, `robot.move_joint_pos({name: q})`, `robot.move_to_joint_pos({name: q})` | same dictionary forms; `robot.stream_joint_pos({name: q})` is the streamed one (`robot.move_joint_pos` warns). Validation is all-or-none, delivery is per component and a failure raises with `component` and `delivery_report` attributes; `robot.start_joint_motions` returns `(group, report)` without raising |
| streamed goal while a planned move runs | refused (`ValueError` naming the running motion); wait on or cancel the handle first |
| `set_joint_pos` beyond a joint limit | 0.5 clipped to the limit silently; now rejected (`ValueError` naming the joint and its limit) before anything is published |
| `set_joint_pos` step guard | 0.5 refused a step over 0.5 rad; the default is now 0.3 rad per component (`max_step=` to change it) |
| `go_to_pose(name, timeout)` | unchanged; also `wait=False`, `velocity_scale` |
| `handle.wait()` returns `"finished"` etc. | unchanged names; `handle.status.state` is the `MotionState` enum |
| `except RuntimeError` around `wait()` | `except dexcontrol.MotionStoppedError`; timeouts raise `MotionTimeoutError`, still a `TimeoutError` |
| `chassis.set_velocity(vx, vy, wz, wait_time=2.0)` | `chassis.drive_for(vx, vy, wz, duration=2.0)` |
| `robot.execute_trajectory(traj, relative=True)` | modern API: offsets relative to one starting position; adapter: rejected because 0.5 used measured position at every tick |
| large single-step `set_joint_pos` jumps | rejected by the configured per-component step guard; use `move_to_joint_pos` |
| `for q in trajectory: arm.set_joint_pos(q); rate.sleep()` (replaying a planned or recorded trajectory) | `robot.execute_trajectory({"left_arm": qs, ...}, control_hz=hz)` for host-paced replay of every component at once, or `arm.move_joint_trajectory([(q, t), ...], wait=True)` for a robot-timed one. The old loop fails part-way on hardware: each setpoint is checked against the measured position, and the arm need only lag 0.3 rad once. 0.5 checked against the previous command |

[`docs/JOINT-CONTROL-PATHS.md`](JOINT-CONTROL-PATHS.md) is the one-table
summary of the three position paths (direct, streamed, planned), which guards
apply to each, and the deprecated spellings.

### Hands and grippers

| 0.5 | Now |
| --- | --- |
| `hand.open_hand(wait_time=...)`, `close_hand(...)` | unchanged, same arguments; an expired `wait_time` is a bounded wait, not an error |
| `DexGripper.set_joint_pos(q)` sent a hard-coded `vel=1.0` and `torque=0.2` with every command | the velocity comes from the robot model: gripper profiles use `pv_mode`, so a direct setpoint without `joint_vel` carries the model's velocity limit (0.9 rad/s, the speed the gripper firmware saturates at) towards the target; `joint_vel=` still overrides it |
| grasp torque fixed at 0.2 | the profile's `grasp_torque` (0.2) is sent with every command; `hand.set_grasp_torque(t)` changes it for the connection (clipped to `[0, 1]`, warned above 0.5 because a stalled gripper can damage its motor) and `hand.grasp_torque` reads it; `None` on a five-finger hand |
| `DexGripper.set_mode("mit" \| "velocity" \| "force")` | no equivalent by those names; `hand.set_mode(name)` speaks the firmware mode vocabulary (`hand.get_modes()` shows the current one). Record the intent as a pending decision rather than mapping the strings |

The 0.8.0 artifacts sent gripper commands without a velocity, and the gripper
firmware does not move on such a command. 0.8.1 carries the corrected model
(`dexcontrol.robot_config()["robot"]["asset_version"]` reports `dexmate-urdf
0.9.1` or newer; 0.8.0 reports `legacy-dexmate-urdf`). On 0.8.0, pass
`joint_vel=` explicitly or upgrade.

The movement rows are migration choices, not timing-equivalent substitutions.
Old `wait_time` bounds a streaming/holding period; modern `timeout` bounds a
completion wait and does not specify the movement duration. Old head/torso/hand
polling returns normally when its budget expires; modern convergence waiting
raises `MotionTimeoutError`. Chassis `drive_for` also has its own stopping
behavior. Check these differences against your application's intent.

For a positions-and-times managed trajectory, explicitly build modern point
pairs (when using the adapter, access the underlying component via `.native`):

```python
if len(positions) != len(time_from_start):
    raise ValueError("positions and time_from_start must have equal length")
points = [(list(q), float(t)) for q, t in zip(positions, time_from_start)]
handle = arm.native.move_joint_trajectory(points, wait=False)
```

On a modern component, call `arm.move_joint_trajectory` directly. The modern
method returns a tracked handle. Old `velocities`, `accelerations`, per-joint
`scale`, and `tracked` arguments cannot simply be forwarded or discarded;
review their intent before migrating. The adapter rejects this old method
with a migration hint rather than guessing a translation.

### Stopping and safety

| 0.5 | Now |
| --- | --- |
| E-stop button pressed → `os._exit(1)` from the monitor thread | the process keeps running; commands raise `EStopActiveError` while an E-stop is engaged, reads, `stop()`, `set_mode`/`set_modes`/`set_idle_mode` still work |
| heartbeat timeout → `os._exit(1)` | the profile's action runs (`request_process_termination`: stop motion, close the connection; later calls raise `RobotConnectionError`); the process exits only with `Robot(..., exit_on_safety_termination=True)` |
| `estop.activate()` / `deactivate()` return on the service reply | `deactivate()` also waits (up to 2 s) until the robot's state reports the software E-stop released, so the next command is not refused; raises `DexTimeoutError` if it never does |
| `estop.toggle()`, `show()`, `is_button_pressed()`, `is_software_estop_enabled()`, `heartbeat.pause()` / `resume()` | unchanged |
| `head.stop()`, `torso.stop()`, `hand.stop()`, `chassis.stop()` re-sent the current position (or zero velocity) once; the arm had none | every joint component has `stop()`: it cancels the component's planned motions and holds at the measured position; `robot.stop_all()` does it for every component plus a zero-velocity base command |
| a `move_joint_pos` loop ended and the arm settled on the last goal | same: a streamed goal stays in effect. End the loop with `stop()`, a final goal, or one `move_to_joint_pos(q, wait=True)`. After streaming, `stop()` sends a replacement streamed goal and raises `StaleStateError` without a fresh measurement |
| — | `robot.wait_safety_event(timeout)` / `take_safety_events()` / `health()` report what the safety layer did |

See [`SAFETY.md`](SAFETY.md) and [`ESTOP-WRITE-GUARD.md`](ESTOP-WRITE-GUARD.md).

### Reading joints

| 0.5 | Now |
| --- | --- |
| `get_joint_pos(3)` returns a scalar | returns a 1-element array; index `[0]` |
| `get_state()` keys `pos`, `vel`, `cur`, `torque`, `error` | `position`, `velocity`, `current`, `torque`, `error_joint_indices` |
| `get_joint_state(i)` shape `(3,)`, torque when no current | shape `(1, 3)`; 0.8.1 restores the current-or-torque fallback and raises on a missing measurement, where 0.8.0 zero-filled missing current |
| `robot.get_joint_pos_dict([...])` flat | nested per component |

The restored combined-state fallback is in 0.8.1 and newer; 0.8.0 zero-filled
missing current. On 0.8.0, select current or torque from raw state explicitly
instead of relying on the combined getter.

### Sensors

| 0.5 | Now |
| --- | --- |
| `robot.sensors.head_imu` | `robot.sensors.get_imu("head_camera")` (enable `head_camera`) |
| `robot.sensors.lidar` | `robot.sensors.lidar_2d_front` |
| IMU `get_obs()` keys `ang_vel` (ZED) / `gyro` (chassis) | `gyro` everywhere |
| `camera.get_obs()`, `get_left_rgb()`, `get_depth()` | `camera.subscribe("left_rgb")` once, then `camera.latest_frame("left_rgb")["data"]`; the frame is `{data, timestamp_ns, frame_id}` or `None` before the first image |
| `camera.height`, `camera.width` | `camera.frame_shape(stream)` after the first frame |
| `camera.get_camera_info(force_refresh=False)` | `camera.get_camera_info(timeout=2.0)` queries fresh server metadata/calibration; the adapter retains caching and returns `None` on service failure |
| `arm.wrench_sensor.get_wrench_state()` | `robot.sensors.get_wrench("left_arm").get_wrench_state()` |
| `arm.wrench_sensor.get_button_state()` | `arm.get_button_state()` |
| `torso.temperature_sensor` | `robot.sensors.get_temperature("torso")` |
| — | `robot.sensors.get_temperature("head_camera")` reads the ZED camera's own temperature (also the wrist cameras); it is not a camera stream and does not appear in `camera.streams` |
| `RTC` camera transport | not in the default build |

Camera compatibility getters return NumPy images. With `include_timestamp=True`,
they return the historical `data` and `timestamp` fields using the frame's
nanosecond timestamp. Dimensions depend on received frames; calibration is not
reconstructed. `base_imu` means `chassis_imu`, never a camera or LiDAR IMU.
Chassis steer/drive temperature readers are not currently exposed by the
native temperature reader; the adapter raises `SensorNotAvailableError` for
those properties. Torso and arm temperature access is available.

### Queries and maintenance

| 0.5 | Now |
| --- | --- |
| `get_version_info(show=True)` | `robot.version_info` (property) |
| `get_component_status(show=True)` | `robot.component_status()` |
| `query_hand_type()` | detected at connect and adopted; see above |
| `query_ntp(n, show, device)` | `query_ntp(n)` |
| `clear_error(component)`, `reboot_component(board)` | `clear_error` unchanged; `reboot_component` is now `reboot(board)` (the old name still works with a `DeprecationWarning`, and the compatibility module keeps it silent); plus `clear_errors()` for every clearable component. Connect the robot you run them on with `Robot(initialize_arm_modes=False)`: the default connection enables the arms first and fails on an arm whose motor cannot enable, which is usually the arm you are trying to recover. 0.5 logged that failure and carried on |
| `RobotQueryInterface.create()` | `dexcontrol.DiagnosticClient` (read-only, no safety setup, cannot call `clear_error` or `reboot`) |
| — | `dexcontrol.robot_config()["robot"]["asset_version"]` names the robot-model release the build embeds (`dexmate-urdf 0.9.1`); it is also in `robot.resolved_config()` |

### Imports that no longer exist

`dexcontrol.core.*`, `dexcontrol.sensors.camera/imu/lidar/ultrasonic`,
`dexcontrol.utils.*`, `dexcontrol.apps.*`, `dexcontrol.comm`,
`dexcontrol.config`. The 0.5 component classes (`Arm`, `Head`, `Torso`,
`Hand`, `Chassis`) exist only in `dexcontrol.compat.v05`; the modern API has
one `JointComponent` type. All 0.5 exception classes keep their names in
`dexcontrol.exceptions`.

## The 0.5 examples

Use `examples/RENAMED.md` to locate current examples. Import-only migration is
not yet certified for the entire old example tree. Check each application
against this guide, especially sensor frame formats, motion timing, profile
selection, and direct imports of dependencies no longer bundled with DexControl.

Examples that add sensor configuration objects, select RTC, query hand hardware,
release unspecified brakes, or use removed utility/teleop modules need additional
changes. Prefer their current examples or an explicit recipe. Passing the
adapter tests is not proof that every old example runs unchanged.
## Support status

Supported adapter calls, including the arm's `wait_time` ramp, emit no
deprecation warnings. No removal date is set. On the modern API, every
deprecated spelling and its replacement is listed in
[`API-PARITY.md`](API-PARITY.md); all of them are removed at 1.0. If an
API in it is ever scheduled for removal, a release will warn first and name
the replacement and the release in which the removal happens. The deprecated
aliases on the modern API (`move_joint_pos`, `set_joint_pos_vel`,
`command_position`, `start_position`) are a separate matter: they warn now,
and removal is evaluated per API, no earlier than 1.0.

Report gaps as issues with the 0.5 call, the error text, and a simulation
reproduction where possible.

## Frequently asked

**Can I keep 0.5 and 0.8 installed side by side?** Not in one environment;
they are the same distribution name. Use separate virtual environments.

**Does the compatibility module slow anything down?** It is a thin Python
wrapper over the same native handles, but conversion and Python scheduling
add work; no unchanged-latency guarantee is made. The arm `wait_time` ramp
runs on your computer, as it did in 0.5.

**My robot is on old software and I need a new 0.8 feature.** The client
cannot bridge an unsupported server protocol. Verify the supported combination
with Dexmate support; an API migration alone does not necessarily require a
robot-software upgrade.

**Will RTC camera streams come back?** Not in the default build. Zenoh
streams depend on the camera and robot profile; not every camera supplies depth.
