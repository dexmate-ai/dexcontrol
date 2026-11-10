<!--
Copyright (C) 2026 Dexmate Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Agent instructions: migrate DexControl 0.5 applications to 0.8

This guide is for Codex, Claude Code, and other coding agents working in a
customer repository. Follow the customer's requested migration strategy:

- **Compatibility mode:** retain supported 0.5 sensor and control calls through
  `dexcontrol.compat.v05`. Choose this when the customer asks to keep old API
  behavior or minimize application changes.
- **Direct migration:** use the modern `dexcontrol` API and update result
  consumers and behavior explicitly. Use this as the default for an otherwise
  unspecified migration request.
- **Gradual migration:** keep explicitly identified legacy paths on a borrowed
  compatibility view while migrating other paths over the same connection.

An import swap is not completion of a requested direct migration. Conversely,
do not rewrite supported legacy calls merely because modern replacements exist
when the customer chose compatibility mode. If their stated requirements
conflict, clarify the conflicting behavior before changing it; a clear request
to preserve old calls already selects the compatibility strategy.

Scope: Python code written against the public 0.5 API, targeting the native
0.8 API. These are not C++ or Rust migration instructions. Use the installed
SDK's signatures and public examples to resolve details; do not invent methods.
See the [API migration reference](MIGRATING-FROM-0.5.md) and
[human guide](MIGRATING-FROM-0.5.md) for additional context.

## Customer prompt

Give your agent this file and choose the task matching your intent.

For compatibility mode:

> Read MIGRATING-FROM-0.5-AGENTS.md and follow its compatibility path in this
> repository. Run our existing Python integration on the updated SDK while
> retaining supported 0.5 sensor getters, control commands, result shapes, and
> timing behavior. Update imports/configuration where needed and identify
> unsupported calls rather than silently rewriting their behavior. Share one
> connection if old and modern code coexist. Test with fakes or explicit
> simulation only; do not connect to hardware. Report the exact target build,
> changes, validation, and remaining gaps.

For direct migration:

> Read MIGRATING-FROM-0.5-AGENTS.md and follow it in this repository. Migrate
> our existing DexControl integration to the modern API directly. Inspect
> application call sites and tests before editing. Make the straightforward
> changes, ask about ambiguous motion or hardware behavior, and continue
> independent work while waiting. Test with fakes or explicit simulation;
> do not connect to real hardware. Finish with changed files, tests run,
> intentional behavior changes, and any remaining migration blockers.

The instructions below apply to the migration task; they do not override the
customer's repository instructions or authorize deployment, firmware changes,
or hardware operation.

## 0. Material the agent needs in the working tree

Use documentation and examples from the SDK export matching the target build
(the public source revision or release tag identified by the SDK release).
Older releases may not include this guide. Obtain the historical
0.5 sources separately when needed. Before starting, make these available next
to the customer code or give the agent their paths:

- this file, [`MIGRATING-FROM-0.5.md`](MIGRATING-FROM-0.5.md) (per-API
  reference) and [`MIGRATING-FROM-0.5.md`](MIGRATING-FROM-0.5.md);
- the SDK's `examples/vega_1/python/` tree and `examples/RENAMED.md`: use the
  rename map to find relevant current examples. Some old examples were removed,
  consolidated, or substantially changed; pairs illustrate migration patterns
  but do not prove behavioral equivalence. Read the old `examples/` and
  `src/dexcontrol/` from the public 0.5 source corresponding to the customer's
  dependency when the intent of an old call is unclear;
- the Safety section of the SDK README for what the guards do,
  [`SAFETY.md`](SAFETY.md) and [`ESTOP-WRITE-GUARD.md`](ESTOP-WRITE-GUARD.md)
  for what happens on an E-stop or a lost heartbeat (0.5 and 0.8 differ, see
  section 4), [`JOINT-CONTROL-PATHS.md`](JOINT-CONTROL-PATHS.md) for the
  direct / streamed / planned motion paths in one table, and
  `examples/BEHAVIOR.md` for what each current example does on a robot;
- `dexcontrol/_native.pyi` beside the installed package, or
  `python/dexcontrol/_native.pyi` in the matching SDK source export, for signatures.

Useful example pairs to inspect (verify behavior rather than copying blindly): `move_arm.py` → `cycle_arm.py`
(direct and planned joint moves), `move_torso.py` → `cycle_torso.py`
(`supported_models`, `go_to_pose`, `wait_time`), `get_head_zed_x_mini_data.py`
→ `monitor_head_camera.py` (sensor enabling, camera subscription, frames),
`get_imu_data.py` → `read_imu_states.py` (sensor names, IMU keys),
`display_robot_info.py` → `display_robot_info.py` (diagnostics),
`chassis_head_teleop.py` → `gamepad_chassis_head.py` (control loop with
`RateLimiter`, chassis velocity, head setpoints).

## 1. Establish the installed target without connecting

Inspect the repository's environment/lockfile instructions first. Preserve
unrelated work. Use a branch or separate environment according to the project's
workflow; do not upgrade a shared deployment environment as an audit step.

Run this with the interpreter intended for the migrated application. It imports
only the SDK, not customer modules that may connect at import time:

```python
from importlib.metadata import version
import dexcontrol

print("distribution:", version("dexcontrol"))
print("module:", dexcontrol.__file__)
if hasattr(dexcontrol, "build_info"):
    print("native build:", dexcontrol.build_info())
print("profiles:", dexcontrol.available_profiles())
print("robot model:", dexcontrol.robot_config("vega_1")["robot"]["asset_version"])
```

`asset_version` names the robot-model release the native build embeds
(`dexmate-urdf 0.9.1` or newer). A build that prints `legacy-dexmate-urdf`
predates the gripper model fix: on it, DexS/DexD grippers do not move on
`open_hand`/`close_hand` or `set_joint_pos` without an explicit `joint_vel`.
Record it with the build provenance.

Keep build provenance locally with the test results; redact local paths before
sharing reports publicly. Do not print credentials or dump the environment.

Version caveat: `dexcontrol.compat.v05` and the restored current-or-torque
fallback with missing-measurement errors for combined joint-state getters ship
from **0.8.1** on. The 0.8.0 wheels have neither: they lack the adapter and
zero-filled missing current. Confirm `dexcontrol.__version__` is 0.8.1 or newer
before relying on the restored contract; importing the adapter alone does not
prove the native binary contains this change.

Direct migration does not require the adapter. For compatibility or gradual
migration, verify `from dexcontrol.compat.v05 import Robot, get_robot_config`
against the application's actual interpreter. There is no compatibility
environment flag; the import selects the API. Original 0.5 and modern releases
cannot be installed side by side in one environment under the same package name.
Do not silently fall back to an older package or an unapproved download source.

Check each required feature against the same build. For example, an older
adapter build may import successfully while lacking camera information or shared
camera subscription ownership. Inspect the installed `CameraSensor` stub for
`get_camera_info(timeout=2.0)` and verify used behavior offline; the presence of
an import or method alone is not proof of complete compatibility.

`docs/API-PARITY.md` in the SDK lists the canonical name of every operation in
Python, C++, Rust and C and every deprecated alias; migrated code uses the
canonical names, never an alias that merely still works.

Inspect `dexcontrol/_native.pyi`, `help(...)`, and public examples supplied with
the target SDK. If a native signature is not available through
`inspect.signature`, consult its stub/docstring instead of guessing. If the
stub, documentation, and observed behavior disagree, inspect the implementation
for that exact build and run a targeted offline test where possible. Record
unresolved discrepancies rather than treating this guide as more authoritative
than the installed implementation. For example, some older stubs annotated
`Robot.get_joint_pos_dict` as taking only one name, while its implementation
also accepted a list of component names. A simulation test with complete mock
state does not establish behavior when measurements are absent.

## 2. Inventory actual usage; do not run old entry points yet

Search application code, scripts, notebooks, configuration, and dependency files.
A starting search (adapt exclusions and paths to the repository):

```bash
rg -n 'dexcontrol|dexbot_utils|dexcomm' . -g '*.py' -g '*.ipynb' -g '*.toml' -g '*requirements*.txt'
rg -n 'wait_time|wait_kwargs|exit_on_reach|relative=|release_brake|move_joint_trajectory|query_hand_type|get_camera_info' . -g '*.py'
rg -n 'head_imu|base_imu|wrench_sensor|temperature_sensor|set_subscription_policy|get_subscription_status' . -g '*.py'
rg -n 'move_joint_pos|set_joint_pos_vel|command_position|start_position|start_joint_motions' . -g '*.py'
rg -n 'estop|heartbeat|os\._exit|SIGTERM|atexit' . -g '*.py'
rg -n 'open_hand|close_hand|DexGripper|set_mode\(|clear_error|reboot_component' . -g '*.py'
```

The `estop|heartbeat` line finds code that assumed 0.5's process-exit
behavior on an E-stop or heartbeat failure (section 4); such code usually has
to change on every strategy, including compatibility mode. The last line finds
gripper code and maintenance scripts, which have their own rules in sections 3
and 4.

Treat every `set_joint_pos` call inside a loop that is paced by a
`RateLimiter`, `time.sleep` or a fixed `control_hz` as a **trajectory replay
loop**, even when the call itself needs no edit. It is the one 0.5 pattern
that still imports, still runs, and then fails part-way through on hardware;
section 4 has the rule and the recipe.

Inspect aliases, helper wrappers, and consumers of results, not just matching
lines. A dictionary-key rename may break a serializer or downstream controller.
Read each test's setup before running it: `Robot()` is a real connection unless
simulation is explicitly selected. Importing a module can execute that call.

Maintain a short checklist in the task notes:

| Location | Current use | Selected API/view | Behavior decision | Verification |
| --- | --- | --- | --- | --- |
| application file/function | signature and consumed result | exact installed API | equivalent, intentional change, or pending | test or manual review |

Record the application's intended robot profile, sensor/stream requirements,
motion timing, cancellation policy, and lifecycle. Ask only for information that
cannot be established from code or project configuration. An SDK version gate
is not a substitute for a verified robot-software compatibility matrix.

## 2a. Compatibility mode and gradual migration

Use this path when the customer chose compatibility or retaining old calls.
Sections 3 and 4 describe **direct migration**: consult them for explicit gaps
and selected modern paths, not as automatic rewrites of adapter code. Then
complete sections 5 and 6 for the selected strategy.

1. Replace old imports with:

   ```python
   from dexcontrol.compat.v05 import Robot, get_robot_config, supported_models
   ```

   Import only what the application uses. Obtain `configs` from this
   `get_robot_config`, not old configuration classes. Keep the application's
   profile and sensor-enable choices. With `configs=`, do not also pass
   `profile=` or `enable_sensors=`. Other configuration mutations require
   explicit migration; do not silently discard them.
2. Preserve supported calls and their consumers together. In particular:

   | Legacy call through the adapter | Preserve |
   | --- | --- |
   | `get_joint_pos(i)` / `get_joint_state(i)` | Scalar / one joint's row; do not apply modern `[0]` rewrites. |
   | Camera `get_left_rgb()` / `get_rgb()` / `get_depth()` | Lazy subscription, image array or `None`. |
   | Camera `get_obs(..., include_timestamp=True)` | Legacy `data` / `timestamp` wrapper and nanosecond timestamp. |
   | `sensors.head_imu.get_obs()` | Legacy `ang_vel` key; do not rename to `gyro` on this path. |
   | `set_joint_pos` / `set_joint_pos_vel` | Supported legacy signatures, relative targets, and velocity handling. |
   | Arm `set_joint_pos(..., wait_time=t)` | Host-side ramp/holding semantics; do not replace `wait_time` with `timeout`. |
   | `move_joint_pos(q)` | Untracked streamed goal, returning `None`. |
   | `move_to_joint_pos(q)` | Legacy motion handle and status vocabulary; successful `wait` returns `"finished"`. |
   | Chassis timed velocity calls | Their legacy repetition and interruption behavior; do not automatically replace with `drive_for`. |
   | `get_camera_info(force_refresh=False)` | Cached successful reply; `None` on service failure. A failed refresh preserves the cache for the next non-refresh read. |

   Use the API reference for exact signatures and unsupported calls. Native
   guards still apply: compatibility does not restore silent limit clipping,
   automatic head homing, or safety bypasses. Sensor names must resolve in the
   configured profile; `base_imu` is not a substitute for any available IMU.
3. For mixed code, share one modern connection using `LegacyRobot.wrap`.
   Keep modern handles on modern paths and adapter handles on legacy paths:

   ```python
   from dexcontrol import Robot
   from dexcontrol.compat.v05 import Robot as LegacyRobot

   # Demonstration only: retain the application's profile in migrated code.
   with Robot(profile="vega_1", simulation=True,
              enable_sensors=["head_camera"]) as modern:
       with LegacyRobot.wrap(modern) as legacy:
           print(legacy.left_arm.get_joint_pos(0))       # scalar
           print(modern.left_arm.get_joint_pos(0)[0])    # array element
           legacy.sensors.head_camera.get_left_rgb()    # lazy subscription
           with modern.sensors.subscribe_camera("head_camera", "left_rgb") as camera:
               print(camera.latest_frame("left_rgb"))  # None in simulation
       print(modern.left_arm.get_joint_pos())            # connection still open
   ```

   `.native` also exposes the underlying modern robot/component when needed.
   Do not construct two Robot instances expecting shared state or ownership.
   Coordinate command producers for each component; sharing a connection is
   not permission to run competing control loops.
4. Preserve cleanup ownership. An adapter-created robot owns its connection;
   `wrap` borrows it. Closing the borrowed view releases that view's camera
   ownership without closing the modern robot. Legacy camera getters and
   modern `subscribe_camera` contexts share leases, preserve an existing
   buffering policy, and release the stream after the final owner exits.
   Manual `unsubscribe()` still stops the shared stream; never call it while
   other readers need that stream. Closed legacy views cannot resubscribe.
5. Keep the support policy accurate. Supported adapter calls emit no
   deprecation warnings and have **no scheduled removal**; do not describe the
   adapter as expiring automatically at 1.0. Deprecated aliases on modern
   objects are separate: removal is evaluated individually, no earlier than
   1.0, after a released warning period.

The [human guide](MIGRATING-FROM-0.5.md#mix-old-and-new-code-on-one-connection)
contains another shared-connection example. Do not build a second compatibility
framework or monkey-patch modern classes in the customer application.

## 3. Direct migration: apply straightforward changes first

### Imports, configuration, and lifetime

- Prefer `from dexcontrol import Robot` for direct migration. Obtain components
  from the robot instead of constructing old `dexcontrol.core.Arm`/Head/etc.
  classes. Update annotations to the installed modern types where appropriate.
- Replace `from dexcomm import RateLimiter` with `from dexcontrol import RateLimiter` if rate limiting is the only reason for that dependency. Keep
  public exception imports from `dexcontrol.exceptions`. Removed utility/app
  modules need a concrete supported replacement or application-owned code;
  do not assume an old module path still exists.
- Replace legacy configuration objects that only enable sensors with
  `Robot(profile=..., enable_sensors=[...])`. Preserve the application's profile
  and deployment identity; do not replace them with a demo profile.
- If existing code deliberately relies on environment resolution, preserve it.
  `ROBOT_CONFIG` takes precedence over `ROBOT_NAME`, as in 0.5, and takes a
  profile name or a robot identity; an unknown value fails instead of
  defaulting. When **neither** variable is set the client assumes `vega_1`
  and only warns. A deployment that relied on 0.5 defaults with no variables set must
  get an explicit `profile=` or a documented environment setting; do not let
  it fall through to `vega_1`.
- Inspect every other config mutation. Translate it only to a documented
  profile/config-file option. Do not silently delete changes to components,
  limits, transport, or E-stop monitoring.
- Use the robot as a context manager or keep a guaranteed `shutdown()` path.
  Components share its lifetime. Do not create another Robot to access sensors.
- A `Robot` connection puts every arm into position mode before it returns,
  and a required arm whose motor does not answer fails the connection with
  `ComponentError` (0.5 logged the failure and continued). Scripts whose job
  is recovery or maintenance — `clear_error`/`clear_errors`,
  `reboot` (0.5: `reboot_component`), brake and software E-stop tools — must connect with
  `Robot(..., initialize_arm_modes=False)`, otherwise the fault they exist to
  clear stops them from connecting. `DiagnosticClient` cannot call those
  services; it is read-only by design. Do not add the flag to ordinary control
  applications, which need the arms enabled.
- For shutdown signals, the application may opt into
  `dexcontrol.install_signal_handlers()` on the main thread. Inspect existing
  handlers first; this helper replaces handlers. Do not install it blindly in
  imported modules or worker threads.
- **0.5 killed the process; 0.8 does not.** 0.5 called `os._exit(1)` from its
  monitor threads when an E-stop button was pressed and when the heartbeat
  timed out. 0.8 keeps the process alive: commands raise `EStopActiveError`
  while an E-stop is engaged, and a heartbeat failure runs the profile's
  action (`request_process_termination` in shipped profiles: stop motion and
  tear the connection down, after which calls raise `RobotConnectionError`).
  Process exit is opt-in: `Robot(..., exit_on_safety_termination=True)`.
  Supervisors, launch files and watchdog scripts that relied on the exit need
  an explicit decision: opt in, or react to `robot.wait_safety_event(timeout)`
  / `robot.take_safety_events()` / `robot.health()` in the application. The
  compatibility adapter does not restore the exit. Record the decision in the
  report; do not leave it implicit.
- Use `DiagnosticClient` for tools that genuinely need a read-only connection;
  do not emulate read-only mode by disabling safety on Robot. `DiagnosticClient`
  supports `simulation=True` for offline tests and exposes typed live sensors
  through `.sensors` in Python. Without that flag it connects to hardware.
- Keep dependencies the application still imports directly. Remove `dexcomm`,
  `dexbot_utils`, `loguru`, or other dependencies only after checking actual use.
  SDK records go to loguru when it is installed (now an optional dependency)
  and to stdlib `logging` otherwise; keep whichever the application relies on.
  Update the dependency and lockfile to the tested artifact using the customer's
  pinning policy. Do not introduce an unbounded `dexcontrol>=0.8` requirement;
  pin the tested release (`==0.8.1` or newer). Record the approved artifact
  and hash where the
  project's tooling supports it. `dexcontrol[examples]` carries example-only
  extras; install them only if the application needs them.
- `@supported_models("vega_1", ...)` from `dexcontrol.utils.compat` has no
  modern decorator. Check the resolved configuration's base model before
  constructing Robot, using the same profile selection as the application.
  Keep the original supported-model list. A profile such as `vega_1_f5d6` is
  not the base model `vega_1`, and reading `ROBOT_NAME` directly ignores a
  `ROBOT_CONFIG` override. The offline helper below covers built-in profiles.
- Old `Robot(auto_shutdown=False)` only suppressed signal handlers; drop the
  argument. There is no modern equivalent to pass.

For built-in profiles, resolve and check without opening a connection:

```python
import dexcontrol

def checked_profile(supported_models, profile=None):
    selected = profile if profile is not None else dexcontrol.profile_from_environment()
    model = dexcontrol.robot_config(selected)["robot"]["model"]
    if model not in supported_models:
        raise ValueError(f"Unsupported robot model: {model}")
    return selected
```

Call this with the application's explicit profile, or its deliberately configured
environment, then pass the returned profile to Robot. Do not use the resolver's
implicit default to guess deployment hardware. For custom `config_file` setups,
check the model in the effective custom configuration instead of substituting
built-in profile metadata.

### Joint reads and result consumers

| Legacy assumption | Direct migration |
| --- | --- |
| `arm.get_joint_pos(i)` is a scalar | Use `arm.get_joint_pos(i)[0]`; use `float(...)` only where a Python float is required. |
| `arm.get_joint_state(i)` is a three-element row | Use `arm.get_joint_state(i)[0]`. The updated source uses a complete current vector, otherwise a complete torque vector, and raises if neither covers all joints. 0.8.0 zero-filled missing current instead. Name the column in the consumer. |
| `get_state()['pos']`, `['vel']`, `['cur']`, `['torque']` | Use `position`, `velocity`, `current`, `torque`; an absent measurement is an empty array, never zeros. Update every consumer of that schema. |
| `get_state()['error']` contains joint error codes | Use `get_joint_err()` or `get_joint_err_dict()`; `error_joint_indices` is not an array of error codes. |
| `get_joint_err(joint_id=i)` | Read `get_joint_err()` and select explicitly, or use the filtered dictionary getter. |
| `robot.get_joint_pos_dict([...])` is flat | Handle the returned component-to-joint mapping, or explicitly flatten it if the application's schema must stay flat. |

Modern joint arrays are generally float64; historical arrays were generally
float32. Preserve required external dtypes explicitly. Do not change numerical
precision throughout the application merely to match an SDK return type.

Old combined joint state used torque when current was absent. 0.8.1 restores
that fallback and raises when velocity or effort is incomplete; 0.8.0 does not
behave this way, so on 0.8.0 select the needed raw-state measurements
explicitly in the application. Where the application needs to know *which*
effort it got, read `get_state()["current"]` or `["torque"]` and verify the vector
covers the relevant joints; preserve units. Never substitute zero for missing
measurements. Test torque-only and missing-measurement cases, not only complete
mock state.

Partial velocity dictionaries need explicit conversion for the modern API:

```python
import numpy as np

def velocity_vector(component, values):
    names = list(component.joint_names)
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"Unknown joint names: {sorted(unknown)}")
    vector = np.asarray([values.get(name, 0.0) for name in names], dtype=float)
    if vector.shape != (len(names),) or not np.isfinite(vector).all():
        raise ValueError("Expected one finite velocity per joint")
    return vector
```

Unspecified velocities are zero. Never fill them with current positions. Validate
full joint-vector length before relative arithmetic; NumPy broadcasting a single
value across an arm is not a substitute for selecting a joint.

### Sensors and camera frames

Select sensors from the intended profile (`dexcontrol.robot_config(profile)`),
not from a hard-coded list copied from another robot. Enable only those needed.

| Legacy access | Modern access |
| --- | --- |
| `sensors.head_imu` | `sensors.get_imu('head_camera')`, with head_camera enabled |
| — | `sensors.get_temperature('head_camera')` for the ZED camera's own temperature (new; not a camera stream) |
| `sensors.base_imu` | `sensors.get_imu('chassis_imu')`, when declared and enabled |
| `sensors.lidar` | Explicit configured 2D LiDAR name via `get_lidar_2d(name)` |
| embedded LiDAR IMU | `get_imu(lidar_name)` for a device exposing that capability |
| ZED IMU observation `ang_vel` | `gyro`; check all consumers and retain the required coordinate frame/units |
| `arm.wrench_sensor.get_wrench_state()` | `sensors.get_wrench('left_arm').get_wrench_state()` for that arm |
| arm button getters | `arm.get_button_state()` with `blue_button` / `green_button` keys |
| `torso.temperature_sensor` | `sensors.get_temperature('torso')` |

Do not resolve base IMU to an arbitrary sensor with IMU capability. Camera and
LiDAR IMUs belong to those devices and may have different reference frames.
Chassis steer/drive temperature access is currently unavailable through the
native temperature reader; record that gap rather than fabricating a value.

For a modern camera, prefer `robot.sensors.subscribe_camera(name, stream)`
contexts for managed ownership, especially alongside compatibility code. Unpack
the returned frame after establishing that subscription:

```python
def poll_left_image(camera):
    frame = camera.latest_frame("left_rgb")
    if frame is None:
        return None
    return frame["data"], frame["timestamp_ns"]

# At setup, after enabling the camera on the Robot:
# camera = robot.sensors.get_camera("head_camera")
# with robot.sensors.subscribe_camera("head_camera", "left_rgb") as camera:
#     image_and_timestamp = poll_left_image(camera)
# The context releases its ownership automatically on exit.
```

Choose stream names from `camera.streams`. RGB-only devices use `rgb`, not
`left_rgb`. Preserve RGB/BGR order, depth dtype/units, and timestamp units in
consumers. `latest_frame` returns a dictionary, not an image array, and returns
None before a frame arrives. Handle no data and stale/disconnected streams.
Do not subscribe in every iteration or replace someone else's buffering policy.
Keep monitor loops running until cancellation unless an explicit duration was
requested; keep one-shot readers bounded. Do not add image subscriptions merely
to read a camera IMU.

`frame_shape(stream)` returns dimensions after a frame is available; it is not
calibration. Modern `camera.get_camera_info(timeout=2.0)` queries the camera's
server metadata/calibration without subscribing to images; it raises on service
or decoding failures. The 0.5 adapter keeps `get_camera_info(force_refresh=False)`
with its successful-result cache and `None` on service failure. Fields depend on
the camera/server: do not invent missing intrinsics or extrinsics. RTC-only
applications need a transport decision; do not pretend the default build
provides RTC. When migrating `get_camera_info(True)`, do not forward that boolean
positionally to the modern method: the parameter there is a timeout, not refresh.
Choose an explicit timeout and migrate any code depending on caching or `None`
on failure. Camera-info replies may omit calibration and simulation has no reply
by default; test populated, sparse, malformed, and unavailable responses where
the application relies on this service.

### Queries

- `get_version_info(show=...)` → `robot.version_info`; printing belongs to the app.
- `get_component_status(show=...)` → `robot.component_status()`; check schema use.
- `query_ntp(n, show, device)` → `query_ntp(n)` for SOC only; preserve units and
  handle no replies. Do not silently discard a different device selection.
- `query_hand_type()` has no equivalent call: the connection asks the robot itself
  and adopts the hands it reports into the profile (0.9), so `has_component("left_hand")`
  after connect answers the question the 0.5 call did. Code that branched on the
  hand type should branch on the hand component's driver or metadata instead.
- State subscriptions are always on. `Robot.set_idle_timeout`,
  `Robot.set_subscription_policy` and `get_subscription_status` are gone:
  delete the calls. `JointComponent.set_subscription_policy` and
  `Chassis.set_subscription_policy` still exist as `RuntimeWarning` no-ops, so
  a call to them is dead code to remove, not a blocker and not something to
  shim. Camera streams have explicit `subscribe`/`unsubscribe`.

## 4. Direct migration: review motion semantics before changing calls

These are decision rules, not global search-and-replace recipes:

| Old intent | Candidate modern API | Required review |
| --- | --- | --- |
| Send a direct control-loop setpoint | `set_joint_pos(q)` or `set_joint_pos(q, joint_vel=qd)` | Preserve rate, units, relative meaning, and full vector dimensions. The configured component-specific step guard applies; 0.5 clipped to joint limits, the modern API rejects. |
| Move the goal every tick (teleop, servoing); old `move_joint_pos(q)` | `stream_joint_pos(q)` | Same untracked, plugin-smoothed goal as 0.5. Never rewrite a per-tick `move_joint_pos` as `move_to_joint_pos(wait=False)`: each call replans and the arm lags. The old name still works and warns; removal is evaluated per API, no earlier than 1.0. |
| Plan a move and return immediately | `move_to_joint_pos(q, wait=False)` | Keep the handle if completion/cancellation matters; compare old fire-and-forget assumptions. |
| `robot.set_joint_pos({...})` / `robot.move_joint_pos({...})` / `robot.move_to_joint_pos({...})` on several components | `robot.set_joint_pos(dict)`, `robot.stream_joint_pos(dict)`, `robot.move_to_joint_pos(dict, wait=...)` | Validation is all-or-none; delivery is per component and a failure raises with `component` and `delivery_report` attributes. `robot.start_joint_motions(dict)` returns `(group, report)` and does **not** raise on a delivery failure: check `report["all_delivered"]`. |
| `set_joint_pos_vel(q, qd)`, and the original-0.8.0 spellings `command_position(q, max_steps=...)` / `start_position(q)` | `set_joint_pos(q, joint_vel=qd)`, `set_joint_pos(q, max_steps=...)`, `move_to_joint_pos(q)` | The old names still work and emit `DeprecationWarning`; a direct migration leaves none of them behind. |
| Plan a move and wait for completion | `move_to_joint_pos(q, wait=True, timeout=t)` | `t` is a completion deadline, not requested trajectory duration. |
| Direct setpoint with convergence waiting | `set_joint_pos(q, wait=True, timeout=t, tolerance=...)` | Old polling could expire normally; expiration of the direct-setpoint convergence timeout raises `MotionTimeoutError`. Waiting alone does not turn a direct step into a planned ramp. |
| `DexGripper.set_joint_pos(q)` / `set_joint_pos_vel(q, qd)`, which sent `vel=1.0` and `torque=0.2` with every command | `hand.set_joint_pos(q)` (velocity from the model via `pv_mode`, `joint_vel=` to override); `hand.set_grasp_torque(t)` once for the grip force | The model velocity is 0.9 rad/s, the firmware's saturation speed, so timing is close to 0.5's `vel=1.0`. The profile's grasp torque is 0.2, the 0.5 value; only call `set_grasp_torque` where the application changed it, and keep it at or below 0.5 unless the customer confirms the hardware. `DexGripper.set_mode("mit"/"velocity"/"force")` has no mapping; record the intent as a pending decision. On a build whose `asset_version` is `legacy-dexmate-urdf`, pass `joint_vel=` explicitly. |
| `hand.open_hand(wait_time=..., exit_on_reach=...)` / `close_hand(...)` | unchanged | These two helpers keep their 0.5 arguments and treat an expired wait as a normal bounded wait, not an error. Leave them alone even though the inventory grep flags `wait_time`/`exit_on_reach`. |
| Chassis velocity for a period | `drive_for(vx, vy, wz, duration=t)` | Review stopping at the end and interruption; this is not automatically equivalent to repeated old velocity commands. |
| Timed position waypoints | `move_joint_trajectory(points, ...)` | Use `(positions, time_from_start)` pairs; review feedforward, scaling, and tracking changes. |
| Replay a precomputed trajectory tick by tick (0.5: `for q in trajectory: arm.set_joint_pos(q); rate.sleep()`, typically the output of a planner such as dexmotion or a recorded file) | `robot.execute_trajectory({component: waypoints}, control_hz=hz)`, or `move_joint_trajectory([(q, t), ...], wait=True)` when the robot should own the timing | **Never leave it as a `set_joint_pos` loop**, and never silence the guard with `max_step=None`. See "Replaying host-generated trajectories" below. |

### Replaying host-generated trajectories

The 0.5 way to run a planned or recorded trajectory was a loop that sent each
waypoint with `set_joint_pos` and slept for one control period. That loop
still compiles and still starts on the current client, and then fails on
hardware part-way through with:

```text
ValueError: invalid command for "left_arm": target moves a joint 0.310 native
units from its current position, exceeding max_step 0.3. set_joint_pos
publishes a single unshaped setpoint ... Use move_to_joint_pos() for planned
motion, or Robot.execute_trajectory() to stream it.
```

The cause is not a bad trajectory. `set_joint_pos` checks every setpoint
against the *measured* position, so the loop works only while the arm keeps
up with the host clock; as soon as it lags by more than 0.3 rad the next
setpoint is refused. 0.5 compared against the previous command, which is why
the same loop used to run. Simulation follows every setpoint instantly, so a
simulated run of the migrated loop proves nothing about this failure.

Migrate the loop, not the call, to one of two forms:

```python
# Host-paced, direct setpoints, every component in one call. The step guard
# is checked once, up front, on the waypoint spacing (measured position to the
# first waypoint, then waypoint to waypoint), so a well-sampled trajectory
# passes however much the arm lags. Velocities are optional feed-forward.
robot.execute_trajectory(
    {"left_arm": left_waypoints, "right_arm": right_waypoints},  # (T, N) arrays
    control_hz=control_hz,
    max_tracking_error=0.2,  # optional: abort if the arm falls this far behind
)
```

```python
# Planned and tracked on the robot; the robot owns the timing and reports
# completion. points are (positions, time_from_start) pairs.
motion = arm.move_joint_trajectory(
    [(q, t) for q, t in zip(waypoints, times)], wait=True, timeout=duration + 5.0
)
```

`execute_trajectory` is the direct translation of the 0.5 loop: it keeps the
host timing, takes every component at once, and replaces the application's
rate limiter. Use `move_joint_trajectory` when the application does not need
to pace the motion itself. `stream_joint_pos` is for goals that change every
tick in response to input, not for replaying a known trajectory: it gives no
completion signal and no tracking guard. Never force the old loop through
with `max_step=None`, `set_joint_pos(..., max_steps=...)` or the C++
`without_step_guard`: the guard is what keeps an unshaped setpoint from being
silently dropped by the controller.

### Stopping, E-stop handling and streamed loops

- `component.stop()` cancels that component's tracked motions and publishes a
  hold at the measured position; `robot.stop_all()` does it for every
  component and zeroes the base. Neither raises while an E-stop is engaged.
  A stop returns when the request was published, not when the joints are
  measured stationary; wait on state if the application needs that.
- A streamed loop (`stream_joint_pos` every tick) leaves its **last goal in
  effect** when it ends: the plugin keeps converging on it. End the loop
  deliberately: stream a final safe goal, hand off with one
  `move_to_joint_pos(q, wait=True)`, or call `stop()`. After streaming,
  `stop()` sends a replacement streamed goal instead of a raw hold and raises
  `StaleStateError` if no fresh measurement exists, because there is nothing
  safe to hold at; the robot-level and safety stops do the same for every
  component that streamed.
- `stream_joint_pos` raises `ValueError` while a tracked motion of the same
  component is still running. Wait on or cancel the handle before switching
  from planned moves to streaming; do not retry in a loop.
- `estop.activate()` / `estop.deactivate()` keep their 0.5 names. `deactivate()`
  now returns only once the robot's state reports the software E-stop released
  (bounded by 2 s) and raises `DexTimeoutError` otherwise; code that sent a
  command immediately after `deactivate()` no longer races the release. While
  an E-stop is engaged, motion, PID, brake and raw-request calls raise
  `EStopActiveError`; reads, `stop()`, `set_mode`/`set_modes`/`set_idle_mode`
  still work. `estop.toggle()`, `show()`, `is_button_pressed()` and
  `is_software_estop_enabled()` exist unchanged.
- `heartbeat.pause()` / `resume()` exist unchanged. Pausing suspends heartbeat
  staleness evaluation, emits a `Notify` safety event and shows in
  `health()["heartbeat_paused"]`; it is not a way to run without supervision.

An old arm `wait_time` may encode a host-side ramp plus repeated holding; old
head/torso/hand calls may send once then sleep or poll. Do not substitute the
same modern call everywhere. Read the caller to determine whether duration,
convergence, command repetition, or synchronization with another device matters.
Ask the customer when the intent cannot be determined; continue unrelated edits.

Old managed trajectories can include velocities, accelerations, per-joint scale,
and a tracked flag. The modern signature does not accept those fields directly.
For position/time-only intent, the mechanical conversion is:

```python
def position_time_points(positions, time_from_start):
    if len(positions) != len(time_from_start):
        raise ValueError("positions and times must have equal length")
    return [(list(q), float(t)) for q, t in zip(positions, time_from_start)]
```

Then call the modern component's `move_joint_trajectory(points, wait=False)`
where the application owns motion execution. Validate time ordering and joint
counts. Do not discard legacy feedforward/scaling fields to make the call pass.

`Robot.execute_trajectory(relative=True)` is a particularly important change:
0.5 resolved each tick against that tick's measured position; the modern API
uses one starting position. A cumulative sum of desired points is not generally
equivalent when tracking lags. Determine the intended semantics before rewriting.
The compatibility adapter currently rejects this option; it is not an automatic
solution for preserving that old behavior.

Preserve E-stop, freshness, joint/velocity limits, step guards, watchdogs, and
bounded waits. Fix the command or choose an appropriate motion API when a guard
rejects it; do not disable guards to get a migration test to pass. Do not add
startup homing or replace `release_brake(..., joints=None)` with all-joint brake
release automatically. These require an explicit application/hardware decision.

Use `MotionTimeoutError` and `MotionStoppedError` where appropriate. A timeout
can end waiting without stopping motion: do not blindly retry and duplicate a
command. Decide whether the application should cancel, stop, report failure, or
continue monitoring. Read actual return/status types; `handle.wait()` and
`handle.status.state` are not interchangeable forms.

## 5. Verify without hardware

1. Review test fixtures and import side effects. Force every test connection to
   `simulation=True` or replace it with a fake. Never retry without simulation
   because the simulated service or camera is unavailable.
2. Compile/check changed Python files and run the relevant application tests.
   Avoid executing unreviewed entry points, notebooks, or broad integration suites.
3. Add focused tests for used contracts: scalar/array shapes, nested dictionaries,
   torque-only state, populated camera frames/timestamps, partial velocities,
   malformed targets, and cleanup. Include only cases relevant to the application.
   For compatibility mode, assert the legacy shapes/keys rather than the modern
   replacements. For mixed camera consumers, cover both acquisition and release
   orders, retaining image delivery until the final owner exits. For camera-info
   users, test cached reads, forced refresh, and failed-refresh behavior.
4. Test changed motion logic with fake command recording and controlled time;
   assert rejected inputs send zero commands. Test interruption and timeout paths.
   Where the application reacted to an E-stop or heartbeat failure (section 4),
   test the new reaction with a fake raising `EStopActiveError` /
   `RobotConnectionError` or a fake safety event; do not assume the process
   still exits.
5. Run a small simulation smoke test with the intended profile. Simulation does
   not certify firmware services, real sensor timing, images, or physical motion,
   and it cannot show the step-guard failure of a trajectory replay loop
   (section 4), because the simulated arm never lags: a replay path migrated
   to `execute_trajectory` or `move_joint_trajectory` is verified by reading,
   a replay path left as a `set_joint_pos` loop is a defect even when every
   simulated test passes.

For compatibility/gradual migration, also exercise an application-style
sensor-read → compute → command loop through the selected view using simulated
state and populated fake camera data. Check command targets, units, timing,
return shapes, and cleanup. No image/service data in simulation is not a reason
to switch to hardware or silently skip a required path.

The following standalone smoke checks the **modern** connection only (save as
`sdk_smoke.py`); adapter users must also test their legacy application paths:

```python
import argparse
from dexcontrol import Robot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    with Robot(profile=args.profile, simulation=True) as robot:
        for name, component in robot.get_controllable_component_map().items():
            positions = component.get_joint_pos()
            assert positions.shape == (len(component.joint_names),), name
        print("SDK simulation smoke passed")

if __name__ == "__main__":
    main()
```

Run `python sdk_smoke.py --profile YOUR_CONFIGURED_PROFILE`, substituting a real
profile from the inspected configuration. The smoke reads simulated state and
sends no explicit motion commands. It is not a substitute for application tests.

If mock services have no reply, test the caller with a fake response and report
that limitation. For images, supply a populated native-style frame dictionary;
testing only None frames misses shape/timestamp conversion bugs.

## 6. Deliver the migration, with remaining decisions visible

For every strategy, result consumers must match the chosen API, configuration
fields and command arguments must not be silently dropped, and required paths
must pass relevant offline tests or have explicitly reported blockers.

A compatibility migration is complete when the agreed application paths use
adapter imports/configuration, preserve their supported legacy behavior, and
pass sensor/control/cleanup tests against the identified artifact. Identify any
remaining unsupported calls. A gradual migration must additionally identify
which paths use each view and verify shared connection/subscription ownership.
Neither strategy should be reported as a complete direct migration.

A direct migration is complete when:

- Active application paths use modern imports and signatures; residual legacy
  references are documented fixtures, comments, or explicitly agreed exceptions.
- Result consumers and schemas match the tested target API.
- There are no silently dropped configuration fields or motion arguments.
- Relevant offline tests pass and unsupported features have clear failures.
- The report identifies the installed build, behavior changes, and any hardware
  validation still needed. Source compatibility is not a claim of hardware safety.

Finish with a concise report:

```text
Strategy: <compatibility, gradual, or direct>
Target build: <version and local build provenance>
Files changed: <paths and purpose>
Behavior changes: <timing, return schemas, lifecycle, intentionally unsupported APIs>
Validation: <exact commands, results, simulation/fake limitations>
Pending decisions: <only issues requiring customer input>
Hardware validation: <not run, or separately authorized results>
```

Do not claim completion if a required path is still a placeholder or silently
skipped. If direct migration needs an unresolved behavior choice, state the
specific blocker. If a different strategy would unblock it, explain the
tradeoff and get the customer's choice instead of silently changing the goal.
The explicit adapter is a supported compatibility path with no scheduled removal,
not proof that every historical API or hardware configuration is supported.
