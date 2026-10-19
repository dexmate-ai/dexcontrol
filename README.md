<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# DexControl

Control Dexmate robots from Python, C++, or Rust. Read joint positions, move
joints, use named poses, and monitor robot state through a shared robot API.

The examples below use a simulated `vega_1`. They run without a connected robot.
Joint angles are in radians; timeouts are in seconds.

## Get the examples

Download a release from [GitHub Releases](https://github.com/dexmate-ai/dexcontrol/releases),
or clone the matching version of this repository:

```bash
git clone --branch v0.8.0 https://github.com/dexmate-ai/dexcontrol.git
cd dexcontrol
```

This release uses `0.8.0` for Python, Rust, and the native SDK. Keep the repository,
Python package, native SDK, and Rust dependency on the same release.
Run the example commands from the repository root.

## Run examples

After installing the package or native SDK below, use the same launcher for
all three languages:

```bash
./run python read_joint_positions
./run cpp read_joint_positions
./run rust read_joint_positions
```

The launcher uses your active Python environment or installed native SDK.
C++ needs CMake and a C++ compiler; Rust needs Cargo and Rust. It checks only
the selected language and prints installation commands for missing tools.
Python 3.8 or newer is needed to run the launcher itself.

```bash
./run --list
./run cpp --list
./run --release cpp read_joint_positions
./run --build-only rust read_joint_positions
./run --dry-run cpp read_joint_positions
```

`--release` builds optimized C++/Rust examples. `--build-only` compiles without
running. `--dry-run` prints commands without building or running. Put launcher
options before the example name; anything after it is passed to the example.
The included `read_joint_positions` example always uses simulation.

## Full example suite

The complete [examples directory](examples/) is included: Python, C++, Rust,
shared helpers, task descriptions, fixtures, and the Vega dance recording.
See [example behavior](examples/BEHAVIOR.md) before running a task.

Python and C++ tasks can be selected by filename without its extension:

```bash
python -m pip install 'dexcontrol[examples]==0.8.0'
./run python cycle_arm --simulated --profile vega_1 --delta 0.01
./run cpp cycle_arm --simulated --profile vega_1 --delta 0.01
./run python replay_trajectory --help
```

Most full-suite examples select real hardware unless `--simulated` is given.
Planning and teleoperation programs require the dependencies and hardware
listed in their individual documentation. Listing or displaying help does
not run a motion.

The full Rust suite uses the public `dexcontrol` crate and precompiled native
SDK. It requires no private repositories. For example:

```bash
./run rust cycle_arm --simulated --profile vega_1 --delta 0.01
./run rust read_battery_status --simulated --profile vega_1
./run rust --list
```

## Python

Requires Python 3.8 or newer. Wheels are available for Linux x86_64/ARM64 and
macOS Intel/Apple Silicon. No C++ or Rust compiler is needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install dexcontrol==0.8.0
./run python read_joint_positions
```

The example prints positions for the head, both arms, and torso:

```text
head joint positions (rad): 0.0000, 0.0000, 0.0000
left_arm joint positions (rad): 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
right_arm joint positions (rad): 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
torso joint positions (rad): 0.0000, 0.0000, 0.0000
```

To read positions and move the simulated head, save this as `move_head.py`:

```python
from dexcontrol import Robot

with Robot(profile="vega_1", simulation=True) as robot:
    head = robot.joints("head")
    print("Before (rad):", head.get_joint_pos())

    motion = head.move_to_joint_pos([0.1, 0.0, 0.0], wait=False)
    motion.wait(timeout=10.0)

    print("After (rad):", head.get_joint_pos())
```

Run it with `python move_head.py`. The `with` block closes the connection when
it exits. Use `robot.joints("left_arm")`, `robot.joints("right_arm")`, or
`robot.joints("torso")` to access another joint component.

## Native SDK for C++ and Rust

Download the matching archive and `SHA256SUMS` from
[GitHub Releases](https://github.com/dexmate-ai/dexcontrol/releases).
Choose the platform that matches your computer:

| Computer | Archive suffix |
|---|---|
| Linux x86_64 | `linux-x86_64` |
| Linux ARM64 | `linux-aarch64` |
| macOS Apple Silicon | `macos-arm64` |
| macOS Intel | `macos-x86_64` |

For example, on Linux x86_64, place both downloads in the repository root:

```bash
sha256sum --ignore-missing -c SHA256SUMS
python3 install_sdk.py dexcontrol-sdk-v0.8.0-linux-x86_64.tar.gz "$HOME/.local/dexcontrol/0.8.0"
source "$HOME/.local/dexcontrol/0.8.0/activate"
```

On macOS, use `shasum -a 256 <archive>` and compare the digest with the matching
entry in `SHA256SUMS`, then run the installer with your archive's filename.

Activate the SDK in each new terminal before building or running C++/Rust
applications. Installation does not require administrator privileges. Use a new
installation directory when installing another version.

## C++

Requires a C++17 compiler, CMake 3.16 or newer, and the activated native SDK.

Build and run the joint-position example:

```bash
./run cpp read_joint_positions
```

It reads the same four components as the Python example and prints radians.

For your own application, save this as `main.cpp`:

```cpp
#include <dexcontrol/dexcontrol.hpp>
#include <iostream>
#include <chrono>

int main() {
    try {
        auto robot = dexcontrol::Robot::simulated("vega_1");
        auto head = robot.joints("head");
        auto motion = head.move_to_joint_pos(
            {0.1, 0.0, 0.0}, dexcontrol::NoWait{});
        motion.wait(dexcontrol::WaitFor{std::chrono::seconds(10)});

        for (double position : head.get_joint_pos()) {
            std::cout << position << ' ';
        }
        std::cout << "rad\n";
        robot.close();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

Create a `CMakeLists.txt` beside it:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_robot LANGUAGES CXX)
find_package(dexcontrol CONFIG REQUIRED)
add_executable(my_robot main.cpp)
target_compile_features(my_robot PRIVATE cxx_std_17)
target_link_libraries(my_robot PRIVATE dexcontrol::dexcontrol)
```

From your application's directory:

```bash
cmake -S . -B build
cmake --build build --parallel
./build/my_robot
```

## Rust

The single `dexcontrol` crate includes the Rust API and native-library bindings.
It links to the separately installed native SDK.

Requires Rust 1.90 or newer and the activated native SDK.

Run the joint-position example from this repository:

```bash
./run rust read_joint_positions
```

For your own application:

```bash
cargo new my_robot
cd my_robot
cargo add dexcontrol@=0.8.0
```

Replace `src/main.rs` with:

```rust
use dexcontrol::{MotionOptions, Robot};
use std::time::Duration;

fn main() -> dexcontrol::Result<()> {
    let robot = Robot::simulated("vega_1")?;
    let head = robot.joints("head")?;
    let motion = head.move_to_joint_pos(
        &[0.1, 0.0, 0.0], MotionOptions::default(),
    )?;
    motion.wait(Duration::from_secs(10))?;

    println!("Head positions (rad): {:?}", head.get_joint_pos()?);
    robot.close()
}
```

Run it with `cargo run`.

The Rust API also provides `WaitOptions` (tolerance and settling), motion groups,
recorded trajectories, named chassis control, firmware services, E-stop status,
battery/IMU/LiDAR/camera reads, `DiagnosticClient`, and `RateLimiter`.
Use `cargo doc --open` in your application for signatures and ownership rules.
Sensor snapshots own their data; borrowed channels remain valid while the
snapshot is alive. SDK calls are synchronous; async applications should run
blocking operations on a worker thread.

Simulation supports motion and sensor-state testing. It does not supply camera
images or firmware replies for PID, brakes, baud rate, force-torque modes,
reboot, or clear-error requests. Those examples report the unavailable service;
read-only diagnostic examples require a real robot.

## Basic API

| Task | Python | C++ | Rust |
|---|---|---|---|
| Select a joint component | `robot.joints("head")` | `robot.joints("head")` | `robot.joints("head")?` |
| Read positions | `head.get_joint_pos()` | `head.get_joint_pos()` | `head.get_joint_pos()?` |
| Resolve a named pose | `head.get_pose("home")` | `head.get_pose("home")` | `head.get_pose("home", false)?` |
| Cancel a motion | `motion.cancel()` | `motion.cancel()` | `motion.cancel()?` |

Pose names and available components depend on the robot model. Pose resolution
applies the model's adjustments. To request stored joint values, use
`passthrough=True` in Python, or `true` as the second `get_pose` argument in
C++/Rust.

Motion handles let you start a movement and wait for it later. The examples
explicitly wait up to 10 seconds. Python and C++ report failures as exceptions;
Rust returns `Result` values. Active E-stops block movement commands.

Browse the API definitions for more options:

- [Python types and methods](python/dexcontrol/_native.pyi)
- [C++ API](include/dexcontrol/dexcontrol.hpp)
- [Rust API](rust/dexcontrol/src/lib.rs)

## Sensor data

Enable the sensors your application needs, then use their typed accessors.
The following example reads a head-camera image on a real robot:

```python
from dexcontrol import Robot

with Robot(profile="vega_1p", enable_sensors=["head_camera"]) as robot:
    camera = robot.sensors.get_camera("head_camera")
    print(camera.streams)
    camera.subscribe("left_rgb")
    try:
        frame = camera.wait_for_frame("left_rgb", timeout=5.0)
        if frame is not None:
            print(frame["data"].shape, frame["timestamp_ns"])
    finally:
        camera.unsubscribe("left_rgb")
```

A camera frame contains `data` (a NumPy array), `timestamp_ns`, and `frame_id`.
Images require an explicit stream subscription. Use `camera.streams` to select
an available stream. Simulation can exercise subscriptions but does not generate
camera images. A normal Robot connection can enable compatible arm modes during
startup; use the simulated connection when testing without hardware.

Other Python reads are:

| Data | Accessor on `robot.sensors` | Read |
| --- | --- | --- |
| IMU | `get_imu(name)` | `get_obs()` — `acc`, `gyro`, `quat`, `timestamp_ns` |
| 2D LiDAR | `get_lidar_2d(name)` | `get_obs()` — ranges, angles, qualities, timestamp |
| 3D LiDAR | `get_lidar_3d(name)` | `get_points()` — `(N, 3)` XYZ array |
| Wrist force/torque | `get_wrench("left_arm")` | `get_wrench_state()` |
| Component temperatures | `get_temperature("left_arm")` | `get_temperatures()` |

IMU, LiDAR, and wrench handles provide `wait_for_active(timeout=5.0)`. Reads return the
latest cached sample, not necessarily a new one; observation and frame reads
can return `None` before data arrives. Check timestamps or state-handle
`get_age()` / `is_active(window=...)` when freshness matters.

Inspect the sensor names declared by your installed SDK without connecting:

```python
from dexcontrol import robot_config

config = robot_config(profile="vega_1p")
for name, sensor in config["sensors"].items():
    print(name, [cap["id"] for cap in sensor["capabilities"]])
```

## Camera streams

The standard packages support Zenoh camera image and depth streams. They do not
include FFmpeg or WebRTC video decoding. A camera stream configured only for RTC
reports that RTC support is not enabled; use a configured Zenoh topic stream.
When both transports are configured for a stream, the SDK prefers the topic.
Simulation does not verify that a real camera publisher offers that transport.

## Connect to a robot

Configure robot access on your computer and select the robot with
`dextop robot use <robot-name>`. Use the model profile that matches your robot,
such as `vega_1`, `vega_1p`, or `vega_1u`.

Replace the simulated connection in the examples with:

```python
with Robot(profile="vega_1p") as robot:
    print(robot.joints("head").get_joint_pos())
```

```cpp
dexcontrol::ConnectOptions options;
options.profile = "vega_1p";
auto robot = dexcontrol::Robot::connect(options);
```

```rust
let robot = Robot::connect(dexcontrol::ConnectOptions {
    profile: Some("vega_1p"),
    ..Default::default()
})?;
```

Start with the joint-position example to check the connection. Movement commands
on a real connection operate the physical robot.

## License and support

See [LICENSE](LICENSE) for licensing terms. For support and commercial licensing,
contact [contact@dexmate.ai](mailto:contact@dexmate.ai).
