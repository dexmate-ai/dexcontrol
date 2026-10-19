<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# Rust examples

From the repository root, build and run an example with one command:

```bash
./run rust cycle_arm --simulated --profile vega_1p --delta 0.01
./run rust --list
./run rust cycle_arm --help
```

Use `./run --release rust cycle_arm --simulated` for an optimized build, or
`./run --build-only rust cycle_arm` to compile without running. See the
[launcher guide](../../README.md#run-examples) for options. Manual build commands
remain available below.

These examples use the public `dexcontrol` Rust API and the compiled native SDK.
The wrapper calls the same runtime as Python and C++; no private Rust source or
registry access is needed. Tokio provides example timing and Ctrl-C handling;
SDK calls themselves are synchronous.

After activating the matching native SDK, manual builds also work:

```bash
cargo build --manifest-path examples/vega_1/rust/Cargo.toml --all-targets --locked
cargo run --manifest-path examples/vega_1/rust/Cargo.toml --bin control-cycle-arm -- --simulated --profile vega_1
```

The SDK includes the watchdog executable. Keep its `bin` directory on `PATH`
by sourcing the SDK's `activate` script.

Robot startup may enable compatible arm modes, and shutdown requests a stop even for sensor examples.

Every control and sensor example accepts `--simulated`, which uses the native
deterministic loopback transport and never contacts hardware. Diagnostic
examples are production-only and reject `--simulated` before connecting.
Robot examples close the connection on success and errors. A separate task
closes it on Ctrl-C, including while a synchronous SDK call is waiting.

Common arguments are `--profile NAME`, `--config FILE`, repeatable
`--enable-sensor NAME`, and `--simulated`. Each binary documents its specific
arguments in the source and accepts `--help`.

See the [cross-language task inventory](../../README.md) for checked paths,
matching simulation commands and known behavioral differences.

## Coverage

The binaries mirror every C++ example and every Python example whose
behavior is implemented by DexControl itself: basic control and sensors,
firmware configuration, E-stop, folding and named poses, custom end effectors,
host trajectory replay, tracking benchmarks, and troubleshooting.

Seven additional runnable Python programs depend on Python-specific tooling or external libraries; four underscored Python files are shared helpers. See [exact behavior](../../BEHAVIOR.md) and [renamed paths](../../RENAMED.md). Runtime defaults and output formats differ where that reference says so.

These programs move physical hardware unless `--simulated` is present. Keep a
hardware E-stop within reach, clear the workspace, and begin with the read-only
diagnostic example.

## Image visualization

Camera monitors open an owned Rerun window when native visualization support
is built. Packaged control-SDK examples use `--no-display` for terminal output.
Install the viewer with
`pip install rerun-sdk==0.36.2`; native image logging does not call Python.
Use `--no-display` for terminal output or `--viewer-url HOST:PORT` for an
existing viewer. Closing the window stops monitoring; exiting or killing the
monitor closes its owned window. The native viewer build requires Rust 1.95+.

With native visualization enabled, the reusable viewer supports both live
streams and one-time images independently of Robot. See
[visualization behavior](../../BEHAVIOR.md#connections-waiting-and-output).
