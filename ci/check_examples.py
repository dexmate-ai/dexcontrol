# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai
"""Run every public Rust task in simulation; never contact a robot."""
import json
import os
from pathlib import Path
import subprocess
import selectors
import signal
import sys
import tempfile


def check(root, binaries):
    environment = dict(os.environ)
    for key in ("ROBOT_NAME", "ZENOH_CONFIG", "CARGO_REGISTRIES_DEXMATE_TOKEN"):
        environment.pop(key, None)
    tasks = json.loads((root / "examples/tasks.json").read_text())["tasks"]
    failures = []
    with tempfile.TemporaryDirectory(prefix="dexcontrol-example-test-") as temporary:
        trajectory = Path(temporary) / "tiny.csv"
        trajectory.write_text("control_hz,100\nhead:0,head:1,head:2\n0,0,0\n0.01,0,0\n")
        for task in tasks:
            binary = task["rust_bin"]
            path = binaries / binary
            # Check every task's parser before a connection is even possible.
            subprocess.run([str(path), "--help"], env=environment, stdout=subprocess.DEVNULL, check=True, timeout=10)
            unknown = subprocess.run([str(path), "--simluated"], env=environment, capture_output=True, text=True, timeout=10)
            if unknown.returncode == 0 or not any(message in unknown.stderr for message in ("unknown option", "requires a value")):
                raise RuntimeError(binary + ": misspelled simulation option was not rejected")
            diagnostic = binary in {"troubleshooting-display-robot-info", "troubleshooting-measure-robot-clock-offset"}
            profile = "vega_1p" if binary == "sensor-monitor-3d-lidar-points" else "vega_1_f5d6" if "hand" in binary else "vega_1"
            arguments = ["--simulated", "--profile", profile]
            options = set(task["cli_rust"])
            for option, value in (("samples", "1"), ("period", "0.01"), ("wait-time", "0"), ("settle", "0"), ("delta", "0.01")):
                if option in options:
                    arguments += ["--" + option, value]
            if "no-display" in options:
                arguments += ["--no-display"]
            if "duration" in options:
                arguments += ["--duration", "0.05"]
            if binary.startswith("benchmark-arm-"):
                arguments += ["--control-hz", "100", "--output-dir", temporary]
                if binary.endswith("step"):
                    arguments += ["--step-time", "0.01", "--transition-time", "0.01"]
            if binary in {"advanced-configure-arm-pid", "advanced-configure-ee-baud-rate", "advanced-configure-arm-ft-sensor"}:
                arguments += ["get"]
            if binary == "advanced-manage-arm-brakes":
                arguments += ["status"]
            if binary == "advanced-reboot-control-board":
                arguments += ["arm"]
            if binary == "advanced-replay-trajectory":
                arguments += [str(trajectory)]
            if binary == "sensor-monitor-lidar-imu":
                arguments += ["--sensor", "head_camera"]
            if binary == "sensor-monitor-chassis-cameras":
                arguments += ["--camera", "head_camera"]
            result = subprocess.run([str(path), *arguments], input="w\ns\nq\n", env=environment, cwd=root,
                                    capture_output=True, text=True, timeout=30)
            # Firmware request handlers are not emulated by the public simulator.
            # Require their specific transport failure, never treat arbitrary failure as success.
            unsupported = {
                "advanced-configure-arm-pid": "no mock response queued for system/arm_pid/left",
                "advanced-configure-ee-baud-rate": "no mock response queued for system/ee_baud_rate/left",
                "advanced-configure-arm-ft-sensor": "no mock response queued for mode/force_torque_sensor/left",
                "advanced-manage-arm-brakes": "no mock response queued for system/arm_brake/left",
                "troubleshooting-clear-component-errors": "no mock response queued for system/clear_error",
                "advanced-reboot-control-board": "no mock response queued for system/reboot",
            }
            expected_error = unsupported.get(binary)
            if diagnostic:
                passed = result.returncode != 0 and "simulation" in result.stderr
            elif expected_error:
                passed = result.returncode != 0 and expected_error in (result.stdout + result.stderr)
            else:
                passed = result.returncode == 0
            if not passed:
                failures.append(f"{binary} failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
                continue
            print(f"Verified {binary}" + (" (rejects simulation before connecting)" if diagnostic else " (reports unsupported simulated firmware service)" if expected_error else ""), flush=True)
    # With idle stdin the keyboard example must still close and exit on Ctrl-C.
    keyboard = binaries / "advanced-keyboard-jog-joint"
    process = subprocess.Popen([str(keyboard), "--simulated", "--profile", "vega_1"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=environment)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(10):
                raise RuntimeError("keyboard example did not become ready")
            if "w/s:" not in process.stdout.readline():
                raise RuntimeError("keyboard example failed before input loop")
        process.send_signal(signal.SIGINT)
        # Keep stdin open: closing it would hide the blocked-input bug.
        status = process.wait(timeout=5)
        if status == 0:
            raise RuntimeError("interrupted keyboard example reported success")
        print("Verified idle keyboard Ctrl-C cleanup", flush=True)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    if failures:
        raise RuntimeError("\n\n".join(failures))
    print(f"Verified {len(tasks)} public Rust tasks without hardware.")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
