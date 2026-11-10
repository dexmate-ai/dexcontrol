# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guide both arms by hand: yield to pushes sensed from joint current.

Capture each arm's joint positions and currents at rest, then at 200 Hz treat a per-joint current offset beyond --threshold A (default 0.25) as a push and send a joint setpoint that moves with it: each joint's velocity is the push divided by a damping of 2*gain (--gain, default 0.3; larger is stiffer, below 0.3 is refused), and no tick commands a setpoint more than 0.2 rad from the measured position. By default the arms stay wherever they are pushed; --hold-pose adds a spring back to the captured pose. Run until Ctrl-C or --duration seconds (default 0, unlimited), then stop both arms. No force sensor, no model and no identification: the current threshold is the only filter, so run it on a machine wired to the robot.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from dexcontrol import RateLimiter, Robot
from typing_extensions import Annotated

# Per-joint spring pattern of the 7-joint Vega arm, scaled by --gain when
# --hold-pose is set: the shoulder and elbow pitch joints carry the load.
SPRING = np.array([2.0, 0.25, 0.25, 2.0, 0.25, 0.25, 0.25])
# Largest move one tick may command, rad. A direct setpoint is checked
# against a fresh measurement with a 0.3 rad step guard; a slow loop or a
# hard shove must slow the arm down, not end the example.
MAX_STEP = 0.2


@dataclass
class Args:
    """Command-line options."""

    gain: Annotated[float, tyro.conf.arg(help="Stiffness/damping scale, >= 0.3")] = 0.3
    threshold: Annotated[float, tyro.conf.arg(help="Current offset that counts as a push, A")] = 0.25
    control_hz: float = 200.0
    duration: Annotated[float, tyro.conf.arg(help="Seconds to run; 0 = until Ctrl-C")] = 0.0
    hold_pose: Annotated[bool, tyro.conf.arg(help="Spring back to the pose captured at start")] = False
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )
    profile: Annotated[Optional[str], tyro.conf.arg(help="Built-in robot profile")] = (
        None
    )
    config: Annotated[
        Optional[str], tyro.conf.arg(help="Custom robot configuration")
    ] = None


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if args.gain < 0.3:
        raise SystemExit("--gain below 0.3 lets a current blip fling the arm; refusing")
    if args.threshold < 0.0:
        raise SystemExit("--threshold must not be negative")
    if args.control_hz <= 0.0:
        raise SystemExit("--control-hz must be positive")

    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        arms = {}
        for name in ("left_arm", "right_arm"):
            if not robot.has_component(name):
                raise SystemExit(f"this robot has no {name}")
            arm = robot.joints(name)
            if len(arm.joint_names) != len(SPRING):
                raise SystemExit(f"hand guiding is tuned for the {len(SPRING)}-joint Vega arm; {name} has {len(arm.joint_names)}")
            arms[name] = arm
        # A rest baseline: everything the joints draw to hold still here is
        # not a push.
        rest_pos = {name: np.array(arm.get_joint_pos()) for name, arm in arms.items()}
        rest_current = {name: np.array(arm.get_joint_current()) for name, arm in arms.items()}
        for name, current in rest_current.items():
            if current.shape != (len(SPRING),):
                raise SystemExit(f"{name} reports no per-joint current")
        damping = 2.0 * args.gain
        spring = SPRING * args.gain if args.hold_pose else np.zeros(len(SPRING))
        dt = 1.0 / args.control_hz
        print(f"hand guiding both arms at {args.control_hz:g} Hz; threshold {args.threshold:g} A; "
              f"{'spring back to the start pose' if args.hold_pose else 'float where pushed'}", flush=True)
        limiter = RateLimiter(args.control_hz)
        ticks = 0
        try:
            while args.duration <= 0.0 or ticks * dt < args.duration:
                for name, arm in arms.items():
                    position = np.array(arm.get_joint_pos())
                    offset = np.array(arm.get_joint_current()) - rest_current[name]
                    # Below the threshold is noise and the load of holding
                    # still; above it, the sign is inverted so the joint
                    # gives way to the push.
                    push = np.where(np.abs(offset) < args.threshold, 0.0, -offset)
                    velocity = (spring * (rest_pos[name] - position) + push) / damping
                    # The direct setpoint path, as the 0.5 script used: the
                    # motion plugin's own smoothing would lag the target and
                    # its catch-up current reads as a new push.
                    step = np.clip(velocity * dt, -MAX_STEP, MAX_STEP)
                    arm.set_joint_pos(position + step)
                ticks += 1
                limiter.sleep()
        except KeyboardInterrupt:
            print("stopped", flush=True)
        finally:
            # Stop every arm even if one stop raises; report the first failure.
            stop_errors = []
            for arm in arms.values():
                try:
                    arm.stop()
                except Exception as error:  # noqa: BLE001 - collected, then re-raised
                    stop_errors.append(error)
            if stop_errors:
                raise stop_errors[0]


if __name__ == "__main__":
    main()
