"""Recovery script for when a joint is stuck beyond its limit.

When a joint exceeds its position limit, the motor locks up and won't move
even after power cycling. This script helps recover by:

1. Showing current joint positions and which joints are over-limit
2. Auto-recovery: clear error + motor command to drive joints back (recommended)
3. Manual release: release brake so you can move joints by hand
4. Re-engaging the brake / re-enabling motors after recovery

Usage:
  # Check which joints are over-limit (read-only, safe to run anytime)
  python recover_joint_limit.py status --side left

  # Auto-recover: scan BOTH arms, release all over-limit joints (recommended)
  python recover_joint_limit.py auto
  python recover_joint_limit.py auto --side right
  python recover_joint_limit.py auto --side right --joints 3

  # Release specific joints for manual recovery
  python recover_joint_limit.py release --side left --joints 2 5

  # Release ALL joints on left arm
  python recover_joint_limit.py release --side left

  # Re-engage after manually moving joints back within limits
  python recover_joint_limit.py engage --side left

WARNING: When joints are released, the arm will be limp and may fall
due to gravity. Support the arm before releasing!
"""

import logging
import sys
import time
from typing import Literal

import numpy as np
import tyro

from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_arm(bot: Robot, side: str):
    """Get arm component by side name."""
    if side == "left":
        return bot.left_arm
    elif side == "right":
        return bot.right_arm
    else:
        raise ValueError(f"Invalid side: {side}")


def status(
    side: Literal["left", "right"] = "right",
) -> None:
    """Show current joint positions and highlight any that are over-limit.

    This is read-only and safe to run anytime.

    Args:
        side: Which arm to check.
    """
    with Robot() as bot:
        arm = _get_arm(bot, side)
        positions = arm.get_joint_pos()
        limits = arm.joint_pos_limit  # shape (7, 2): [min, max]
        names = arm.joint_name

        print(f"\n{'='*60}")
        print(f"  {side.upper()} ARM JOINT STATUS")
        print(f"{'='*60}")
        print(f"  {'Joint':<12} {'Position':>10} {'Min':>10} {'Max':>10}  Status")
        print(f"  {'-'*56}")

        over_limit_joints = []
        for i in range(len(positions)):
            pos = positions[i]
            lo, hi = limits[i]
            name = names[i] if names else f"joint_{i}"

            if pos < lo:
                flag = " << UNDER LIMIT"
                over_limit_joints.append(i)
            elif pos > hi:
                flag = " >> OVER LIMIT"
                over_limit_joints.append(i)
            else:
                flag = " OK"

            print(f"  {name:<12} {pos:>10.4f} {lo:>10.4f} {hi:>10.4f} {flag}")

        print(f"{'='*60}")
        if over_limit_joints:
            print(f"\n  Joints over limit: {over_limit_joints}")
            print(f"  To recover, run:")
            print(f"    python {__file__} release --side {side} --joints {' '.join(str(j) for j in over_limit_joints)}")
        else:
            print(f"\n  All joints are within limits.")
        print()


def release(
    side: Literal["left", "right"] = "left",
    joints: list[int] | None = None,
) -> None:
    """Release joints so they can be manually moved back within limits.

    Tries release_brake first (full free-drive). If that service is
    unavailable, falls back to set_modes(disable) which removes motor
    torque but may keep the mechanical brake engaged.

    Args:
        side: Which arm to release.
        joints: Specific joint indices (0-6) to release. If None, releases all.
    """
    if joints is None:
        joints = list(range(7))

    for j in joints:
        if j < 0 or j > 6:
            logger.error(f"Invalid joint index: {j}. Must be 0-6.")
            sys.exit(1)

    with Robot() as bot:
        arm = _get_arm(bot, side)

        # Show current status first
        positions = arm.get_joint_pos()
        limits = arm.joint_pos_limit
        names = arm.joint_name

        print(f"\n  Joints to release on {side} arm: {joints}")
        for j in joints:
            pos = positions[j]
            lo, hi = limits[j]
            name = names[j] if names else f"joint_{j}"
            in_limit = "OK" if lo <= pos <= hi else "OVER LIMIT"
            print(f"    [{j}] {name}: {pos:.4f}  (limits: [{lo:.4f}, {hi:.4f}])  {in_limit}")

        print(f"\n  WARNING: The arm may fall due to gravity when released!")
        print(f"  Make sure you are supporting the arm before continuing.")
        confirm = input("\n  Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return

        # Step 1: Disable motor torque (so motor stops holding position)
        modes = ["position"] * 7
        for j in joints:
            modes[j] = "disable"
        try:
            arm.set_modes(modes)
            logger.info(f"Motor torque disabled for joints {joints}.")
        except Exception as e:
            logger.error(f"Failed to disable motors: {e}")
            sys.exit(1)

        # Step 2: Release mechanical brake (so joint can move freely)
        brake_released = False
        try:
            logger.info(f"Attempting brake release for {side} arm joints {joints}...")
            result = arm.release_brake(enable=True, joints=joints)
            if result.get("success", False):
                brake_released = True
                logger.info(f"Brake released successfully: {result.get('message', '')}")
            else:
                logger.warning(f"Brake release returned failure: {result.get('message', '')}")
        except ServiceUnavailableError:
            logger.warning(f"Brake service not available for {side} arm. Motor torque is off but mechanical brake may resist.")
        except Exception as e:
            logger.warning(f"Brake release failed: {e}. Motor torque is off but mechanical brake may resist.")

        print(f"\n  Joints released. Manually move the arm back within limits.")
        print(f"  When done, run:")
        print(f"    python {__file__} engage --side {side}")
        print()

        # Keep the script running so the robot connection stays alive
        print("  Press Ctrl+C when you're done moving the arm...")
        try:
            while True:
                time.sleep(1)
                # Periodically show position
                positions = arm.get_joint_pos()
                status_parts = []
                for j in joints:
                    pos = positions[j]
                    lo, hi = limits[j]
                    if lo <= pos <= hi:
                        status_parts.append(f"  j{j}: {pos:.3f} OK")
                    else:
                        status_parts.append(f"  j{j}: {pos:.3f} !!")
                print(f"\r  {' | '.join(status_parts)}", end="", flush=True)
        except KeyboardInterrupt:
            print("\n")
            logger.info("Stopping...")

            # Re-engage brake first, then re-enable motors
            if brake_released:
                try:
                    arm.release_brake(enable=False)
                    logger.info("Brake re-engaged.")
                except Exception:
                    logger.warning("Could not re-engage brake automatically.")

            try:
                arm.set_modes(["position"] * 7)
                logger.info("All motors re-enabled in position mode.")
            except Exception:
                logger.warning("Could not re-enable motors automatically.")

            # Final status check
            positions = arm.get_joint_pos()
            all_ok = True
            for j in joints:
                pos = positions[j]
                lo, hi = limits[j]
                if pos < lo or pos > hi:
                    all_ok = False
                    name = names[j] if names else f"joint_{j}"
                    logger.warning(f"  {name} (j{j}) is still over limit: {pos:.4f} (limits: [{lo:.4f}, {hi:.4f}])")

            if all_ok:
                logger.info("All released joints are now within limits!")
            else:
                logger.warning("Some joints are still over limit. You may need to run this again.")


def auto(
    side: Literal["left", "right", "both"] = "both",
    joints: list[int] | None = None,
    margin: float = 0.05,
    move_time: float = 5.0,
) -> None:
    """Automatically recover over-limit joints.

    Scans arms for over-limit joints, tries motor commands first, then
    falls back to brake release for manual recovery. Monitors all joints
    and auto re-engages when everything is back within limits.

    Args:
        side: Which arm(s) to recover. Default "both" scans left and right.
        joints: Specific joint indices (0-6) to recover. If None, auto-detects
            all over-limit joints. Only used when side is "left" or "right".
        margin: How far inside the limit to target (rad).
        move_time: Time for the motor to reach target position (seconds).
    """
    sides = ["left", "right"] if side == "both" else [side]

    with Robot() as bot:
        # --- Phase 1: Scan all arms and detect over-limit joints ---
        # Per-arm data: {side: {arm, part, positions, limits, names, over_joints}}
        arm_data = {}
        for s in sides:
            arm = _get_arm(bot, s)
            positions = arm.get_joint_pos()
            limits = arm.joint_pos_limit
            names = arm.joint_name

            if side != "both" and joints is not None:
                over = joints
            else:
                over = []
                for i in range(len(positions)):
                    if positions[i] < limits[i][0] or positions[i] > limits[i][1]:
                        over.append(i)

            if over:
                arm_data[s] = {
                    "arm": arm,
                    "part": f"{s}_arm",
                    "positions": positions,
                    "limits": limits,
                    "names": names,
                    "over_joints": over,
                }

        if not arm_data:
            logger.info("No over-limit joints on any arm. Nothing to do.")
            return

        # --- Phase 2: Display summary ---
        for s, d in arm_data.items():
            print(f"\n  Over-limit joints on {s} arm:")
            for j in d["over_joints"]:
                pos = d["positions"][j]
                lo, hi = d["limits"][j]
                name = d["names"][j] if d["names"] else f"joint_{j}"
                over = pos - hi if pos > hi else lo - pos
                print(f"    [{j}] {name}: {pos:.4f}  limit=[{lo:.4f}, {hi:.4f}]  over by {over:.4f} rad ({np.degrees(over):.1f} deg)")

        print(f"\n  Will try motor command first, then brake release if needed.")
        print(f"  Be ready to press e-stop if needed!")
        confirm = input("\n  Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return

        # --- Phase 3: Try motor command recovery per arm ---
        # Track which joints still need brake release after motor attempt
        brake_needed = {}  # {side: [joint_indices]}

        for s, d in arm_data.items():
            arm = d["arm"]
            part = d["part"]
            over_joints = d["over_joints"]
            original_limits = arm.joint_pos_limit.copy()
            d["original_limits"] = original_limits

            # Expand software limits
            for j in over_joints:
                pos = d["positions"][j]
                lo, hi = d["limits"][j]
                arm._joint_pos_limit[j] = [
                    min(lo, pos - 0.1),
                    max(hi, pos + 0.1),
                ]
            logger.info(f"[{s}] Software limits temporarily expanded.")

            # Clear error
            try:
                bot.clear_error(part)
                logger.info(f"[{s}] Cleared error.")
            except Exception as e:
                logger.warning(f"[{s}] clear_error: {e}")

            # Re-engage brake if it was released
            try:
                arm.release_brake(enable=False)
            except Exception:
                pass

            # Set position mode + send motor command
            arm.set_modes(["position"] * 7)
            time.sleep(0.5)

            target_pos = arm.get_joint_pos().copy()
            for j in over_joints:
                lo, hi = original_limits[j]
                if target_pos[j] > hi:
                    target_pos[j] = hi - margin
                elif target_pos[j] < lo:
                    target_pos[j] = lo + margin

            logger.info(f"[{s}] Attempting motor command...")
            try:
                arm.set_joint_pos(target_pos, wait_time=move_time)
            except Exception as e:
                logger.warning(f"[{s}] set_joint_pos failed: {e}")

            # Check which are still over limit
            current_pos = arm.get_joint_pos()
            still_over = []
            for j in over_joints:
                lo, hi = original_limits[j]
                if current_pos[j] < lo or current_pos[j] > hi:
                    still_over.append(j)

            # Restore software limits
            arm._joint_pos_limit[:] = original_limits
            logger.info(f"[{s}] Software limits restored.")

            if not still_over:
                try:
                    bot.clear_error(part)
                except Exception:
                    pass
                logger.info(f"[{s}] All joints recovered via motor command!")
            else:
                still_names = [d["names"][j] for j in still_over]
                logger.warning(f"[{s}] Motor didn't move joints {still_over} ({still_names}).")
                brake_needed[s] = still_over

        # If everything recovered via motor, we're done
        if not brake_needed:
            logger.info("All joints recovered!")
            print()
            return

        # --- Phase 4: Brake release for remaining joints ---
        logger.info("Falling back to brake release for manual recovery...")
        brake_released = {}  # {side: bool}

        for s, still_over in brake_needed.items():
            d = arm_data[s]
            arm = d["arm"]

            # Disable motor torque on stuck joints
            modes = ["position"] * 7
            for j in still_over:
                modes[j] = "disable"
            arm.set_modes(modes)
            logger.info(f"[{s}] Motor torque disabled for joints {still_over}.")

            # Release brake
            brake_released[s] = False
            try:
                result = arm.release_brake(enable=True, joints=still_over)
                if result.get("success", False):
                    brake_released[s] = True
                    logger.info(f"[{s}] Brake released: {result.get('message', '')}")
                else:
                    logger.warning(f"[{s}] Brake release failed: {result.get('message', '')}")
            except ServiceUnavailableError:
                logger.warning(f"[{s}] Brake service not available.")
            except Exception as e:
                logger.warning(f"[{s}] Brake release error: {e}")

        # --- Phase 5: Monitor all joints across all arms ---
        print(f"\n  Released joints:")
        for s, still_over in brake_needed.items():
            names = arm_data[s]["names"]
            joint_desc = ", ".join(f"j{j}({names[j]})" for j in still_over)
            print(f"    {s}: {joint_desc}")
        print(f"\n  Push them back within limits.")
        print(f"  Monitoring... (will auto re-engage when all OK, or Ctrl+C to stop)\n")

        try:
            while True:
                time.sleep(0.5)
                parts = []
                all_ok = True

                for s, still_over in brake_needed.items():
                    arm = arm_data[s]["arm"]
                    original_limits = arm_data[s]["original_limits"]
                    current_pos = arm.get_joint_pos()

                    for j in still_over:
                        pos = current_pos[j]
                        lo, hi = original_limits[j]
                        label = f"{s[0].upper()}j{j}"
                        if lo <= pos <= hi:
                            parts.append(f"{label}: {pos:.3f} OK")
                        else:
                            parts.append(f"{label}: {pos:.3f} !!")
                            all_ok = False

                print(f"\r  {' | '.join(parts)}    ", end="", flush=True)

                if all_ok:
                    print()
                    logger.info("All joints within limits!")
                    break
        except KeyboardInterrupt:
            print()

        # --- Phase 6: Re-engage all arms ---
        for s in brake_needed:
            arm = arm_data[s]["arm"]
            part = arm_data[s]["part"]

            if brake_released.get(s, False):
                try:
                    arm.release_brake(enable=False)
                    logger.info(f"[{s}] Brake re-engaged.")
                except Exception:
                    logger.warning(f"[{s}] Could not re-engage brake.")

            arm.set_modes(["position"] * 7)
            logger.info(f"[{s}] All motors re-enabled in position mode.")

            try:
                bot.clear_error(part)
            except Exception:
                pass

        # --- Final check ---
        print(f"\n  Recovery result:")
        all_ok = True
        for s, d in arm_data.items():
            original_limits = d["original_limits"]
            arm = d["arm"]
            final_pos = arm.get_joint_pos()
            for j in d["over_joints"]:
                pos = final_pos[j]
                lo, hi = original_limits[j]
                name = d["names"][j] if d["names"] else f"joint_{j}"
                label = f"{s[0].upper()} [{j}] {name}"
                if lo <= pos <= hi:
                    print(f"    {label}: {pos:.4f}  OK")
                else:
                    all_ok = False
                    print(f"    {label}: {pos:.4f}  STILL OVER LIMIT")

        if all_ok:
            logger.info("All joints recovered!")
        else:
            logger.warning("Some joints still over limit.")
        print()


def engage(
    side: Literal["left", "right"] = "left",
) -> None:
    """Re-engage motors and brakes after manual recovery.

    Args:
        side: Which arm to re-engage.
    """
    with Robot() as bot:
        arm = _get_arm(bot, side)

        # Try to disable brake release
        try:
            result = arm.release_brake(enable=False)
            logger.info(f"Brake re-engaged: {result.get('message', '')}")
        except (ServiceUnavailableError, Exception) as e:
            logger.info(f"Brake service not used (may not be needed): {e}")

        # Re-enable all motors in position mode
        try:
            arm.set_modes(["position"] * 7)
            logger.info(f"All {side} arm motors re-enabled in position mode.")
        except Exception as e:
            logger.error(f"Failed to re-enable motors: {e}")
            sys.exit(1)

        # Show final status
        positions = arm.get_joint_pos()
        limits = arm.joint_pos_limit
        names = arm.joint_name
        print(f"\n  Final joint positions ({side} arm):")
        all_ok = True
        for i in range(len(positions)):
            pos = positions[i]
            lo, hi = limits[i]
            name = names[i] if names else f"joint_{i}"
            if pos < lo or pos > hi:
                all_ok = False
                print(f"    [{i}] {name}: {pos:.4f}  STILL OVER LIMIT [{lo:.4f}, {hi:.4f}]")
            else:
                print(f"    [{i}] {name}: {pos:.4f}  OK")

        if all_ok:
            print(f"\n  All joints within limits. Robot should be operational.")
        else:
            print(f"\n  Some joints still over limit. Robot may not move properly.")
        print()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {"status": status, "auto": auto, "release": release, "engage": engage}
    )
