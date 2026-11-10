# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Display robot topology, observed state, and server diagnostics.

Use a read-only DiagnosticClient and a 0.25 s state observation window, then print the snapshot. Optional --enable-sensor includes declared optional sensors. Does not enable motors or issue stop/motion commands; no simulation connection is supported.

Formatted tables, optionally using Rich."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import tyro
from dexcontrol import DiagnosticClient
from typing_extensions import Annotated

try:
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    _console: Console | None = Console()
except ImportError:  # pragma: no cover
    _console = None

_DASH = "—"
_NA = "N/A"


def _plain_table(
    title: str, headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]
) -> None:
    rows = list(rows)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(f"\n{title}")
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("  ".join("─" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _table(title: str, columns: tuple[tuple[str, dict[str, Any]], ...]) -> Table | None:
    if _console is None:
        return None
    table = Table(
        title=title,
        title_justify="left",
        box=box.ROUNDED,
        header_style="bold",
        show_lines=False,
        pad_edge=True,
    )
    for label, options in columns:
        table.add_column(label, **options)
    return table


def _show(table: Table) -> None:
    assert _console is not None
    _console.print()
    _console.print(table)


def _number(values: list[Any], index: int, unit: str = "") -> str:
    if index >= len(values) or values[index] is None:
        return _DASH
    try:
        value = float(values[index])
    except (TypeError, ValueError):
        return str(values[index])
    return f"{value:+.4f}{unit}"


def _humanize(value: Any) -> str:
    return str(value if value is not None else _DASH).replace("_", " ").title()


def _end_effectors(snapshot: dict[str, Any]) -> str:
    """Vega's bring-up report: the hand each side configures and what the robot says is there."""
    bringup = snapshot.get("model_bringup") or {}
    report = bringup.get("end_effectors") if isinstance(bringup, dict) else None
    if not isinstance(report, dict):
        return _DASH
    # The core renders the line (same words as the connect log) so a
    # mismatch names the profile and the variant to select. It quotes the
    # robot's hand-type reply, so control characters are scrubbed before
    # they reach the terminal.
    summary = report.get("summary")
    if not isinstance(summary, str) or not summary:
        return _DASH
    return "".join(" " if ch.isspace() or not ch.isprintable() else ch for ch in summary)


def show_overview(snapshot: dict[str, Any]) -> None:
    rows = [
        ("Profile", str(snapshot.get("profile_name", _DASH))),
        ("Transport", "DexComm"),
        ("Components", ", ".join(snapshot.get("component_names", [])) or _DASH),
        ("Sensors", ", ".join(snapshot.get("sensor_names", [])) or "None enabled"),
        ("End effectors", _end_effectors(snapshot)),
    ]
    table = _table(
        "Robot overview", (("", {"style": "cyan", "no_wrap": True}), ("", {}))
    )
    if table is None:
        _plain_table("Robot overview", ("Field", "Value"), rows)
        return
    table.show_header = False
    for field, value in rows:
        table.add_row(field, escape(value))
    _show(table)


def show_joint_state(snapshot: dict[str, Any]) -> None:
    states = snapshot.get("joint_states", {})
    configured_names = snapshot.get("joint_names", {})
    errors_by_key = snapshot.get("errors", {})
    rows: list[tuple[str, ...]] = []
    # Iterate the configured components, not just the ones that reported:
    # a component that published nothing must render as N/A rather than
    # disappear, so it is distinguishable from one that genuinely published
    # zeros.
    for component in sorted(set(configured_names) | set(states)):
        state = states.get(component)
        if state is None:
            reason = errors_by_key.get(f"joint_state:{component}", "no state received")
            for joint in configured_names.get(component, []) or ["—"]:
                rows.append(
                    (component, joint, _NA, _NA, _NA, _NA, f"No data ({reason})")
                )
            continue
        positions = state.get("position", [])
        velocities = state.get("velocity", [])
        currents = state.get("current", [])
        torques = state.get("torque", [])
        names = configured_names.get(component, [])
        error_indices = set(state.get("error_joint_indices", []))
        errors = state.get("errors", {})
        count = max(len(names), len(positions))
        for index in range(count):
            joint = names[index] if index < len(names) else f"joint_{index + 1}"
            error = errors.get(joint) or errors.get(str(index))
            status = (
                str(error) if error else ("Error" if index in error_indices else "OK")
            )
            rows.append(
                (
                    component,
                    joint,
                    _number(positions, index),
                    _number(velocities, index),
                    _number(currents, index),
                    _number(torques, index),
                    status,
                )
            )
    if not rows:
        rows = [(_DASH, "No joint state received", _DASH, _DASH, _DASH, _DASH, _DASH)]

    headers = (
        "Component",
        "Joint",
        "Position",
        "Velocity",
        "Current",
        "Torque",
        "Status",
    )
    table = _table(
        "Joint state",
        (
            ("Component", {"style": "cyan", "no_wrap": True}),
            ("Joint", {"no_wrap": True}),
            ("Position [rad]", {"justify": "right", "no_wrap": True}),
            ("Velocity [rad/s]", {"justify": "right", "no_wrap": True}),
            ("Current [A]", {"justify": "right", "no_wrap": True}),
            ("Torque [Nm]", {"justify": "right", "no_wrap": True}),
            ("Status", {"no_wrap": True}),
        ),
    )
    if table is None:
        _plain_table("Joint state", headers, rows)
        return
    previous = None
    for component, joint, position, velocity, current, torque, status in rows:
        if previous is not None and component != previous:
            table.add_section()
        shown_component = component if component != previous else ""
        styled_status = (
            "[green]OK[/green]" if status == "OK" else f"[red]{escape(status)}[/red]"
        )
        table.add_row(
            shown_component,
            escape(joint),
            position,
            velocity,
            current,
            torque,
            styled_status,
        )
        previous = component
    _show(table)


def estop_row(estop: dict[str, Any]) -> tuple[str, str, str]:
    """Render the E-stop monitoring row, absence included.

    The robot server publishes E-stop state only while the button is engaged,
    so a released E-stop publishes nothing at all. That is the normal state of
    a healthy robot, not a missing diagnostic, and the snapshot says so with
    ``source: no_event_published`` plus ``state_known``, which is true only
    when the server also reports the E-stop board connected.
    """
    if estop.get("source") == "no_event_published":
        if estop.get("state_known"):
            return (
                "E-stop",
                "Ready",
                "not engaged (no event published; board connected)",
            )
        reason = (
            "board disconnected"
            if estop.get("board_connected") is False
            else "board status unavailable"
        )
        return ("E-stop", "Unknown", f"no event published; {reason}")
    pressed = any(
        bool(estop.get(name))
        for name in (
            "left_base_estop_enabled",
            "right_base_estop_enabled",
            "torso_estop_enabled",
            "remote_estop_enabled",
        )
    )
    software = bool(estop.get("software_estop_enabled"))
    details = f"button={'pressed' if pressed else 'released'}, software={'on' if software else 'off'}"
    return ("E-stop", "STOPPED" if pressed or software else "Clear", details)


def show_monitoring(snapshot: dict[str, Any]) -> None:
    monitoring = snapshot.get("monitoring", {})
    rows: list[tuple[str, str, str]] = []
    estop = monitoring.get("estop")
    if isinstance(estop, dict):
        rows.append(estop_row(estop))
    heartbeat = monitoring.get("heartbeat")
    if isinstance(heartbeat, dict):
        if heartbeat.get("monitoring_disabled"):
            status = "Disabled"
        elif heartbeat.get("paused"):
            status = "Paused"
        else:
            status = "Active" if heartbeat.get("is_active") else "No sample"
        rows.append(("Heartbeat", status, "Native supervision"))
    battery = monitoring.get("battery")
    if isinstance(battery, dict):
        percentage = battery.get("percentage")
        status = f"{float(percentage):.0f}%" if percentage is not None else _DASH
        voltage = battery.get("voltage")
        current = battery.get("current")
        details = (
            ", ".join(
                value
                for value in (
                    f"{float(voltage):.2f} V" if voltage is not None else "",
                    f"{float(current):.2f} A" if current is not None else "",
                    f"{float(battery['power']):.2f} W" if battery.get("power") is not None else "",
                    f"{float(battery['temperature']):.2f} °C" if battery.get("temperature") is not None else "",
                )
                if value
            )
            or _DASH
        )
        rows.append(("Battery", status, details))
    if not rows:
        return

    table = _table(
        "Monitoring",
        (
            ("Monitor", {"style": "cyan", "no_wrap": True}),
            ("Status", {"no_wrap": True}),
            ("Details", {}),
        ),
    )
    if table is None:
        _plain_table("Monitoring", ("Monitor", "Status", "Details"), rows)
        return
    for monitor, status, details in rows:
        healthy = status in {"Clear", "Ready", "Active"} or status.endswith("%")
        unknown = {"Paused", "Disabled", "No sample", "Unknown"}
        color = "green" if healthy else "yellow" if status in unknown else "red"
        table.add_row(monitor, f"[{color}]{escape(status)}[/{color}]", escape(details))
    _show(table)


def show_versions(snapshot: dict[str, Any]) -> None:
    info = snapshot.get("version_info")
    if not isinstance(info, dict):
        rows = [("Server versions", "Unavailable")]
        _plain_table("Versions", ("Item", "Value"), rows)
        return
    firmware = info.get("firmware_version", {})
    rows = []
    for board, entry in sorted(firmware.items()):
        rows.append(
            (
                board,
                str(entry.get("hardware_version", _DASH)),
                str(entry.get("software_version", _DASH)),
                str(entry.get("release_version", _DASH)),
                str(entry.get("main_hash", _DASH)),
                str(entry.get("compile_time", _DASH)),
            )
        )
    table = _table(
        "Firmware versions",
        (
            ("Board", {"style": "cyan"}),
            ("Hardware", {"justify": "right"}),
            ("Software", {"justify": "right"}),
            ("Release", {"justify": "right"}),
            ("Commit", {}),
            ("Compiled", {"justify": "right"}),
        ),
    )
    if table is None:
        _plain_table(
            "Firmware versions",
            ("Board", "Hardware", "Software", "Release", "Commit", "Compiled"),
            rows,
        )
        return
    for row in rows:
        table.add_row(*[escape(value) for value in row])
    server = info.get("version", _DASH)
    minimum = info.get("min_client_version", _DASH)
    table.caption = f"Robot server {server}  •  minimum client {minimum}"
    _show(table)


def show_component_status(snapshot: dict[str, Any]) -> None:
    report = snapshot.get("component_status") or {}
    states = report.get("states", {}) if isinstance(report, dict) else {}
    configured = set(snapshot.get("component_names", [])) | set(
        snapshot.get("sensor_names", [])
    )
    rows = []
    for name, state in sorted(states.items()):
        error = state.get("error") if isinstance(state, dict) else None
        message = error.get("error_message", "") if isinstance(error, dict) else ""
        rows.append(
            (
                name,
                _humanize(state.get("connection") if isinstance(state, dict) else None),
                _humanize(state.get("operation") if isinstance(state, dict) else None),
                message or _DASH,
                "Configured" if name in configured else "Server only",
            )
        )
    if not rows:
        rows = [(_DASH, "Unavailable", _DASH, _DASH, _DASH)]
    table = _table(
        "Component status",
        (
            ("Component", {"style": "cyan", "no_wrap": True}),
            ("Connection", {"no_wrap": True}),
            ("Operation", {"no_wrap": True}),
            ("Error", {}),
            ("Scope", {"style": "dim", "no_wrap": True}),
        ),
    )
    if table is None:
        _plain_table(
            "Component status",
            ("Component", "Connection", "Operation", "Error", "Scope"),
            rows,
        )
        return
    for name, connection, operation, error, scope in rows:
        connection_style = "green" if connection == "Connected" else "red"
        operation_style = "green" if operation == "Enabled" else "yellow"
        table.add_row(
            escape(name),
            f"[{connection_style}]{escape(connection)}[/{connection_style}]",
            f"[{operation_style}]{escape(operation)}[/{operation_style}]",
            escape(error),
            scope,
            style="dim" if scope == "Server only" else None,
        )
    _show(table)


def show_ntp(snapshot: dict[str, Any]) -> None:
    ntp = snapshot.get("ntp")
    if not isinstance(ntp, dict):
        return
    success = ntp.get("success")
    rows: list[tuple[str, str]] = []
    if success is False:
        rows.append(("Status", "Unavailable"))
        rows.append(("Message", str(ntp.get("message") or "Query failed")))
    else:
        rows.append(("Status", "Synchronized" if success is True else "Reported"))
        if "offset" in ntp:
            rows.append(("Clock offset", f"{float(ntp['offset']) * 1_000:+.3f} ms"))
        if "rtt" in ntp:
            rows.append(("Round-trip time", f"{float(ntp['rtt']) * 1_000:.3f} ms"))
        if "replies" in ntp or "samples" in ntp:
            rows.append(
                (
                    "Samples",
                    f"{ntp.get('replies', 0)} / {ntp.get('samples', 0)} replies",
                )
            )
        for key, value in ntp.items():
            if key in {"success", "message", "offset", "rtt", "replies", "samples"}:
                continue
            rows.append((_humanize(key), str(value)))
        if ntp.get("message"):
            rows.append(("Message", str(ntp["message"])))
    table = _table(
        "Clock synchronization", (("Metric", {"style": "cyan"}), ("Value", {}))
    )
    if table is None:
        _plain_table("Clock synchronization", ("Metric", "Value"), rows)
        return
    for metric, value in rows:
        table.add_row(escape(metric), escape(value))
    _show(table)


def show_connection_timing(snapshot: dict[str, Any]) -> None:
    connection = snapshot.get("connection", {})
    phase_labels = {
        "configuration": "Configuration",
        "discovery": "Transport discovery",
        "operational_config": "Operational model",
        "subscription_activation": "Subscriptions",
        "version_check": "Version check",
        "safety_startup": "Safety setup",
        "readiness": "Readiness",
    }
    rows = [
        (
            phase_labels.get(phase.get("phase"), _humanize(phase.get("phase"))),
            f"{float(phase.get('duration_ms', 0.0)):.2f} ms",
        )
        for phase in connection.get("phases", [])
    ]
    rows.append(("Total", f"{float(connection.get('total_ms', 0.0)):.2f} ms"))
    table = _table(
        "Connection timing",
        (
            ("Phase", {"style": "cyan"}),
            ("Duration", {"justify": "right", "no_wrap": True}),
        ),
    )
    if table is None:
        _plain_table("Connection timing", ("Phase", "Duration"), rows)
        return
    for index, (phase, duration) in enumerate(rows):
        table.add_row(
            escape(phase),
            duration,
            style="bold" if index == len(rows) - 1 else None,
            end_section=index == len(rows) - 2,
        )
    _show(table)


def show_errors(snapshot: dict[str, Any]) -> None:
    errors = snapshot.get("errors", {})
    if not errors:
        return
    rows = [(name, str(message)) for name, message in sorted(errors.items())]
    table = _table(
        "Unavailable diagnostics",
        (("Section", {"style": "yellow", "no_wrap": True}), ("Reason", {})),
    )
    if table is None:
        _plain_table("Unavailable diagnostics", ("Section", "Reason"), rows)
        return
    for section, reason in rows:
        table.add_row(escape(section), escape(reason))
    _show(table)


def show_snapshot(snapshot: dict[str, Any]) -> None:
    show_overview(snapshot)
    show_joint_state(snapshot)
    show_monitoring(snapshot)
    show_versions(snapshot)
    show_component_status(snapshot)
    show_ntp(snapshot)
    show_connection_timing(snapshot)
    show_errors(snapshot)


@dataclass
class Args:
    """Command-line options."""

    profile: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="built-in robot profile (default: auto-detect from ROBOT_NAME)"
        ),
    ] = None
    config: Annotated[
        Optional[str], tyro.conf.arg(help="path to a custom robot YAML file")
    ] = None
    enable_sensor: Annotated[
        List[str],
        tyro.conf.UseAppendAction,
        tyro.conf.arg(
            help="enable an optional configured sensor; repeat for multiple sensors",
            metavar="NAME",
        ),
    ] = field(default_factory=lambda: [])
    settle_timeout: Annotated[
        float,
        tyro.conf.arg(
            help="maximum state observation window in seconds (default: 0.25)"
        ),
    ] = 0.25


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if args.profile is not None and args.config is not None:
        raise SystemExit("--profile and --config are mutually exclusive")

    options: dict[str, Any] = {"enable_sensors": args.enable_sensor}
    if args.config:
        options["config_file"] = args.config
    elif args.profile is not None:
        options["profile"] = args.profile

    with DiagnosticClient(**options) as client:
        snapshot = client.snapshot(settle_timeout=max(args.settle_timeout, 0.0))
    show_snapshot(snapshot)


if __name__ == "__main__":
    main()
