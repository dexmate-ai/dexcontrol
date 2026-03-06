#!/usr/bin/env python3
"""Analyze stage-wise directionality from NDJSON debug logs and write Markdown.

Targets stage messages:
- H30: ik_fk_projection_error
- H32: post_filter_fk_error
- H33: post_limits_fk_error
- H34: joint_direction_transfer
- H31: achieved_tracking_error
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


STAGE_ORDER = ["H30", "H32", "H33", "H34", "H31"]


def _vec3(values: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return arr[:3]


def _cos_perp(ref_vec: np.ndarray, actual_vec: np.ndarray, eps: float = 1e-9) -> tuple[float | None, float | None]:
    ref_norm = float(np.linalg.norm(ref_vec))
    act_norm = float(np.linalg.norm(actual_vec))
    if ref_norm <= eps or act_norm <= eps:
        return None, None
    u = ref_vec / ref_norm
    proj = float(np.dot(actual_vec, u))
    perp = actual_vec - proj * u
    cosine = float(np.dot(ref_vec, actual_vec) / (ref_norm * act_norm))
    perp_ratio = float(np.linalg.norm(perp) / act_norm)
    return cosine, perp_ratio


def _target_key(target_xyz: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.round(target_xyz.astype(np.float64), 6).tolist())


@dataclass
class StageStats:
    cosines: list[float] = field(default_factory=list)
    perps: list[float] = field(default_factory=list)

    def add(self, cosine: float | None, perp_ratio: float | None) -> None:
        if cosine is None or perp_ratio is None:
            return
        self.cosines.append(float(cosine))
        self.perps.append(float(perp_ratio))

    def summarize(self) -> dict[str, Any]:
        if not self.cosines:
            return {
                "n": 0,
                "cos_mean": None,
                "cos_p05": None,
                "cos_p50": None,
                "cos_lt_0_pct": None,
                "cos_lt_03_pct": None,
                "perp_mean": None,
                "perp_p95": None,
            }
        c = np.asarray(self.cosines, dtype=np.float64)
        p = np.asarray(self.perps, dtype=np.float64)
        return {
            "n": int(c.size),
            "cos_mean": float(c.mean()),
            "cos_p05": float(np.percentile(c, 5)),
            "cos_p50": float(np.percentile(c, 50)),
            "cos_lt_0_pct": float((c < 0).mean() * 100.0),
            "cos_lt_03_pct": float((c < 0.3).mean() * 100.0),
            "perp_mean": float(p.mean()),
            "perp_p95": float(np.percentile(p, 95)),
        }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _select_run(run_id: str | None, allow_substrings: list[str]) -> bool:
    if run_id is None:
        return not allow_substrings
    if not allow_substrings:
        return True
    return any(s in run_id for s in allow_substrings)


def _resolve_start_line_for_last_loops(
    log_path: Path,
    run_filters: list[str],
    last_loop_timing: int,
) -> tuple[int, int]:
    """Return (start_line, total_loop_timing_matches_in_filter)."""
    if last_loop_timing <= 0:
        return 1, 0
    matched_loop_lines: list[int] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = rec.get("runId")
            if not _select_run(run_id, run_filters):
                continue
            if rec.get("message") == "loop_timing":
                matched_loop_lines.append(line_no)
    if not matched_loop_lines:
        return 1, 0
    if len(matched_loop_lines) <= last_loop_timing:
        return matched_loop_lines[0], len(matched_loop_lines)
    return matched_loop_lines[-last_loop_timing], len(matched_loop_lines)


def _resolve_line_window_for_loop_chunk(
    log_path: Path,
    run_filters: list[str],
    chunk_size: int,
    chunk_index_from_end: int,
) -> tuple[int, int, int]:
    """Return (start_line, end_line, total_loop_timing_matches_in_filter)."""
    if chunk_size <= 0 or chunk_index_from_end <= 0:
        return 1, 10**18, 0
    matched_loop_lines: list[int] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = rec.get("runId")
            if not _select_run(run_id, run_filters):
                continue
            if rec.get("message") == "loop_timing":
                matched_loop_lines.append(line_no)
    total = len(matched_loop_lines)
    if total == 0:
        return 1, 10**18, 0
    end_idx_exclusive = total - (chunk_index_from_end - 1) * chunk_size
    start_idx = max(0, end_idx_exclusive - chunk_size)
    end_idx_exclusive = max(0, min(total, end_idx_exclusive))
    if start_idx >= end_idx_exclusive:
        return 1, 0, total
    start_line = matched_loop_lines[start_idx]
    end_line = matched_loop_lines[end_idx_exclusive - 1]
    return start_line, end_line, total


def analyze(
    log_path: Path,
    run_filters: list[str],
    start_line: int = 1,
    end_line: int = 10**18,
) -> dict[str, Any]:
    stage_stats_all = {s: StageStats() for s in STAGE_ORDER}
    stage_stats_arm = defaultdict(lambda: {s: StageStats() for s in STAGE_ORDER})
    message_counts = defaultdict(int)

    # Per-arm latest command context from H30; used by H32/H33.
    latest_cmd_by_arm: dict[str, dict[str, Any]] = {}
    # Per-arm target-key indexed queue from H30; used by H31 matching.
    h30_target_queue: dict[str, dict[tuple[float, float, float], deque[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(deque)
    )

    with log_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if line_no < start_line:
                continue
            if line_no > end_line:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            run_id = rec.get("runId")
            if not _select_run(run_id, run_filters):
                continue

            msg = rec.get("message")
            data = rec.get("data") or {}
            if not isinstance(data, dict):
                continue
            message_counts[msg] += 1

            if msg == "ik_fk_projection_error":
                arm = str(data.get("arm_side", "unknown"))
                current = _vec3(data.get("current_cart"))
                intended = _vec3(data.get("intended_target_cart"))
                ik_fk = _vec3(data.get("ik_fk_cart"))
                if current is None or intended is None or ik_fk is None:
                    continue
                intended_delta = intended - current
                h30_vec = ik_fk - current
                cos, perp = _cos_perp(intended_delta, h30_vec)
                stage_stats_all["H30"].add(cos, perp)
                stage_stats_arm[arm]["H30"].add(cos, perp)
                sample = {
                    "current_xyz": current,
                    "intended_delta_xyz": intended_delta,
                    "target_key": _target_key(intended),
                }
                latest_cmd_by_arm[arm] = sample
                h30_target_queue[arm][sample["target_key"]].append(sample)

            elif msg == "post_filter_fk_error":
                arm = str(data.get("arm_side", "unknown"))
                sample = latest_cmd_by_arm.get(arm)
                err = _vec3(data.get("post_filter_err_xyz"))
                if sample is None or err is None:
                    continue
                h32_vec = sample["intended_delta_xyz"] - err
                cos, perp = _cos_perp(sample["intended_delta_xyz"], h32_vec)
                stage_stats_all["H32"].add(cos, perp)
                stage_stats_arm[arm]["H32"].add(cos, perp)

            elif msg == "post_limits_fk_error":
                arm = str(data.get("arm_side", "unknown"))
                sample = latest_cmd_by_arm.get(arm)
                err = _vec3(data.get("post_limits_err_xyz"))
                if sample is None or err is None:
                    continue
                h33_vec = sample["intended_delta_xyz"] - err
                cos, perp = _cos_perp(sample["intended_delta_xyz"], h33_vec)
                stage_stats_all["H33"].add(cos, perp)
                stage_stats_arm[arm]["H33"].add(cos, perp)

            elif msg == "joint_direction_transfer":
                arm = str(data.get("arm_side", "unknown"))
                cos = data.get("direction_cosine")
                perp = data.get("perp_ratio")
                if isinstance(cos, (int, float)) and isinstance(perp, (int, float)):
                    stage_stats_all["H34"].add(float(cos), float(perp))
                    stage_stats_arm[arm]["H34"].add(float(cos), float(perp))

            elif msg == "achieved_tracking_error":
                arm = str(data.get("arm_side", "unknown"))
                target = _vec3(data.get("intended_target_cart"))
                achieved = _vec3(data.get("achieved_cart"))
                if target is None or achieved is None:
                    continue
                key = _target_key(target)
                queue = h30_target_queue[arm].get(key)
                if not queue:
                    continue
                sample = queue.popleft()
                h31_vec = achieved - sample["current_xyz"]
                cos, perp = _cos_perp(sample["intended_delta_xyz"], h31_vec)
                stage_stats_all["H31"].add(cos, perp)
                stage_stats_arm[arm]["H31"].add(cos, perp)

    return {
        "all": {s: stage_stats_all[s].summarize() for s in STAGE_ORDER},
        "by_arm": {
            arm: {s: stage_stats_arm[arm][s].summarize() for s in STAGE_ORDER}
            for arm in sorted(stage_stats_arm.keys())
        },
        "message_counts": dict(message_counts),
    }


def render_markdown(
    result: dict[str, Any],
    log_path: Path,
    run_filters: list[str],
    start_line: int,
    end_line: int,
    last_loop_timing: int,
    matched_loop_timing_total: int,
    loop_window_size: int,
    loop_window_index_from_end: int,
) -> str:
    lines: list[str] = []
    lines.append("# Stage Directionality Report")
    lines.append("")
    lines.append(f"- log_path: `{log_path}`")
    lines.append(f"- run_id_filters: `{run_filters if run_filters else 'ALL'}`")
    lines.append(f"- line_window_start: `{start_line}`")
    lines.append(f"- line_window_end: `{end_line}`")
    if last_loop_timing > 0:
        lines.append(
            f"- last_loop_timing_requested: `{last_loop_timing}` "
            f"(matched_total={matched_loop_timing_total})"
        )
    if loop_window_size > 0:
        lines.append(
            f"- loop_window: size={loop_window_size}, index_from_end={loop_window_index_from_end} "
            f"(matched_total={matched_loop_timing_total})"
        )
    lines.append("")
    lines.append("## Stage Definitions")
    lines.append("")
    lines.append("- `H30` (`ik_fk_projection_error`): intended Cartesian target vs FK of IK-solved joints.")
    lines.append("- `H32` (`post_filter_fk_error`): intended Cartesian target vs FK after smoothing/filter stage.")
    lines.append("- `H33` (`post_limits_fk_error`): intended Cartesian target vs FK after clip/jerk limits.")
    lines.append("- `H34` (`joint_direction_transfer`): commanded joint delta vs next-tick achieved joint delta.")
    lines.append("- `H31` (`achieved_tracking_error`): intended Cartesian target direction vs next-tick observed Cartesian movement.")
    lines.append("")
    lines.append("## Overall (All Arms)")
    lines.append("")
    lines.append("| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stage in STAGE_ORDER:
        s = result["all"][stage]
        lines.append(
            f"| {stage} | {_fmt(s['n'])} | {_fmt(s['cos_mean'])} | {_fmt(s['cos_p05'])} | "
            f"{_fmt(s['cos_p50'])} | {_fmt(s['cos_lt_0_pct'], 2)} | {_fmt(s['cos_lt_03_pct'], 2)} | "
            f"{_fmt(s['perp_mean'])} | {_fmt(s['perp_p95'])} |"
        )

    lines.append("")
    lines.append("## By Arm")
    lines.append("")
    for arm, arm_stats in result["by_arm"].items():
        lines.append(f"### arm: `{arm}`")
        lines.append("")
        lines.append("| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for stage in STAGE_ORDER:
            s = arm_stats[stage]
            lines.append(
                f"| {stage} | {_fmt(s['n'])} | {_fmt(s['cos_mean'])} | {_fmt(s['cos_p05'])} | "
                f"{_fmt(s['cos_p50'])} | {_fmt(s['cos_lt_0_pct'], 2)} | {_fmt(s['cos_lt_03_pct'], 2)} | "
                f"{_fmt(s['perp_mean'])} | {_fmt(s['perp_p95'])} |"
            )
        lines.append("")

    lines.append("## Raw Message Counts")
    lines.append("")
    for k, v in sorted(result["message_counts"].items()):
        lines.append(f"- `{k}`: {v}")

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- Higher `cos_mean` (closer to 1) is better direction match.")
    lines.append("- Lower `perp_mean` is better (less orthogonal contamination).")
    lines.append("- Typical degradation pattern in this pipeline is `H30~H33` stable, then drop at `H34/H31`.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage directionality markdown report from NDJSON logs")
    parser.add_argument(
        "--log-path",
        default="/home/dexmate/.cursor/debug-daf7f0.log",
        help="Path to NDJSON debug log",
    )
    parser.add_argument(
        "--output-md",
        default="/home/dexmate/custom_dexcontrol/examples/stage_direction_report.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--run-id-contains",
        action="append",
        default=[],
        help="Only include records whose runId contains this substring (repeatable)",
    )
    parser.add_argument(
        "--last-loop-timing",
        type=int,
        default=0,
        help="Use only the log tail starting from the N-th last loop_timing record (0=all)",
    )
    parser.add_argument(
        "--loop-window-size",
        type=int,
        default=0,
        help="Use exactly this many loop_timing records in a chunk window (0=disabled)",
    )
    parser.add_argument(
        "--loop-window-index-from-end",
        type=int,
        default=1,
        help="When --loop-window-size is set, select which chunk from end (1=latest)",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.output_md)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    last_loop_timing = max(0, int(args.last_loop_timing))
    loop_window_size = max(0, int(args.loop_window_size))
    loop_window_index_from_end = max(1, int(args.loop_window_index_from_end))

    if loop_window_size > 0:
        start_line, end_line, matched_total = _resolve_line_window_for_loop_chunk(
            log_path=log_path,
            run_filters=args.run_id_contains,
            chunk_size=loop_window_size,
            chunk_index_from_end=loop_window_index_from_end,
        )
    else:
        start_line, matched_total = _resolve_start_line_for_last_loops(
            log_path=log_path,
            run_filters=args.run_id_contains,
            last_loop_timing=last_loop_timing,
        )
        end_line = 10**18

    result = analyze(
        log_path=log_path,
        run_filters=args.run_id_contains,
        start_line=start_line,
        end_line=end_line,
    )
    md = render_markdown(
        result=result,
        log_path=log_path,
        run_filters=args.run_id_contains,
        start_line=start_line,
        end_line=end_line,
        last_loop_timing=last_loop_timing,
        matched_loop_timing_total=matched_total,
        loop_window_size=loop_window_size,
        loop_window_index_from_end=loop_window_index_from_end,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] wrote markdown report: {out_path}")


if __name__ == "__main__":
    main()
