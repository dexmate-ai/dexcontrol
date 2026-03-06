# Stage Directionality Report

- log_path: `/home/dexmate/.cursor/debug-daf7f0.log`
- run_id_filters: `ALL`
- line_window_start: `5123`
- line_window_end: `10153`
- loop_window: size=240, index_from_end=4 (matched_total=1200)

## Stage Definitions

- `H30` (`ik_fk_projection_error`): intended Cartesian target vs FK of IK-solved joints.
- `H32` (`post_filter_fk_error`): intended Cartesian target vs FK after smoothing/filter stage.
- `H33` (`post_limits_fk_error`): intended Cartesian target vs FK after clip/jerk limits.
- `H34` (`joint_direction_transfer`): commanded joint delta vs next-tick achieved joint delta.
- `H31` (`achieved_tracking_error`): intended Cartesian target direction vs next-tick observed Cartesian movement.

## Overall (All Arms)

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 238 | 0.9734 | 0.9361 | 0.9766 | 0.00 | 0.00 | 0.2037 | 0.3517 |
| H32 | 238 | 0.9727 | 0.9320 | 0.9805 | 0.00 | 0.00 | 0.2011 | 0.3625 |
| H33 | 238 | 0.9727 | 0.9320 | 0.9805 | 0.00 | 0.00 | 0.2011 | 0.3625 |
| H34 | 478 | 0.7544 | -0.9609 | 0.9464 | 9.00 | 10.88 | 0.3517 | 0.9101 |
| H31 | 238 | 0.7436 | -0.8842 | 0.9202 | 9.66 | 11.34 | 0.3881 | 0.7514 |

## By Arm

### arm: `left`

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 119 | 0.9827 | 0.9488 | 0.9927 | 0.00 | 0.00 | 0.1544 | 0.3159 |
| H32 | 119 | 0.9819 | 0.9436 | 0.9924 | 0.00 | 0.00 | 0.1543 | 0.3310 |
| H33 | 119 | 0.9819 | 0.9436 | 0.9924 | 0.00 | 0.00 | 0.1543 | 0.3310 |
| H34 | 239 | 0.7460 | -0.9543 | 0.9422 | 9.62 | 11.30 | 0.3575 | 0.9292 |
| H31 | 119 | 0.7507 | -0.9359 | 0.9464 | 10.08 | 11.76 | 0.3433 | 0.7514 |

### arm: `right`

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 119 | 0.9640 | 0.9338 | 0.9635 | 0.00 | 0.00 | 0.2531 | 0.3577 |
| H32 | 119 | 0.9636 | 0.9310 | 0.9700 | 0.00 | 0.00 | 0.2479 | 0.3649 |
| H33 | 119 | 0.9636 | 0.9310 | 0.9700 | 0.00 | 0.00 | 0.2479 | 0.3649 |
| H34 | 239 | 0.7628 | -0.9608 | 0.9496 | 8.37 | 10.46 | 0.3459 | 0.8683 |
| H31 | 119 | 0.7365 | -0.8548 | 0.9086 | 9.24 | 10.92 | 0.4330 | 0.6667 |

## Raw Message Counts

- `achieved_tracking_error`: 478
- `axis_segment_start`: 11
- `ik_fk_projection_error`: 478
- `ik_solve_result`: 956
- `ik_solve_start`: 956
- `joint_direction_transfer`: 478
- `joint_update_metrics`: 478
- `loop_timing`: 240
- `post_filter_fk_error`: 478
- `post_limits_fk_error`: 478

## Interpretation Guide

- Higher `cos_mean` (closer to 1) is better direction match.
- Lower `perp_mean` is better (less orthogonal contamination).
- Typical degradation pattern in this pipeline is `H30~H33` stable, then drop at `H34/H31`.
