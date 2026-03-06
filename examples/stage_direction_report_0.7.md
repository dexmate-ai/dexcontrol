# Stage Directionality Report

- log_path: `/home/dexmate/.cursor/debug-daf7f0.log`
- run_id_filters: `ALL`
- line_window_start: `10223`
- line_window_end: `15253`
- loop_window: size=240, index_from_end=3 (matched_total=1200)

## Stage Definitions

- `H30` (`ik_fk_projection_error`): intended Cartesian target vs FK of IK-solved joints.
- `H32` (`post_filter_fk_error`): intended Cartesian target vs FK after smoothing/filter stage.
- `H33` (`post_limits_fk_error`): intended Cartesian target vs FK after clip/jerk limits.
- `H34` (`joint_direction_transfer`): commanded joint delta vs next-tick achieved joint delta.
- `H31` (`achieved_tracking_error`): intended Cartesian target direction vs next-tick observed Cartesian movement.

## Overall (All Arms)

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 238 | 0.9707 | 0.9079 | 0.9810 | 0.00 | 0.00 | 0.1991 | 0.4192 |
| H32 | 238 | 0.9707 | 0.9079 | 0.9810 | 0.00 | 0.00 | 0.1991 | 0.4192 |
| H33 | 238 | 0.9707 | 0.9079 | 0.9810 | 0.00 | 0.00 | 0.1991 | 0.4192 |
| H34 | 478 | 0.7509 | -0.9327 | 0.9460 | 9.83 | 12.34 | 0.3538 | 0.9261 |
| H31 | 238 | 0.7481 | -0.7836 | 0.8865 | 9.24 | 10.92 | 0.3997 | 0.8687 |

## By Arm

### arm: `left`

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 119 | 0.9849 | 0.9472 | 0.9908 | 0.00 | 0.00 | 0.1426 | 0.3206 |
| H32 | 119 | 0.9849 | 0.9472 | 0.9908 | 0.00 | 0.00 | 0.1426 | 0.3206 |
| H33 | 119 | 0.9849 | 0.9472 | 0.9908 | 0.00 | 0.00 | 0.1426 | 0.3206 |
| H34 | 239 | 0.7425 | -0.8594 | 0.9451 | 11.30 | 12.97 | 0.3573 | 0.9077 |
| H31 | 119 | 0.7898 | -0.7846 | 0.9564 | 7.56 | 9.24 | 0.3366 | 0.6636 |

### arm: `right`

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 119 | 0.9565 | 0.8835 | 0.9679 | 0.00 | 0.00 | 0.2555 | 0.4684 |
| H32 | 119 | 0.9565 | 0.8835 | 0.9679 | 0.00 | 0.00 | 0.2555 | 0.4684 |
| H33 | 119 | 0.9565 | 0.8835 | 0.9679 | 0.00 | 0.00 | 0.2555 | 0.4684 |
| H34 | 239 | 0.7592 | -0.9433 | 0.9472 | 8.37 | 11.72 | 0.3503 | 0.9433 |
| H31 | 119 | 0.7064 | -0.7827 | 0.8574 | 10.92 | 12.61 | 0.4628 | 0.9098 |

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
