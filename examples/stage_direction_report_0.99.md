# Stage Directionality Report

- log_path: `/home/dexmate/.cursor/debug-daf7f0.log`
- run_id_filters: `ALL`
- line_window_start: `20423`
- line_window_end: `25453`
- loop_window: size=240, index_from_end=1 (matched_total=1200)

## Stage Definitions

- `H30` (`ik_fk_projection_error`): intended Cartesian target vs FK of IK-solved joints.
- `H32` (`post_filter_fk_error`): intended Cartesian target vs FK after smoothing/filter stage.
- `H33` (`post_limits_fk_error`): intended Cartesian target vs FK after clip/jerk limits.
- `H34` (`joint_direction_transfer`): commanded joint delta vs next-tick achieved joint delta.
- `H31` (`achieved_tracking_error`): intended Cartesian target direction vs next-tick observed Cartesian movement.

## Overall (All Arms)

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 238 | 0.9686 | 0.9041 | 0.9792 | 0.00 | 0.00 | 0.2110 | 0.4273 |
| H32 | 238 | 0.9686 | 0.9041 | 0.9792 | 0.00 | 0.00 | 0.2110 | 0.4273 |
| H33 | 238 | 0.9686 | 0.9041 | 0.9792 | 0.00 | 0.00 | 0.2110 | 0.4273 |
| H34 | 478 | 0.7439 | -0.9452 | 0.9496 | 9.62 | 12.55 | 0.3560 | 0.9434 |
| H31 | 238 | 0.7348 | -0.8336 | 0.8883 | 9.24 | 11.76 | 0.4040 | 0.8577 |

## By Arm

### arm: `left`

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 119 | 0.9817 | 0.9421 | 0.9869 | 0.00 | 0.00 | 0.1620 | 0.3353 |
| H32 | 119 | 0.9817 | 0.9420 | 0.9868 | 0.00 | 0.00 | 0.1620 | 0.3356 |
| H33 | 119 | 0.9817 | 0.9420 | 0.9868 | 0.00 | 0.00 | 0.1620 | 0.3356 |
| H34 | 239 | 0.7272 | -0.9450 | 0.9529 | 10.46 | 13.81 | 0.3692 | 0.9607 |
| H31 | 119 | 0.7497 | -0.8728 | 0.9533 | 9.24 | 11.76 | 0.3532 | 0.8698 |

### arm: `right`

| Stage | n | cos_mean | cos_p05 | cos_p50 | cos<0 (%) | cos<0.3 (%) | perp_mean | perp_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H30 | 119 | 0.9556 | 0.8884 | 0.9664 | 0.00 | 0.00 | 0.2600 | 0.4590 |
| H32 | 119 | 0.9556 | 0.8885 | 0.9664 | 0.00 | 0.00 | 0.2600 | 0.4589 |
| H33 | 119 | 0.9556 | 0.8885 | 0.9664 | 0.00 | 0.00 | 0.2600 | 0.4589 |
| H34 | 239 | 0.7606 | -0.9248 | 0.9479 | 8.79 | 11.30 | 0.3429 | 0.9150 |
| H31 | 119 | 0.7200 | -0.8081 | 0.8623 | 9.24 | 11.76 | 0.4548 | 0.7610 |

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
