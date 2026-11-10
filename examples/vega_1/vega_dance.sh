#!/bin/bash
# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

# Replay the Vega-1 dance recording (data/vega-1_dance.csv) with the
# replay_trajectory example of the chosen language, through the launcher, so
# the native versions are built (and given the watchdog) automatically.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECORDING="$ROOT/examples/vega_1/data/vega-1_dance.csv"

LANGUAGE=python
VISUALIZE=""
# Recorded speed. A whole-body dance must not start faster than it was
# recorded unless the operator asks for it with -s.
SPEED=1.0
PASSTHROUGH=()

show_help() {
    cat <<EOF
Usage: ${0##*/} [python|cpp|rust] [OPTIONS] [--<replay option> [value]]...

Replay the dance with the replay_trajectory example of one language
(default: python). C++ and Rust are built on demand, in release mode.

Options:
  -v           Enable visualization (Python plots; C++ prints a summary)
  -s SPEED     Speed factor, clipped to 0.2 .. 3.0 (default: 1.0)
  -h           Show this help message
  --<option>   Any other --option is passed to replay_trajectory, e.g.
               --clamp-to-model-limits (Python) or --simulated --profile vega_1

Python and C++ prepare the robot (fold arms, head home, crouch, close hands),
smooth the recording and honour -v/-s. Rust streams the recording exactly as
recorded from the current pose: no preparation, smoothing, -v or -s.

Examples:
  ${0##*/}                      # Python, recorded speed
  ${0##*/} cpp -s 1.5           # C++, 1.5x
  ${0##*/} rust --no-confirm    # Rust, no prompt
  ${0##*/} python -v --clamp-to-model-limits
EOF
}

validate_speed() {
    # A number, clipped to the range the replay examples accept.
    awk -v s="$1" 'BEGIN {
        if (s !~ /^[0-9]*\.?[0-9]+$/) { print "Error: Speed must be a number" > "/dev/stderr"; exit 1 }
        if (s + 0 < 0.2) { print "Warning: Speed clipped to minimum value 0.2" > "/dev/stderr"; s = 0.2 }
        if (s + 0 > 3.0) { print "Warning: Speed clipped to maximum value 3.0" > "/dev/stderr"; s = 3.0 }
        print s + 0
    }'
}

case "${1:-}" in
    python|cpp|c++|rust)
        LANGUAGE=$1
        shift
        ;;
esac

while [[ $# -gt 0 ]]; do
    case $1 in
        -v)
            VISUALIZE="--visualize"
            shift
            ;;
        -s)
            if [[ -n ${2:-} && $2 != -* ]]; then
                SPEED=$(validate_speed "$2")
                shift 2
            else
                echo "Error: -s requires a speed value" >&2
                exit 1
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --*)
            # Long options belong to replay_trajectory; pass them through,
            # with a following value when it is not itself an option.
            PASSTHROUGH+=("$1")
            shift
            if [[ $# -gt 0 && $1 != -* ]]; then
                PASSTHROUGH+=("$1")
                shift
            fi
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

if [[ ! -f $RECORDING ]]; then
    echo "Error: recording not found: $RECORDING" >&2
    exit 1
fi

# The option names are the ones each replay_trajectory defines; the example
# rejects anything else before the robot moves.
case $LANGUAGE in
    python)
        exec "$ROOT/run" python replay_trajectory "$RECORDING" \
            --control-hz 500 --smooth 0.1 --vel-smooth 0.5 --speed-factor "$SPEED" \
            $VISUALIZE "${PASSTHROUGH[@]}"
        ;;
    cpp|c++)
        exec "$ROOT/run" --release cpp replay_trajectory "$RECORDING" \
            --control-hz 500 --smooth 0.1 --vel-smooth 0.5 --speed-factor "$SPEED" \
            $VISUALIZE "${PASSTHROUGH[@]}"
        ;;
    rust)
        if [[ -n $VISUALIZE || $SPEED != 1.0 ]]; then
            echo "Error: the Rust replay streams the recording as recorded; -v and -s are Python/C++ options" >&2
            exit 1
        fi
        exec "$ROOT/run" --release rust replay_trajectory "$RECORDING" \
            --control-hz 500 "${PASSTHROUGH[@]}"
        ;;
esac
