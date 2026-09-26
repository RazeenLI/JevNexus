#!/usr/bin/env bash
# Evaluate all completed predictions (no model inference) ->
# metrics/{per_case,per_dataset,overall,completeness}.csv
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PYTHON" -m dema.metrics.evaluator --config "$EXPERIMENT_CONFIG" "$@"
