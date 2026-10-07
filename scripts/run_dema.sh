#!/usr/bin/env bash
# Deprecated compatibility entry point. Prefer scripts/run_jevnexus.sh.
# Historical DEMA_METHOD values remain registered aliases.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DEMA_METHOD="${DEMA_METHOD:-dema}"
"$PYTHON" -m dema.experiments.runner --config "$EXPERIMENT_CONFIG" --method "$DEMA_METHOD" \
    --dataset $(experiment_datasets) --resume "$@"
