#!/usr/bin/env bash
# DeMa with the decision model (requires scripts/serve_decision.sh).
# Set DEMA_METHOD=dema_single, dema_shared, or dema_own_retrieval for an ablation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DEMA_METHOD="${DEMA_METHOD:-dema_fusion}"
"$PYTHON" -m dema.experiments.runner --config "$EXPERIMENT_CONFIG" --method "$DEMA_METHOD" \
    --dataset $(experiment_datasets) --resume "$@"
