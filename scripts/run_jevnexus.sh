#!/usr/bin/env bash
# JevNexus with the decision model (requires scripts/serve_decision.sh).
# Set JEVNEXUS_METHOD to a registered JevNexus variant for an ablation.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
JEVNEXUS_METHOD="${JEVNEXUS_METHOD:-jevnexus}"
"$PYTHON" -m dema.experiments.runner --config "$EXPERIMENT_CONFIG" --method "$JEVNEXUS_METHOD" \
  --dataset "${DATASET:-GDC}" "${@:1}"
