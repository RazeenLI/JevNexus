#!/usr/bin/env bash
# DeMa with the decision model (requires scripts/serve_decision.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PYTHON" -m dema.runner --method dema --dataset $(experiment_datasets) --resume "$@"
