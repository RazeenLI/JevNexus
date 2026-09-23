#!/usr/bin/env bash
# Traditional baselines on all configured datasets (resumable).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
methods="${METHODS:-coma coma_plus distribution similarity_flooding isresmat unicorn}"
for method in $methods; do
    "$PYTHON" -m dema.runner --method "$method" --dataset $(experiment_datasets) --resume "$@"
done
