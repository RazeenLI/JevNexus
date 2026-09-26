#!/usr/bin/env bash
# Traditional baselines on all configured datasets (resumable).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
methods="${METHODS:-coma coma_plus distribution similarity_flooding isresmat unicorn}"
for method in $methods; do
    if [[ "$method" == "distribution" ]]; then
        # Distribution's CBC MILPs run in supervised per-case processes so one
        # OOM/timeout cannot discard every remaining benchmark case.
        "$PYTHON" -m dema run --config "$EXPERIMENT_CONFIG" --methods distribution \
            --datasets $(experiment_datasets) --no-evaluate "$@"
    else
        "$PYTHON" -m dema.experiments.runner --config "$EXPERIMENT_CONFIG" --method "$method" \
            --dataset $(experiment_datasets) --resume "$@"
    fi
done
