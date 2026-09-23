#!/usr/bin/env bash
# Magneto-style baseline with Qwen3.5-9B reranking (requires scripts/serve_qwen.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PYTHON" -m dema.runner --method magneto_qwen --dataset $(experiment_datasets) --resume "$@"
