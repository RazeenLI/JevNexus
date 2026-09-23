#!/usr/bin/env bash
# Full experiment. Assumes the Qwen and decision-model servers are already
# running (scripts/serve_qwen.sh, scripts/serve_decision.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

bash scripts/prepare_data.sh
"$PYTHON" -m dema.preflight
bash scripts/smoke_test.sh
bash scripts/run_baselines.sh
bash scripts/run_magneto.sh
bash scripts/run_dema.sh
bash scripts/run_scalability.sh
bash scripts/evaluate.sh
