#!/usr/bin/env bash
# Minimal end-to-end test on one GDC case and one Valentine case with one
# traditional baseline, magneto_qwen and the primary jevnexus_shared variant.
# Requires both model servers.
# Verifies loading, representation, retrieval, Qwen call, decision call,
# ranking, prediction/runtime writing and evaluation. Exits non-zero on failure.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PYTHON" -m dema.experiments.smoke --config "$EXPERIMENT_CONFIG" "$@"
