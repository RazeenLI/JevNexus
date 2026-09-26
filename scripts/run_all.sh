#!/usr/bin/env bash
# One-command experiment entry point. Required model services are started and
# stopped automatically; existing healthy endpoints are reused.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec "$PYTHON" -m dema run --config "$EXPERIMENT_CONFIG" "$@"
