#!/usr/bin/env bash
# Runtime vs. target-schema width (configs/scalability.yaml); raw timings per repetition.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PYTHON" -m dema.scalability --config configs/scalability.yaml "$@"
