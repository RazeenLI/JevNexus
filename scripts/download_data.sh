#!/usr/bin/env bash
# Download the raw benchmark (GDC-SM + Valentine) into data/raw. Idempotent.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
"$PYTHON" -m dema.data.download "$@"
