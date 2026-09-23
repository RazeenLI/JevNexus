#!/usr/bin/env bash
# Validate raw benchmark -> convert to data/processed -> manifest -> integrity checks.
# Idempotent: re-running rewrites identical processed files and re-verifies them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

raw_gdc="data/raw/gdc/data/ground-truth"
raw_valentine="data/raw/valentine/Valentine-datasets"
if [[ ! -d "$raw_gdc" || ! -d "$raw_valentine" ]]; then
    echo "[prepare] raw benchmark missing under data/raw; run: bash scripts/download_data.sh" >&2
    exit 1
fi
"$PYTHON" -m dema.data.prepare "$@"
