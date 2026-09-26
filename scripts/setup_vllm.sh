#!/usr/bin/env bash
# Create the isolated vLLM runtime used by scripts/serve_qwen.sh.
set -euo pipefail

VLLM_ENV="${VLLM_ENV:-$HOME/.venvs/vllm}"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniconda3/envs/airdb/bin/python}"

if [[ -e "$VLLM_ENV" ]]; then
    echo "[setup_vllm] $VLLM_ENV already exists; refusing to overwrite it" >&2
    exit 1
fi
"$BASE_PYTHON" -m venv --copies "$VLLM_ENV"
"$VLLM_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VLLM_ENV/bin/python" -m pip install "vllm==0.30.0"
"$VLLM_ENV/bin/python" -m pip check
echo "[setup_vllm] ready: $VLLM_ENV"
