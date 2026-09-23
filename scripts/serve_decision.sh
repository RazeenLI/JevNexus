#!/usr/bin/env bash
# Start the decision model server (System One HTTP API, POST /v1/systemone).
#
# Default model: Open-Jev-9B (ZefanCai/Open-Jev-9B) — LoRA adapter + scalar
# decision head over the pinned Qwen/Qwen3.5-9B revision — served by the
# Open-Jev reference server (`python -m jev.server`). Only the model server is
# started; no experiments run here. DeMa talks to it purely over HTTP.
#
# Requirements for DECISION_PYTHON (default: airdb python):
#   * the `jev` package (Open-Jev, https://github.com/Zefan-Cai/Open-Jev) and `peft`
#   * the checkpoint package (downloaded below with huggingface_hub if missing)
# Environment overrides: DECISION_PYTHON, DECISION_CHECKPOINT, CUDA_VISIBLE_DEVICES.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DECISION_PYTHON="${DECISION_PYTHON:-$PYTHON}"

host="$(cfg serving.decision.host)"
port="$(cfg serving.decision.port)"
repo="$(cfg serving.decision.hf_repo)"
revision="$(cfg serving.decision.hf_revision)"
local_dir="$(cfg serving.decision.local_dir)"
subdir="$(cfg serving.decision.checkpoint_subdir)"
max_len="$(cfg serving.decision.max_length)"
device="$(cfg serving.decision.device)"
checkpoint="${DECISION_CHECKPOINT:-$local_dir/$subdir}"

if ! "$DECISION_PYTHON" -c "import jev.server, peft" >/dev/null 2>&1; then
    cat >&2 <<MSG
[serve_decision] the Open-Jev server package is not importable by $DECISION_PYTHON.
Install it (adds new packages only; check with --dry-run first), e.g.:
    $DECISION_PYTHON -m pip install --dry-run peft==0.19.1 git+https://github.com/Zefan-Cai/Open-Jev.git
or point DECISION_PYTHON to an interpreter that already has it.
MSG
    exit 1
fi
if [[ ! -d "$checkpoint" ]]; then
    echo "[serve_decision] downloading $repo -> $local_dir"
    "$DECISION_PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', revision='$revision', local_dir='$local_dir')"
fi
echo "[serve_decision] checkpoint=$checkpoint endpoint=http://$host:$port/v1/systemone"
exec "$DECISION_PYTHON" -m jev.server --checkpoint "$checkpoint" --device "$device" \
    --max-length "$max_len" --batch-size 1 --no-prefix-cache --host "$host" --port "$port"
