#!/usr/bin/env bash
# Start the Qwen3.5-9B OpenAI-compatible server (only the server; no experiments).
#
# Same model setup as CoRE (configs/qwen3.5_9B.yaml there): Qwen/Qwen3.5-9B,
# bf16, non-thinking chat template, greedy decoding. Backend selection
# (serving.qwen.backend in configs/models.yaml):
#   vllm : `vllm serve` (used when `import vllm` works and backend is auto)
#   hf   : built-in transformers server (dema.serving.openai_server), CoRE's HF path
# Environment overrides: QWEN_PYTHON, CUDA_VISIBLE_DEVICES, QWEN_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
VLLM_ENV="${VLLM_ENV:-$HOME/.venvs/vllm}"
if [[ -x "$VLLM_ENV/bin/python" ]]; then
    default_qwen_python="$VLLM_ENV/bin/python"
else
    default_qwen_python="$PYTHON"
fi
QWEN_PYTHON="${QWEN_PYTHON:-$default_qwen_python}"

model="$(cfg qwen.model)"
served="$(cfg qwen.served_model_name)"
host="$(cfg serving.qwen.host)"
port="$(cfg serving.qwen.port)"
dtype="$(cfg serving.qwen.dtype)"
max_len="$(cfg serving.qwen.max_model_len)"
gpu_util="$(cfg serving.qwen.gpu_memory_utilization)"
device_map="$(cfg serving.qwen.device_map)"
backend="${QWEN_BACKEND:-$(cfg serving.qwen.backend)}"

if [[ "$backend" == "auto" ]]; then
    if "$QWEN_PYTHON" -c "import vllm" >/dev/null 2>&1; then backend=vllm; else backend=hf; fi
fi
echo "[serve_qwen] model=$model backend=$backend endpoint=http://$host:$port/v1"
if [[ "$backend" == "vllm" ]]; then
    # FlashInfer's sampling JIT is not usable when the wheel contains CUDA 13.0
    # runtime headers alongside a CUDA 13.4 nvcc. Qwen runs greedy decoding in
    # this project, so use vLLM's native sampler and avoid that unsafe toolchain
    # mix. Callers may explicitly opt back in after installing a matched CUDA
    # compiler/runtime pair.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    qwen_bin_dir="$(dirname "$QWEN_PYTHON")"
    export PATH="$qwen_bin_dir:$PATH"
    vllm_cli="$qwen_bin_dir/vllm"
    if [[ -x "$vllm_cli" ]]; then
        command=("$vllm_cli" serve "$model")
    else
        command=("$QWEN_PYTHON" -m vllm.entrypoints.cli.main serve "$model")
    fi
    exec "${command[@]}" \
        --served-model-name "$served" --host "$host" --port "$port" \
        --dtype "$dtype" --max-model-len "$max_len" --gpu-memory-utilization "$gpu_util" \
        --generation-config vllm
else
    exec "$QWEN_PYTHON" -m dema.serving.openai_server \
        --model "$model" --served-model-name "$served" --host "$host" --port "$port" \
        --dtype "$dtype" --device-map "$device_map" --max-model-len "$max_len"
fi
