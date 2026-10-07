#!/usr/bin/env bash
# Resume-safe run for the experiments still needed by the arXiv JevNexus draft.
# Usage: nohup bash scripts/run_arxiv_experiments.sh 1 > logs/arxiv.nohup.log 2>&1 &
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

GPU="${1:-1}"
FULL_CONFIG="${FULL_CONFIG:-configs/experiment.yaml}"
DEV_CONFIG="${DEV_CONFIG:-configs/experiment_dev.yaml}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
RUN_ROOT="logs/arxiv_gpu${GPU}"
TASK_ROOT="$RUN_ROOT/tasks"
mkdir -p "$TASK_ROOT"
exec 9>"$RUN_ROOT/.running"
if ! flock -n 9; then
    echo "arXiv experiment run is already active for GPU $GPU" >&2
    exit 2
fi

FAILURES="$RUN_ROOT/failures.tsv"
[[ -f "$FAILURES" ]] || printf "time\ttask\texit_code\tlog\n" > "$FAILURES"
N_FAILED=0
TASK_NO=0

log() { printf '[%s] [GPU%s] %s\n' "$(date '+%F %T')" "$GPU" "$*"; }

run_task() {
    local name="$1"
    shift
    TASK_NO=$((TASK_NO + 1))
    local task_log="$TASK_ROOT/$(printf '%02d' "$TASK_NO")_${name}.log"
    local started=$SECONDS
    log "START $name"
    "$@" >"$task_log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        log "OK    $name ($((SECONDS - started))s)"
    else
        N_FAILED=$((N_FAILED + 1))
        printf "%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$name" "$rc" "$task_log" >> "$FAILURES"
        log "FAIL  $name (exit $rc); continuing; see $task_log"
    fi
    return 0
}

run_jevnexus() {
    "$PYTHON" -m dema run --config "$1" --methods "${@:2}" \
        --gpus "$GPU" --server-timeout "$SERVER_TIMEOUT" --no-evaluate
}

log "===== arXiv experiment run started ====="

# Full canonical result plus the Always-Jina trace required for exact offline
# alpha/tau replay. Both commands resume complete cases and retry stale/failed ones.
run_task 01_full_gate_and_trace run_jevnexus "$FULL_CONFIG" jevnexus jevnexus_always

# Changing m changes the actual listwise request, so m=2 and m=5 are run online
# only on the deterministic five-case-per-dataset development split. m=3 reuses
# the canonical full run above.
run_task 02_m_sensitivity run_jevnexus "$DEV_CONFIG" jevnexus_gate_m2 jevnexus_gate_m5

# No model inference: alpha is evaluated on fusion alone; tau replays the saved
# m=3 Always-Jina order. Also creates the m comparison table.
run_task 03_offline_sensitivity \
    "$PYTHON" -m dema.experiments.sensitivity \
    --config "$DEV_CONFIG" \
    --source-method jevnexus_always \
    --m-methods jevnexus_gate_m2 jevnexus jevnexus_gate_m5

# Resume skips the 560 completed Magneto cases and retries the one recorded
# OpenData timeout. It remains a separate task because it uses the Qwen service.
run_task 04_retry_magneto \
    "$PYTHON" -m dema run --config "$FULL_CONFIG" --methods magneto_qwen \
    --gpus "$GPU" --server-timeout "$SERVER_TIMEOUT" --no-evaluate

# Refresh the publication tables after the new canonical JevNexus outputs exist.
run_task 05_full_evaluation \
    "$PYTHON" -m dema.metrics.evaluator \
    --config "$FULL_CONFIG" \
    --methods jevnexus jevnexus_always magneto_qwen coma_plus unicorn isresmat \
    jevnexus_no_rerank jevnexus_no_struct jevnexus_decision coma similarity_flooding distribution

# Candidate localization, gate routing, and paired JevNexus-vs-Magneto bootstrap.
# Reports both common-case estimates and the conservative failure-as-zero policy.
run_task 06_publication_analysis \
    "$PYTHON" -m dema.experiments.publication_analysis \
    --config "$FULL_CONFIG" --method jevnexus --compare magneto_qwen

log "===== arXiv experiment run finished: $TASK_NO tasks, $N_FAILED failed ====="
log "failure ledger: $FAILURES"
exit "$N_FAILED"
