#!/usr/bin/env bash
# Run the publication-oriented full benchmark on one GPU, in priority order.
#
# A failed method/group is recorded and does not stop later work.  The Python
# runner also continues after ordinary case exceptions; Distribution adds a
# separate process boundary around every case so timeout/OOM-like exits cannot
# terminate the remaining cases.  Existing complete units are resumed/skipped.
#
# Usage: bash scripts/run_full_gpu.sh [gpu]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

GPU="${1:-1}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/experiment.yaml}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
DISTRIBUTION_CASE_TIMEOUT="${DISTRIBUTION_CASE_TIMEOUT:-7200}"
DISTRIBUTION_CASE_MEMORY_GB="${DISTRIBUTION_CASE_MEMORY_GB:-0}"

RUN_ROOT="logs/full_gpu${GPU}"
TASK_ROOT="$RUN_ROOT/tasks"
mkdir -p "$TASK_ROOT"
exec 9>"$RUN_ROOT/.running"
if ! flock -n 9; then
    echo "full experiment is already running for GPU $GPU (lock: $RUN_ROOT/.running)" >&2
    exit 2
fi

exec >>"$RUN_ROOT/run.log" 2>&1
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
        log "FAIL  $name (exit $rc, $((SECONDS - started))s); continuing; see $task_log"
    fi
    return 0
}

run_methods() {
    "$PYTHON" -m dema run \
        --config "$EXPERIMENT_CONFIG" \
        --methods "$@" \
        --gpus "$GPU" \
        --server-timeout "$SERVER_TIMEOUT" \
        --distribution-case-timeout "$DISTRIBUTION_CASE_TIMEOUT" \
        --distribution-case-memory-gb "$DISTRIBUTION_CASE_MEMORY_GB" \
        --no-evaluate
}

log "===== full experiment started: config=$EXPERIMENT_CONFIG ====="

# Most important claims first.  Separate invocations ensure a fatal process
# exit in one task cannot prevent the next task from starting.
run_task 01_jevnexus run_methods jevnexus jevnexus_always
run_task 02_magneto_qwen run_methods magneto_qwen
run_task 03_coma_plus run_methods coma_plus
run_task 04_unicorn run_methods unicorn
run_task 05_isresmat run_methods isresmat

# Core component ablations share one Open-Jev service lifecycle.
run_task 06_jevnexus_ablations run_methods \
    jevnexus_no_rerank jevnexus_no_struct jevnexus_decision

# Secondary traditional baselines. Distribution is last because it is the
# least reliable/most resource-sensitive and is isolated one case at a time.
run_task 07_coma run_methods coma
run_task 08_similarity_flooding run_methods similarity_flooding
run_task 09_distribution run_methods distribution

# Always evaluate every intended method. Missing/failed units remain visible in
# completeness.csv; evaluator metrics are computed only over completed units.
run_task 10_evaluate \
    "$PYTHON" -m dema.metrics.evaluator \
    --config "$EXPERIMENT_CONFIG" \
    --methods \
    jevnexus jevnexus_always magneto_qwen coma_plus unicorn isresmat \
    jevnexus_no_rerank jevnexus_no_struct jevnexus_decision \
    coma similarity_flooding distribution

log "===== full experiment finished: $TASK_NO tasks, $N_FAILED failed; see $FAILURES ====="
exit "$N_FAILED"
