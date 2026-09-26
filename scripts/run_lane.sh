#!/usr/bin/env bash
# Run one experiment "lane" pinned to one GPU; two lanes together cover everything.
#
#   bash scripts/run_lane.sh qwen 0        # GPU0: baselines + Qwen server + magneto_qwen
#   bash scripts/run_lane.sh decision 1    # GPU1: decision server + dema, then ISResMat
#
# Every task runs to completion even if an earlier one failed. Failed tasks are
# appended to logs/lanes/<lane>/failures.tsv (case-level state is in saves/status
# and aggregate completeness is in metrics/completeness.csv). Re-running resumes:
# completed cases are skipped, failed/missing ones are retried.
#
# Logs: logs/lanes/all.log       both lanes in one file: task status + error excerpts
#       logs/lanes/<lane>/lane.log (this lane), tasks/*.log (full output per task),
#       server.log (model server).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

LANE="${1:?usage: run_lane.sh <qwen|decision> <gpu>}"
GPU="${2:?usage: run_lane.sh <qwen|decision> <gpu>}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"   # seconds to wait for a model server (first start downloads weights)
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/experiment.yaml}"
SCALABILITY_CONFIG="${SCALABILITY_CONFIG:-configs/scalability.yaml}"
export EXPERIMENT_CONFIG SCALABILITY_CONFIG
export CUDA_VISIBLE_DEVICES="$GPU"

LANES_ROOT="logs/lanes"
LANE_DIR="$LANES_ROOT/$LANE"
mkdir -p "$LANE_DIR/tasks"
exec 9>"$LANES_ROOT/.$LANE.running"
flock -n 9 || { echo "lane $LANE is already running" >&2; exit 1; }
exec >>"$LANE_DIR/lane.log" 2>&1

FAILURES="$LANE_DIR/failures.tsv"
[[ -f "$FAILURES" ]] || printf "time\ttask\texit_code\tlog\n" > "$FAILURES"
SERVER_PID=""
TASK_NO=0
N_FAILED=0

ALL_LOG="$LANES_ROOT/all.log"
log() {
    local line="[$(date '+%F %T')] [$LANE/GPU$GPU] $*"
    echo "$line"
    echo "$line" >> "$ALL_LOG"
}

# run_task <name> <command...> : never aborts the lane
run_task() {
    local name="$1"; shift
    TASK_NO=$((TASK_NO + 1))
    local tlog; tlog="$LANE_DIR/tasks/$(printf '%02d' "$TASK_NO")_${name}.log"
    log "START $name"
    local start=$SECONDS
    "$@" >>"$tlog" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        log "OK    $name ($((SECONDS - start))s)"
    else
        N_FAILED=$((N_FAILED + 1))
        printf "%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$name" "$rc" "$tlog" >> "$FAILURES"
        log "FAIL  $name (exit $rc, $((SECONDS - start))s) -> $tlog"
        { grep -E "Error|error|FAILED|Traceback" "$tlog" | tail -n 5 | sed "s/^/    [$LANE] /"; } >> "$ALL_LOG"
    fi
    return 0
}

skip_task() {
    N_FAILED=$((N_FAILED + 1))
    printf "%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$1" "skipped" "$2" >> "$FAILURES"
    log "SKIP  $1 ($2)"
}

# start_server <serve script> <preflight method> : returns 0 when the endpoint answers
start_server() {
    log "starting server: $1"
    setsid bash "$1" >>"$LANE_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    local waited=0
    until "$PYTHON" -m dema.experiments.preflight --config "$EXPERIMENT_CONFIG" --methods "$2" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "server process exited (see $LANE_DIR/server.log)"; SERVER_PID=""; return 1
        fi
        if (( waited >= SERVER_TIMEOUT )); then
            log "server not ready after ${SERVER_TIMEOUT}s"; stop_server; return 1
        fi
        sleep 15; waited=$((waited + 15))
    done
    log "server ready (pid $SERVER_PID, ${waited}s)"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping server (pid $SERVER_PID)"
        kill -TERM -- "-$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}
trap 'stop_server' EXIT
trap 'log "interrupted"; exit 130' INT TERM

# Shared steps, serialized across lanes with a lock (processed data must never be
# rewritten while the other lane reads it; candidates are computed exactly once).
prepare_shared() {
    flock "$LANES_ROOT/.data.lock" bash -c '
        set -e
        if [[ -f data/manifests/all.jsonl ]]; then
            "$0" -m dema.data.prepare --verify-only
        else
            bash scripts/prepare_data.sh
        fi
        "$0" -m dema.experiments.precompute --config "$EXPERIMENT_CONFIG" --scalability "$SCALABILITY_CONFIG"
    ' "$PYTHON"
}

evaluate_shared() {
    flock "$LANES_ROOT/.eval.lock" bash -c '
        "$0" -m dema.metrics.evaluator --config "$EXPERIMENT_CONFIG"; rc=$?
        "$0" -m dema.experiments.scalability --summarize-only || rc=1
        exit $rc
    ' "$PYTHON"
}

# --fail-on-error: the runner still processes every case, but exits non-zero if any failed.
runner() {
    local datasets
    datasets=$("$PYTHON" -c 'import sys, yaml; print(" ".join(yaml.safe_load(open(sys.argv[1]))["datasets"]))' "$EXPERIMENT_CONFIG")
    if [[ "$1" == "distribution" ]]; then
        "$PYTHON" -m dema run --config "$EXPERIMENT_CONFIG" --methods distribution \
            --datasets $datasets --no-evaluate
    else
        "$PYTHON" -m dema.experiments.runner --method "$1" --dataset $datasets \
            --config "$EXPERIMENT_CONFIG" --resume --fail-on-error
    fi
}
scal() { "$PYTHON" -m dema.experiments.scalability --config "$SCALABILITY_CONFIG" --methods "$@"; }

log "===== lane $LANE started (CUDA_VISIBLE_DEVICES=$GPU) ====="
run_task prepare_and_candidates prepare_shared

case "$LANE" in
    qwen)
        for m in coma coma_plus distribution similarity_flooding unicorn; do
            run_task "baseline_$m" runner "$m"
        done
        run_task scalability_baselines scal coma_plus unicorn
        if start_server scripts/serve_qwen.sh magneto_qwen; then
            run_task magneto_qwen runner magneto_qwen
            run_task scalability_magneto_qwen scal magneto_qwen
            stop_server
        else
            skip_task magneto_qwen "Qwen server did not start"
            skip_task scalability_magneto_qwen "Qwen server did not start"
        fi
        ;;
    decision)
        if start_server scripts/serve_decision.sh dema; then
            run_task dema runner dema
            run_task dema_no_rerank runner dema_no_rerank
            run_task dema_no_struct runner dema_no_struct
            run_task dema_decision runner dema_decision
            run_task dema_shared runner dema_shared
            run_task scalability_dema_shared scal dema_shared
            stop_server
        else
            skip_task dema "decision server did not start"
            skip_task dema_no_rerank "decision server did not start"
            skip_task dema_no_struct "decision server did not start"
            skip_task dema_decision "decision server did not start"
            skip_task dema_shared "decision server did not start"
            skip_task scalability_dema_shared "decision server did not start"
        fi
        run_task baseline_isresmat runner isresmat
        run_task scalability_isresmat scal isresmat
        ;;
    *)
        log "unknown lane '$LANE' (use qwen or decision)"; exit 2 ;;
esac

run_task evaluate evaluate_shared
log "===== lane $LANE finished: $TASK_NO task(s), $N_FAILED failed/skipped (see $FAILURES) ====="
