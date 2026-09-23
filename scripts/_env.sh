# Shared environment for all scripts (sourced, not executed).
# Uses the existing conda env "airdb" by default; override with PYTHON=...
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/airdb/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Print one value of configs/models.yaml, e.g. cfg serving.qwen.port
cfg() {
    "$PYTHON" - "$1" <<'PY'
import sys, yaml
value = yaml.safe_load(open("configs/models.yaml"))
for key in sys.argv[1].split("."):
    value = value[key]
print(value)
PY
}

# Datasets from experiment.yaml (space separated).
experiment_datasets() {
    "$PYTHON" -c 'import yaml; print(" ".join(yaml.safe_load(open("configs/experiment.yaml"))["datasets"]))'
}
