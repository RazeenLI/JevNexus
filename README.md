# JevNexus — Decision-Centric Schema Matching

JevNexus is a zero-shot schema-matching research codebase. It combines Magneto
candidate retrieval, Open-Jev binary decisions, COMA++ structured evidence,
and optional Jina listwise reranking. The repository also provides a common
runner and evaluator for classical and neural baselines.

## Repository layout

```text
configs/                 Experiment, model, path, and scalability settings
data/raw/                Downloaded benchmark archives and extracted data
data/processed/          Normalized experiment inputs
data/manifests/          Case manifests and integrity metadata
models/checkpoints/      Locally downloaded model weights
src/dema/data/           Data download, preparation, and loading
src/dema/baselines/      COMA, COMA++, Magneto, ISResMat, Unicorn, and others
src/dema/model/          JevNexus retrieval, scoring, fusion, and ranking
src/dema/metrics/        Metrics and result aggregation
src/dema/experiments/    Runner, preflight, smoke, sensitivity, and scalability
src/dema/serving/        Transformers fallback server for Qwen
vendor/magneto/          Pinned upstream Magneto source and documented patches
cache/                   Disposable candidate caches
saves/                   Predictions, runtime records, status, and manifests
logs/                    Human-readable case, lane, and server logs
metrics/                 Generated CSV result tables
scripts/                 Command-line workflows
tests/                   Fast CPU tests and optional slow model tests
```

Generated data, model weights, caches, saved predictions, logs, and metrics are
kept out of Git. `cache/` may be deleted and rebuilt at any time. `saves/`
contains the state required to resume experiments, while `metrics/` can be
rebuilt from saved results.

## Installation

Python 3.10 or newer is required. Clone the repository and install it in an
isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,plot]"
```

GPU experiments require a compatible PyTorch/CUDA installation. vLLM is
optional but strongly recommended for full experiments; without it, Qwen uses
the lower-throughput Transformers fallback server. The helper below creates a
separate local vLLM environment:

```bash
bash scripts/setup_vllm.sh
```

The legacy shell workflows source `scripts/_env.sh`. They default to
`~/miniconda3/envs/airdb/bin/python`; set `PYTHON` to use another interpreter:

```bash
PYTHON="$PWD/.venv/bin/python" bash scripts/smoke_test.sh
```

## Running experiments

The runner downloads and prepares missing benchmark data. Services required by
`magneto_qwen` and JevNexus variants are started on demand, checked with a real
request, and stopped after the method completes.

Start with the development configuration or a single case:

```bash
python -m dema run --config configs/experiment_dev.yaml
python -m dema run --methods coma --datasets OpenData --limit 1
```

Run the complete 561-case matrix only after the development run succeeds:

```bash
python -m dema run --config configs/experiment.yaml
```

Use `--gpus 1` to place services and in-process models on GPU 1. With
`--gpus 1 0`, services run on GPU 1 and in-process retrieval or baseline models
run on GPU 0. `--verify-data` enables the slower full integrity check.

The compatibility shell entry point remains available:

```bash
EXPERIMENT_CONFIG=configs/experiment_dev.yaml PYTHON=python bash scripts/run_all.sh
```

## Methods

`magneto_qwen` runs the pinned upstream Magneto pipeline in `vendor/magneto`.
Its only functional patch replaces `litellm` with a local OpenAI-compatible
Qwen endpoint and records request statistics. This is a controlled Magneto
variant, not a reproduction of the original GPT-4o-mini numbers.

The primary experiment uses the following method matrix:

| Method | Candidate retriever | Scoring and reranking | Role |
|---|---|---|---|
| `magneto_qwen` | Upstream Magneto | Qwen listwise Top-20 | Baseline |
| `jevnexus` | Magneto candidates | Jev + COMA++ fixed fusion + Jina Top-3 | Main method |
| `jevnexus_no_rerank` | Magneto candidates | Jev + COMA++ fixed fusion | w/o refinement |
| `jevnexus_no_struct` | Magneto candidates | Jev + Jina Top-3 | w/o complementary evidence |
| `jevnexus_decision` | Magneto candidates | Jev only | w/o both components |
| `jevnexus_shared` | Magneto candidates | Jev with shared Top-20 context | Context ablation |

Canonical JevNexus reuses Magneto cleaning, serialization, mixed sampling,
exact-name matching, thresholding, Top-20 candidates, candidate ordering, and
dataset-specific encoding. Open-Jev and COMA++ scores are fused as:

```text
final_score = 0.4 × jev_score + 0.6 × coma_plus_score
```

Jina then reorders the fused Top-3. The Jina model is licensed under
CC BY-NC 4.0; review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
commercial use.

The former `dema*` method IDs remain registered as compatibility aliases for
historical experiment outputs. The Python import namespace also remains
`dema` so existing scripts and saved manifests continue to work.

## Development-set results

The following case-macro results were produced from the 29 valid cases selected
by `configs/experiment_dev.yaml`. They are development results, not the final
561-case benchmark.

| Method | MRR | Recall@GT | Recall@20 | NDCG@10 | MAP | Mean seconds/case |
|---|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 0.9501 | 0.8259 | 0.9683 | 0.9500 | 0.9431 | 133.72 |
| **JevNexus** | **0.9611** | **0.8020** | **0.9683** | **0.9593** | **0.9540** | **17.62** |
| w/o refinement | 0.9343 | 0.8050 | 0.9683 | 0.9401 | 0.9287 | 16.89 |
| w/o complementary evidence | 0.9375 | 0.6990 | 0.9683 | 0.9409 | 0.9298 | 16.85 |
| COMA++ | 0.9162 | 0.7382 | 0.9000 | 0.9089 | 0.9037 | 0.50 |
| w/o both components | 0.8845 | 0.6778 | 0.9683 | 0.9023 | 0.8781 | 16.59 |

### Short reproduction path

```bash
python -m dema run --config configs/experiment_dev.yaml
bash scripts/evaluate.sh
```

The evaluator regenerates `metrics/per_case.csv`, `metrics/per_dataset.csv`,
`metrics/overall.csv`, and `metrics/completeness.csv` from `saves/`. Exact
results require the pinned model revisions in `configs/models.yaml`; GPU,
driver, and library differences may affect runtime. See
[the model audit](docs/MODEL_AUDIT.md) for the measured setup and limitations.

## Saved output and resume semantics

```text
saves/predictions/<method>/<dataset>/<case_id>.json
saves/runtime/<method>/<dataset>/<case_id>.json
saves/status/<method>/<dataset>/<case_id>.json
saves/debug/<run_id>/<method>/<dataset>/<case_id>.jsonl
saves/manifests/<run_id>.json
metrics/{per_case,per_dataset,overall,completeness}.csv
```

A case is skipped by `--resume` only when its status is successful, its
prediction hash matches, runtime data exists, and every source column has a
complete target ranking without duplicates. Failed or partial cases are rerun.

## Configuration

- `configs/paths.yaml` defines repository-relative data and artifact paths.
- `configs/models.yaml` defines model revisions, prompts, servers, and matcher
  parameters.
- `configs/experiment.yaml` defines the complete 561-case experiment.
- `configs/experiment_dev.yaml` selects at most five cases per dataset.
- `configs/scalability.yaml` defines the target-width scalability experiment.

Do not use the complete configuration for routine code validation. It contains
more than 13,522 source-column inference units.

## Tests

```bash
python -m pytest
DEMA_SLOW_TESTS=1 python -m pytest
```

The slow suite downloads and loads the ISResMat and Unicorn models.

## Audits, citation, and licenses

- [Baseline implementation audit](docs/BASELINE_AUDIT.md)
- [Model and performance audit](docs/MODEL_AUDIT.md)
- [Citation metadata](CITATION.cff)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Original JevNexus code and documentation are released under the [MIT License](LICENSE).
Vendored software, downloaded models, and benchmark datasets retain their own
licenses as documented in the third-party notices.
