# DeMa — Decision-based Schema Matching

Self-contained research codebase for zero-shot schema matching on the benchmark
collection used by Magneto (GDC + Valentine: ChEMBL, Magellan, OpenData, TPC-DI,
WikiData). It contains

* the traditional baselines compared in Magneto (COMA, COMA++, Distribution-based,
  Similarity Flooding, ISResMat, Unicorn) plus two trivial ones (Jaccard, Levenshtein);
* `magneto_qwen` — a Magneto-style retrieve-then-rerank baseline with **Qwen3.5-9B**;
* `dema` — the same retrieval with a **decision model** (default: Open-Jev-9B, a
  Jev-like System One model) giving one independent `P(match)` per candidate;
* one evaluator, runtime tracking, resumable runner, scalability runner and scripts.

Nothing is imported from Magneto or other research repositories at run time.
Model servers (vLLM / the built-in HF server for Qwen, the Open-Jev server for the
decision model) are external processes that DeMa talks to over HTTP only.

```
Benchmark -> Column profiles -> Semantic candidate search (top-k)
                                   /                        \
                        Qwen3.5-9B scoring          Decision-model scoring
                          (magneto_qwen)                   (dema)
                                   \                        /
                                     Complete ranking -> Shared evaluator
```

## Environment

All commands use the existing conda env **`airdb`**
(`~/miniconda3/envs/airdb/bin/python`, override with `PYTHON=...`). Packages
added to it for DeMa: `openpyxl` (GDC source extraction), `peft` and `open-jev`
(installed with `--no-deps`; only needed by `scripts/serve_decision.sh`).
Scripts set `PYTHONPATH=src`; tests run with `python -m pytest`.

## Quick start

```bash
bash scripts/download_data.sh      # raw benchmark -> data/raw (idempotent)
bash scripts/prepare_data.sh       # -> data/processed + data/manifests/all.jsonl + integrity checks
bash scripts/serve_qwen.sh         # terminal 1: Qwen3.5-9B, OpenAI API on :8000
bash scripts/serve_decision.sh     # terminal 2: decision model, System One API on :8009
bash scripts/run_all.sh            # preflight, smoke test, all methods, scalability, evaluation
```

Single runs:

```bash
python -m dema.runner --method dema --dataset GDC
python -m dema.runner --method coma --dataset OpenData --case-id <id> --overwrite
python -m dema.runner --method unicorn --dataset all --limit 1 --save-debug
python -m dema.evaluation.evaluator               # == scripts/evaluate.sh
```

Runner flags: `--case-id`, `--resume` / `--overwrite` (default from
`experiment.yaml`), `--config <experiment.yaml>`, `--save-debug`, `--limit N`
(cases per dataset), `--no-cache`, `--fail-on-error`.

## Repository layout

```
configs/       paths.yaml models.yaml experiment.yaml scalability.yaml  (all constants)
data/raw/      downloaded archives + extracted benchmark (never modified)
data/processed/<dataset>/<case_id>/{source,target,ground_truth}.csv, metadata.json
data/manifests/all.jsonl
src/dema/
  data/            download, prepare (+ integrity checks), loader, manifest, types
  representation/  profiler (type inference), sampler (frequency sampling), serializer
  models/          base, retriever (+cache), prompting, generative_reranker (Qwen),
                   decision_reranker (DeMa), two_stage, magneto_qwen, dema, ranking, registry
  baselines/       coma, coma_plus, distribution, similarity_flooding, isresmat, unicorn, simple
  evaluation/      metrics, runtime, evaluator, aggregate
  serving/         openai_server (HF fallback for serve_qwen.sh)
  runner.py  scalability.py  preflight.py  smoke.py
scripts/       download/prepare/serve/smoke/run_*/evaluate/run_all
tests/         unit + integration tests (fake model servers, no GPU needed)
outputs/       cache, predictions, runtime, status, logs, metrics, manifests, scalability, smoke
```

## Benchmark

| Dataset | Cases | Source | Notes |
|---|---:|---|---|
| GDC | 10 | GDC-SM, Zenodo 10.5281/zenodo.14963588 | target: 736-column GDC schema |
| ChEMBL | 180 | Valentine, Zenodo 10.5281/zenodo.5084605 | 4 relatedness types |
| Magellan | 7 | Valentine | |
| OpenData | 180 | Valentine | |
| TPC-DI | 180 | Valentine | |
| WikiData | 4 | Valentine (`Wikidata/Musicians`) | |

GDC-SM does not redistribute the study tables; `dema.data.download` rebuilds
them with the official GDC-SM procedure (`gdc_download.py` from the Zenodo
record, re-implemented with identical logic and its four documented fixes).

Processing copies tables byte-for-byte when they are valid UTF-8 (otherwise the
decoded text is re-written as UTF-8), converts ground truth to
`source_column,target_column`, and never renames columns, alters values, or adds/
drops rows or GT pairs. `prepare_data.sh` verifies this for every case (content,
header, row count, GT pairs, raw file hashes, pandas-loadable headers).

**Known raw defect.** In `wikidata_musicians_unionable` and
`wikidata_musicians_viewunion` the GT pair `givenName -> forename` refers to a
source column that is called `givenNameLabel`. The GT is kept unchanged; the
pair is listed in `paths.yaml: known_gt_defects` and counts as a miss for every
method (as in Magneto's evaluation). Any other invalid reference fails the
integrity check.

## Methods

### Shared representation and retrieval (`magneto_qwen`, `dema`)

* **Profile** per column: name, type (`integer|float|boolean|datetime|string|mixed`,
  deterministic rules), top-10 values by frequency (ties lexicographic; nulls removed).
* **Serialization** for embedding: `Column: <name>\nType: <type>\nValues: v1 | v2 | ...`.
* **Retrieval**: `sentence-transformers/all-mpnet-base-v2` (zero-shot, no fine-tuning),
  cosine similarity, complete target ordering per source column (ties by target
  position); top-k = 20 go to the reranker.
* **Cache** `outputs/cache/candidates/`: key = dataset, case_id, embedding model,
  representation signature, top_k, and source/target profile fingerprints. Both
  matchers reuse the same entries, so their candidate sets and retrieval scores are
  identical.
* **Candidate presentation**: candidates get ids `c0..c{k-1}` in *target-schema
  order* (not retrieval order); rerankers see only name/type/values — never retrieval
  scores or ranks, ground truth, or dataset hints.
* **Final ranking**: top-k sorted by reranker score (ties by retrieval rank), then
  the remaining targets in retrieval order. Every target appears exactly once.
  Stored `score` = reranker score for the top-k, `0.0` for the tail;
  `reranker_score` and `retrieval_score` are stored separately.

### magneto_qwen

One chat request per source column with all k candidates; Qwen must return
`{"c0": 0.91, ...}` with exactly the k ids and scores in [0,1]; temperature 0,
non-thinking chat template (as in CoRE). Invalid output (bad JSON, missing/extra/
duplicate ids, non-numeric, NaN, out of range), timeouts and connection errors are
retried up to `qwen.max_retries`; after that the case fails. There is **no fallback
to retrieval scores**. Differences to original Magneto: zero-shot retriever, DeMa
serialization, Qwen3.5-9B instead of GPT-4o-mini.

### dema

Same pipeline; the reranker is `DecisionReranker` over a pluggable
`DecisionBackend` (`predict(state, questions) -> dict`). The default `system_one`
backend speaks the Jev/Open-Jev System One protocol (`POST /v1/systemone`): the
state holds the source column and the k candidates, and there is one independent
`noul` (yes/no) question per candidate id — "Does candidate cX represent the same
underlying schema attribute as the source column?" — giving `p_i = P(T_i matches S)`.
Responses must contain exactly the k ids with probabilities in [0,1]; same retry/
failure policy as Qwen. The checkpoint (default `ZefanCai/Open-Jev-9B`, LoRA +
decision head on the pinned Qwen3.5-9B revision) is configured only in
`models.yaml`/`serve_decision.sh`.

### Traditional baselines — implementation notes and simplifications

| Method | Implementation |
|---|---|
| `coma` | valentine 1.x pure-Python COMA, schema matchers only. Magneto used Java COMA 3.0 via older valentine; `delta=1.0`, `threshold=0` keep all pairs so a complete ranking exists. |
| `coma_plus` | same with the instance (TF-IDF) matcher = Magneto's `ComaInst`. Reference randomly sampled 500 rows; here the first 500 non-empty rows (deterministic). |
| `distribution` | valentine `DistributionBased` (default thresholds). Pairs it does not output are ranked after scored pairs in target order. |
| `similarity_flooding` | valentine `SimilarityFlooding` (inverse-average, formula C, prefix/suffix). |
| `isresmat` | independent re-implementation of the reference train-to-match inference (BERT + projector, pairwise-fragment contrastive loss, student-t column agents, Sinkhorn rectification loss, 200 column-samples/column, ranking by agent similarity). Not implemented: schema-name transformation variants (reference default uses original names), validation/early stopping; fragments > 512 tokens are truncated rather than resampled; full ranking instead of top-10. |
| `unicorn` | inference-only re-implementation (DeBERTa-base [CLS] -> 6-expert MoE -> classifier) loading the released `UnicornPlus` checkpoint (`RUC-DataLab/unicorn-plus-v1`, strict state-dict load); reference serialization `[ATT] name [VAL] v ...` (first 20 unique values), 128 tokens, score = match logit. |

All baselines implement `BaseMatcher.match(source_df, target_df) -> list[Match]`,
read the same processed data, never see ground truth and never compute metrics.

## Outputs

```
outputs/predictions/<method>/<dataset>/<case_id>.json   complete rankings
outputs/runtime/<method>/<dataset>/<case_id>.json       runtime record
outputs/status/<method>/<dataset>/<case_id>.json        success/failed marker (+ prediction hash)
outputs/logs/<method>/<dataset>/<case_id>.log           per-case log (no prompts, no keys)
outputs/debug/<run_id>/...                              --save-debug requests/responses
outputs/manifests/<run_id>.json                         git commit, config, models, seed, ...
outputs/metrics/{per_case,per_dataset,overall,completeness}.csv
outputs/scalability/{raw,predictions}/..., summary.csv
```

Predictions keep the full ranking for every source column, so new ranking
metrics can be computed without re-running models.

**Resume.** A `method x dataset x case` unit is complete iff its status is
`success`, the prediction hash matches and the file holds a full ranking.
`--resume` skips complete units and re-runs missing, partial and failed ones.
Failures are recorded (status `failed`, traceback in the log) and reported by the
evaluator in `completeness.csv`; `experiment.on_failure` chooses continue/abort.

## Runtime

Per case: `representation_seconds`, `retrieval_seconds`, `reranking_seconds`,
`matching_seconds` (baseline algorithm, incl. ISResMat in-situ training),
`ranking_seconds`, `total_seconds` (sum of components), `model_requests`,
`input_tokens`, `output_tokens`, `retries`, `failures` (failed attempts, incl.
retried ones), `peak_gpu_memory_mb` (this process only), `wall_seconds`.
Model loading happens once per process in `matcher.load()` and is excluded. On a
candidate-cache hit `retrieval_seconds` is the compute time recorded when the
entry was created (`retrieval_cache_hit=true`).

## Evaluation

One evaluator for all methods (`dema.evaluation`). Cases with empty ground truth
are skipped (none exist in the current benchmark). A *query* is a source column
with at least one GT target.

* **MRR** — mean over queries of 1 / (best rank of any correct target); 0 if none.
* **Recall@GT** — Valentine `RecallAtSizeofGroundTruth` (Magneto's metric): pool all
  pairs of the case, sort by score (ties: per-source rank, then source order), keep
  top |GT|, recall against GT.
* **Hits@K** (K = 1, 5, 10) — fraction of queries with a correct target in the top K.
* **Recall@K** (K = 1, 5, 10, 20) — pair level: fraction of GT pairs whose target is in
  the source column's top K. Recall@20 is the retrieval ceiling for the rerankers.
* **NDCG@5/10** (binary relevance) and **MAP** (optional diagnostics).

Aggregation (`weighting` column): `per_dataset.csv` = unweighted mean over cases of
the dataset; `overall.csv` has `case_macro` (mean over all cases) and
`dataset_macro` (mean of dataset means). No micro-averaging. `complete=False`
marks methods with unevaluated cases.

## Scalability

`python -m dema.scalability` (configs/scalability.yaml) subsamples target columns
deterministically (seeded per case/size/repeat), always keeps the GT target
columns and the original column order, and stores every repetition separately.
Units whose target has fewer columns than the requested size are recorded as
`insufficient_columns` — with the current benchmark GDC supports sizes up to 736
and OpenData only 50 (max width 51); no synthetic columns are added.

## Tests

```bash
python -m pytest                      # fast suite, fake model servers
DEMA_SLOW_TESTS=1 python -m pytest    # + ISResMat/Unicorn (downloads models)
```
