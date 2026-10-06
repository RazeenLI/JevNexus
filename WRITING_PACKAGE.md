# DeMa Writing Package

> 事实整理日期：2026-09-26。本文档只整理 repository 中当前代码、配置和已保存输出。
> 它不是论文草稿。当前可比较的主结果来自 `configs/experiment_dev.yaml` 的 29 个 case，
> 不是 561-case 全量实验。除特别说明外，数值均来自 `metrics/*.csv` 的 `case_macro` 行。

## 1. Final Framework

### 1.1 实际 pipeline

输入是一个 benchmark case 的 source table、target table 和仅供评价使用的 ground-truth
column pairs。模型推理不读取 ground truth。

最终 `dema` 的 candidate stage 直接复用 vendored Magneto 的 candidate generator，而不是
早期 DeMa 自有 retriever。Magneto 先清洗表格，对每列使用 column name 和最多 10 个
sample values 序列化；GDC 使用 `header_values_default`，其他数据集使用 Magneto 默认
`header_values_verbose`。检索编码器是 `sentence-transformers/all-mpnet-base-v2`。embedding
similarity、exact normalized-name match、0.1 embedding threshold 和 dataset-specific encoding
均来自 Magneto。每个 source 最多保留 20 个候选；阈值可能使实际候选少于 20。

对于每个 source column，Open-Jev-9B 对候选分别给出独立二元判断概率。正式 `dema`
使用 `single` context：HTTP request 的 state 只含 source column；每个 candidate 对应一个
`noul` question，candidate 的 name 和 sample values 写在该问题中。因此一次 request 可以
包含多个 question，但每个 question 的语义上下文只展示自己的 candidate，而不是完整
candidate list。canonical DeMa 不展示 dtype、retrieval score、retrieval rank、dataset id 或
ground truth。

Open-Jev 概率与 COMA++ sparse schema/instance score 按原始分数融合：

```text
fusion_score(s,t) = 0.4 * P_Jev(match | s,t) + 0.6 * score_COMA++(s,t)
```

COMA++ 未输出的 pair 的分数为 0。融合后的 Top-3 被送入固定 revision 的
`jinaai/jina-reranker-v3.5`。Jina 的 query 是 source column 的 name + values；document 是
candidate name + values，并附带 Jev、COMA++ 和 fixed-fusion 三个分数。Jina 只返回顺序；
代码把原 Top-3 的 fusion score slots 按 Jina 顺序重新分配，保留原分数分布。Top-3 之外、
Top-20 以内的 candidate 保持 fusion 分数；candidate set 外的 target columns 作为 tail，
按照 retrieval order 排在后面且保存分数 0。没有 post-reranking threshold、额外归一化或
dataset-specific fusion weight。

候选结果持久化在 `cache/candidates_magneto/`。cache key 包含 dataset、case、top-k、
Magneto 参数和 source/target 完整数据 fingerprint。

### 1.2 ASCII pipeline

```text
source table + target table
        |
        v
Magneto cleaning + column serialization
(name + <=10 sampled values; no dtype in canonical DeMa)
        |
        v
MPNet embedding retrieval + exact-name matches + threshold 0.1
        |
        v
per-source candidate set (up to Top-20; cached)
        |                         \
        |                          -> COMA++ schema + instance scores
        v
Open-Jev-9B independent P(match) for each candidate
        |                         /
        v
0.4 * Jev + 0.6 * COMA++
        |
        v
Jina reranker-v3.5 listwise reorder of fusion Top-3
        |
        v
Top-20 ranked by resulting scores + untouched retrieval tail
        |
        v
complete target-column ranking for every source column
```

### 1.3 关键实现位置

| Stage | File | Class/function |
|---|---|---|
| Method construction | `src/dema/model/registry.py` | `build_matcher`, `_dema_experimental` |
| Shared pipeline | `src/dema/model/pipeline.py` | `RetrieveRerankMatcher.match` |
| Magneto candidates | `src/dema/model/magneto_retrieval.py` | `MagnetoCandidateRetriever` |
| Decision request | `src/dema/model/decision.py` | `DecisionReranker.build_request`, `score` |
| Fixed fusion | `src/dema/model/fusion.py` | `FixedScoreFusionReranker` |
| Jina stage | `src/dema/model/experimental_rerank.py` | `JinaTopReranker`, `DeMaJinaRerankMatcher` |
| Final complete ranking | `src/dema/model/ranking.py` | `build_final_ranking` |

### 1.4 最终 config values

| Item | Value |
|---|---|
| Retrieval encoder | `sentence-transformers/all-mpnet-base-v2` |
| Candidate top-k | 20 |
| Candidate embedding threshold | 0.1 (Magneto default/output cache metadata) |
| Candidate sampling | Magneto `mixed`, size 10 |
| Decision model | `ZefanCai/Open-Jev-9B` |
| Decision base revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Open-Jev checkpoint revision | `47e966881e489511c0c7f5633a9e1960a676a551` |
| Decision context | `single` |
| Decision max length / batch | 4096 / 20 |
| Decision dtype | **UNCONFIRMED** in project config; server package controls it |
| Fusion | Jev 0.4, COMA++ 0.6 |
| Jina model | `jinaai/jina-reranker-v3.5` |
| Jina revision | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` |
| Jina top-n | 3 |
| Jina input includes component scores | true |
| Candidate dtype shown to Jev/Jina | false |
| Retry policy | 3 additional attempts for Jev |
| Request timeout | 300 s |

## 2. DeMa vs Generative Baseline

The generative baseline is `magneto_qwen`: vendored Magneto commit
`6620623265fc7feac0f053996e62b68a13a72a57`, with only the upstream `litellm`
transport replaced by a local OpenAI-compatible Qwen endpoint.

| Component | Generative baseline (`magneto_qwen`) | DeMa (`dema`) |
|---|---|---|
| Benchmark input | Same source/target CSV | Same source/target CSV |
| Representation | Magneto cleaning; name + 10 samples | Same Magneto representation |
| Retriever | Magneto MPNet candidate generator | Same generator exposed through adapter/cache |
| Candidate set | Up to Top-20 | Same candidate rules and Top-20 |
| Semantic model | `Qwen/Qwen3.5-9B` | `ZefanCai/Open-Jev-9B`; then Jina v3.5 Top-3 |
| Model input | One source and all candidate columns jointly | Jev: source + one candidate per question; Jina: source + fusion Top-3 jointly |
| Output | Generated JSON array of target names and scores | Jev scalar `P(true)` per question; Jina ordering |
| Ranking | Qwen scores; parse failure retries up to 5 then retrieval fallback | 0.4 Jev + 0.6 COMA++; Jina permutes Top-3 score slots |
| Other differences | No COMA++ or Jina | COMA++ and Jina are additional stages |

Qwen3.5-9B and Open-Jev-9B are both identified as 9B models in their configured model IDs.
Open-Jev is configured as a LoRA adapter plus scalar decision head over a pinned Qwen3.5-9B base.
The exact trainable/total parameter counts are not stored: **UNCONFIRMED**.

Generative prompt structure (from `vendor/magneto/magneto/llm_reranker.py`): system role says the
model performs schema matching by providing column similarity scores. The user prompt provides one
`Candidate Column: Column: <name>, Sample values: [...]`, then all `Target Schemas`, and asks for a
JSON array containing `column` and a 0–1 two-decimal `score`, with no explanation. Temperature is 0
and thinking is disabled.

DeMa question structure (from `configs/models.yaml` and `DecisionReranker`):

```text
State:
Source column:
name: <source name>
values: <v1> | ... | <v10>

Question for candidate ci:
Does the following target column represent the same underlying schema attribute
as the source column?
name: <target name>
values: <v1> | ... | <v10>

true criterion: same real-world attribute; values could be mapped
false criterion: different attributes
```

No hidden reasoning is requested or stored.

## 3. Benchmark

### 3.1 Full manifest inventory

All 561 manifest cases have non-empty ground truth. Counts below are computed from
`data/manifests/all.jsonl`; target-column totals count the target schema once per case.

| Dataset | # Cases | Source columns total/range | Target columns total/range | # GT pairs total/range | Notes |
|---|---:|---:|---:|---:|---|
| ChEMBL | 180 | 3,060 / 12–23 | 3,132 / 12–23 | 2,052 / 1–23 | Valentine |
| GDC | 10 | 569 / 16–179 | 7,360 / 736 | 259 / 10–43 | Shared 736-column GDC target |
| Magellan | 7 | 41 / 4–9 | 41 / 4–9 | 41 / 4–9 | Valentine |
| OpenData | 180 | 6,876 / 26–51 | 6,876 / 26–51 | 4,572 / 1–51 | Valentine |
| TPC-DI | 180 | 2,916 / 11–22 | 3,024 / 12–22 | 1,980 / 1–22 | Valentine |
| WikiData | 4 | 60 / 13–20 | 60 / 13–20 | 40 / 6–20 | Two known invalid GT pairs retained |
| **Total** | **561** | **13,522** | **20,493** | **8,944** | |

Main reported results use 5 evenly selected cases per dataset under `experiment_dev.yaml`, except
WikiData, which has only 4: 29 cases total. The current repository does not contain a completed
561-case result matrix for the final `dema` and all baselines.

### 3.2 Sources and preprocessing

- GDC-SM comes from Zenodo record `14963588`; source tables are rebuilt using the official paper
  metadata/download procedure and the supplied common target table.
- ChEMBL, Magellan, OpenData, TPC-DI and WikiData come from Valentine datasets, Zenodo record
  `5084605`.
- `src/dema/data/prepare.py` discovers raw cases and writes a unified layout under
  `data/processed/<dataset>/<case_id>/`.
- Processing normalizes layout, CSV encoding to UTF-8 when necessary, and ground-truth file shape.
  Column names, cells, rows and GT pairs are not changed; integrity checks compare processed data
  with raw content and hashes.
- Ground truth is read from `ground_truth.csv` as `(source_column,target_column)` pairs.
- GDC discovery and Valentine JSON mapping discovery are dataset-specific. Magneto candidate
  serialization is also dataset-specific: GDC `header_values_default`; others upstream default.
- `givenName -> forename` in `wikidata_musicians_unionable` and
  `wikidata_musicians_viewunion` references an absent raw target column. Both pairs remain in metric
  denominators and necessarily count as misses.

## 4. Compared Methods

| Method | Category | Implementation summary | Training required in experiment? | External model/dependency |
|---|---|---|---|---|
| COMA | Traditional | Valentine COMA, schema matcher only | No | `valentine` |
| COMA++ | Traditional | COMA with instance TF-IDF matcher, seeded 500-row sampling | No | `valentine` |
| Distribution | Traditional | Valentine distribution-based matcher | No | `valentine`, PuLP/CBC |
| Similarity Flooding | Traditional | Valentine graph propagation, formula C | No | `valentine` |
| ISResMat | Neural in-situ | Independent implementation; BERT is trained per case | **Yes, per case** | `bert-base-uncased` |
| Unicorn | Pretrained neural | Zero-shot UnicornPlus inference | No local training | `microsoft/deberta-base`, pinned UnicornPlus checkpoint |
| Magneto-Qwen | Generative LLM | Upstream Magneto candidate and LLM reranking pipeline | No | MPNet, `Qwen/Qwen3.5-9B` |
| DeMa | Decision/fusion/listwise | Magneto candidates → Jev + COMA++ → Jina Top-3 | No | Open-Jev-9B, COMA++, Jina v3.5 |

All rows in the current metric files were recalculated by this repository from saved predictions;
no metric row is marked as imported. Magneto code is vendored upstream code with the documented
transport patch. ISResMat and Unicorn are project implementations using their stated reference
behavior/checkpoints.

Reported ablations are `dema_no_rerank` (DeMa−R), `dema_no_struct` (DeMa−S),
`dema_decision` (DeMa−R−S), and `dema_shared`. Implemented and completed but not part of the final
method are `dema_jev_weight`, `dema_own_retrieval`, and `dema_legacy`. Jaccard and Levenshtein are
implemented but have no rows in the final metric files.

## 5. Experimental Configuration

- Seed: 42.
- Main result config: `configs/experiment_dev.yaml`; maximum 5 cases per dataset.
- Full intended config: `configs/experiment.yaml`; no case limit, but final complete matrix absent.
- Qwen: `Qwen/Qwen3.5-9B`, bfloat16, maximum model length 32,768, temperature 0,
  `max_tokens=1024`, thinking disabled and request timeout 300 s. Magneto permits up to 5
  generation/parse attempts; its local HTTP transport does not add a separate retry loop. The
  `qwen.max_retries=3` config belongs to the generic project Qwen client and is not consumed by
  `magneto_qwen`'s `_local_llm.py` transport.
- Qwen backend: configured `auto`; `~/.venvs/vllm` contains vLLM 0.30.0. Server logs prove vLLM
  was launched during some attempts, but output metadata does not record the backend per case;
  backend for every successful Magneto case is therefore **UNCONFIRMED**.
- vLLM settings: GPU memory utilization 0.85, prefix caching/chunked prefill enabled by vLLM log,
  native sampler forced with `VLLM_USE_FLASHINFER_SAMPLER=0` by serving script.
- Open-Jev: System One endpoint `/v1/systemone`, max length 4096, batch 20, prefix cache enabled,
  device `cuda:0`, timeout 300 s, maximum 3 retries after the first attempt.
- Jina: `dtype=auto`, `device_map=auto`, Top-3, component scores included.
- MPNet retrieval batch: 64 in generic retriever config; canonical DeMa uses upstream Magneto's
  embedding implementation and its own batching behavior.
- Experiment runner processes methods sequentially. Service and worker GPU placement is controlled
  by CLI `--gpus`; no case-level concurrent workers are configured for these results.
- Output manifests record Python 3.11.16 and PyTorch `2.14.0+cu130`.
- Current environment: Transformers 5.17.0, pandas 2.3.3, scikit-learn 1.9.1, vLLM 0.30.0.
  These are current-environment values, not all guaranteed to be the versions at every run.
- Hardware model: `docs/MODEL_AUDIT.md` records an RTX PRO 6000, but the result manifest does not
  contain GPU name and current `nvidia-smi` is unavailable: **UNCONFIRMED from machine-readable
  experiment metadata**.
- Resume requires a successful status record, matching experiment fingerprint, prediction hash,
  runtime file and complete ranking. Candidate retrieval cache is persistent.

## 6. Evaluation Metrics

Let `GT` be the unique ground-truth pair set. A query is a source column with at least one GT
target. All rankings are complete over the target schema.

| Metric | Actual definition |
|---|---|
| MRR | Mean over GT-bearing source queries of `1 / min rank(correct target)`; 0 when none is ranked. With multiple GT targets, only the best rank is used. |
| Recall@GT | Pool every `(source,target,score)` in a case globally, sort by descending score, take exactly `|GT|` pairs, return intersection size divided by `|GT|`. Deterministic ties: per-source rank, source insertion order, then rank. |
| Hits@K | Fraction of GT-bearing source queries having at least one correct target in their own Top-K. K = 1, 5, 10. |
| Recall@K | Fraction of unique GT pairs whose target is within that source query's Top-K. K = 1, 5, 10, 20. |
| NDCG@K | Binary relevance NDCG per GT-bearing source, then query mean. IDCG contains `min(#gold_for_source,K)` hits. K = 5, 10. |
| MAP | Per-source AP over the complete ranking, divided by all gold targets for that source, then query mean. |
| Runtime | Component sum: representation + retrieval + reranking + traditional matching + ranking. Model/server startup excluded. Cached retrieval reports original compute time, not cache lookup latency. |
| Tokens/calls | Sum of backend-reported input/output token counts and request counter. Jina increments requests but does not add token counts. |
| GPU memory | Peak CUDA memory allocated by the experiment worker process only; external Qwen/Jev server memory is excluded. |

Per-dataset rows are unweighted means over cases (`case_macro`). Overall `case_macro` is the
unweighted mean over all 29 cases. Overall `dataset_macro` is an unweighted mean over six dataset
means. No micro-average is calculated. Definitions are implemented in
`src/dema/metrics/metrics.py`; aggregation is in `src/dema/metrics/aggregate.py`.

## 7. Main Results

### 7.1 Overall results (29-case case macro)

| Method | Complete | MRR | Recall@GT | Hits@1 | Recall@20 | NDCG@10 | MAP | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COMA | 29/29 | 0.848554 | 0.718148 | 0.828656 | 0.889665 | 0.848044 | 0.838538 | 0.281111 |
| COMA++ | 29/29 | 0.916160 | 0.738211 | 0.907800 | 0.899951 | 0.908903 | 0.903711 | 0.503428 |
| Distribution | **24/29** | 0.774353 | 0.572148 | 0.723041 | 0.975496 | 0.797671 | 0.774353 | 1.232582 |
| Similarity Flooding | 29/29 | 0.856297 | 0.676639 | 0.818987 | 0.932023 | 0.866961 | 0.845137 | 34.231606 |
| ISResMat | 29/29 | 0.825082 | 0.728907 | 0.795998 | 0.920488 | 0.829139 | 0.822338 | 50.348989 |
| Unicorn | 29/29 | 0.869214 | 0.669951 | 0.829676 | 0.935111 | 0.883231 | 0.864433 | 10.889511 |
| Magneto-Qwen | 29/29 | 0.950083 | **0.825850** | 0.938560 | 0.968338 | 0.950017 | 0.943096 | 133.719250 |
| **DeMa** | **29/29** | **0.961135** | 0.802018 | **0.949599** | **0.968338** | **0.959310** | **0.954000** | **17.618727** |

Distribution excludes all five selected GDC cases; its aggregate must not be treated as directly
comparable to the 29-case rows.

### 7.2 Per-dataset MRR

| Method | ChEMBL | GDC | Magellan | OpenData | TPC-DI | WikiData |
|---|---:|---:|---:|---:|---:|---:|
| COMA | 0.8485 | 0.5068 | 1.0000 | 0.9625 | 0.9765 | 0.7841 |
| COMA++ | 0.9758 | 0.5972 | 1.0000 | 0.9840 | 1.0000 | 0.9458 |
| Distribution | 0.7286 | NA | 0.7217 | 0.7485 | 0.8718 | 0.8079 |
| Similarity Flooding | 0.8877 | 0.5714 | 1.0000 | 0.9343 | 0.9604 | 0.7659 |
| ISResMat | 0.9368 | 0.4397 | 1.0000 | 0.6901 | 0.9622 | 0.9458 |
| Unicorn | 0.8182 | 0.6077 | 0.9482 | 0.9880 | 0.9593 | 0.9000 |
| Magneto-Qwen | 1.0000 | 0.7768 | 1.0000 | 0.9887 | 0.9933 | 0.9396 |
| **DeMa** | **1.0000** | **0.8477** | **1.0000** | **0.9944** | **0.9933** | **0.9240** |

### 7.3 Per-dataset Recall@GT

| Method | ChEMBL | GDC | Magellan | OpenData | TPC-DI | WikiData |
|---|---:|---:|---:|---:|---:|---:|
| COMA | 0.6000 | 0.3060 | 1.0000 | 0.7543 | 0.9733 | 0.6646 |
| COMA++ | 0.5625 | 0.3713 | 1.0000 | 0.6629 | 0.9867 | 0.8729 |
| Distribution | 0.6875 | NA | 0.6162 | 0.4743 | 0.4533 | 0.6438 |
| Similarity Flooding | 0.5750 | 0.2995 | 1.0000 | 0.6400 | 0.8933 | 0.6458 |
| ISResMat | 0.7250 | 0.2558 | 1.0000 | 0.6286 | 0.9200 | 0.8729 |
| Unicorn | 0.5125 | 0.3404 | 0.8881 | 0.7714 | 0.7200 | 0.8167 |
| Magneto-Qwen | **0.8000** | **0.5278** | 1.0000 | **0.7771** | 0.9733 | 0.8896 |
| DeMa | 0.7750 | 0.4960 | 1.0000 | 0.6857 | 0.9733 | **0.9021** |

No mean ± standard deviation table exists for the main experiment.

## 8. Efficiency Results

| Method | Total 29-case runtime (s) | Mean / median / p95 case runtime (s) | Mean rerank (s) | Calls | Input / output tokens | Retries / failures | Worker peak GPU MB mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 3877.858 | 133.719 / 90.314 / 356.109 | 0.000* | 790 | 898,003 / 302,299 | 0 / 0 | 1804.5 / 4757.7 |
| DeMa | 510.943 | 17.619 / 14.914 / 39.284 | 16.458 | 1,580 | 2,836,470 / 0 | 0 / 0 | 1689.4 / 1715.3 |
| DeMa−R | 489.674 | 16.885 / 14.588 / 38.506 | 15.712 | 790 | 2,836,470 / 0 | 0 / 0 | 418.7 / 418.7 |
| DeMa−S | 488.747 | 16.853 / 14.587 / 37.831 | 16.164 | 1,580 | 2,836,470 / 0 | 0 / 0 | 1680.7 / 1716.1 |
| DeMa−R−S | 481.067 | 16.589 / 14.266 / 37.088 | 15.911 | 790 | 2,836,470 / 0 | 0 / 0 | 418.7 / 418.7 |

`*` Magneto is implemented through `ScoreMatrixBaseline`, so its LLM work is included in the
baseline matching/total scope rather than `reranking_seconds`. DeMa counts one Jev request per
source and one Jina call per source, hence 1,580 calls versus Magneto's 790. Jina tokenization is
not added to token counters. Jev's System One response reports probabilities rather than generated
tokens, hence output tokens are 0. GPU memory excludes external model-server processes and is not a
whole-system VRAM comparison.

DeMa's measured mean runtime is 0.13176 times Magneto's (about 7.59x lower), while its total input
token counter is larger because it records the System One decision input accounting and Magneto's
counter follows the OpenAI-compatible server usage format. These token totals are not necessarily
tokenizer-identical.

## 9. Additional Experiments

### 9.1 Component ablation (completed, 29 cases)

Independent variables are presence of Jina reranker (R) and COMA++ structured matcher (S).
Candidates, Open-Jev, top-k, datasets and evaluator are controlled.

| Method | R | S | MRR | Recall@GT | NDCG@10 | MAP | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeMa | yes | yes | 0.961135 | 0.802018 | 0.959310 | 0.954000 | 17.618727 |
| DeMa−R | no | yes | 0.934325 | 0.804973 | 0.940075 | 0.928706 | 16.885323 |
| DeMa−S | yes | no | 0.937460 | 0.699029 | 0.940884 | 0.929767 | 16.853357 |
| DeMa−R−S | no | no | 0.884503 | 0.677800 | 0.902291 | 0.878137 | 16.588503 |

The full system has the highest MRR/NDCG/MAP. Removing Jina slightly raises Recall@GT by 0.002955.

### 9.2 Single vs shared Jev context (completed, 29 cases)

`dema_decision` uses single-candidate questions; `dema_shared` repeats all candidate information in
shared state. Both use Magneto candidates and Jev without COMA++/Jina.

| Context | MRR | Recall@GT | Mean runtime | Input tokens |
|---|---:|---:|---:|---:|
| Single | 0.884503 | 0.677800 | 16.5885 s | 2,836,470 |
| Shared | 0.772615 | 0.526434 | 66.1748 s | 15,397,100 |

Shared context is 3.99x slower by mean case runtime and uses 5.43x input tokens in these outputs.

### 9.3 Dynamic Jev weight (completed, not final)

`dema_jev_weight` adds one Jev evidence-reliability decision per source and maps its probability to
a Jev weight in `[0.1,0.7]`; `P=0.5` yields the fixed 0.4 weight.

| Method | MRR | Recall@GT | Mean runtime | Calls |
|---|---:|---:|---:|---:|
| Fixed fusion (DeMa−R) | 0.934325 | 0.804973 | 16.8853 s | 790 |
| Dynamic weight | 0.932366 | 0.799185 | 19.2988 s | 1,580 |

The dynamic variant is lower on both reported ranking metrics and slower.

### 9.4 Retrieval variants (completed, not final)

`dema_legacy` (early own retriever + single Jev) has MRR 0.841492 and Recall@GT 0.651847 on the
29-case table. `dema_own_retrieval` (own retriever + shared Jev) has MRR 0.871679,
Recall@GT 0.658799 and mean runtime 66.1831 s. They are not controlled final-method comparisons
because context and/or downstream components differ.

### 9.5 Scalability output (historical/incompatible with final config)

`saves/scalability/summary.csv` contains completed GDC rows only for COMA++ and Unicorn at target
sizes 50/100/200/400, each marked `n_units=100`. For example, mean runtime rises from 0.3014 to
1.4839 s for COMA++, and 3.2116 to 25.6259 s for Unicorn. No final `dema` row is present. The
current `configs/scalability.yaml` instead specifies 2 cases per dataset and 3 repeats, so the saved
summary was produced under an earlier configuration. It must not be reported as the current final
scalability experiment without recovering its exact historical config.

No completed top-k sensitivity, candidate-size sensitivity for final DeMa, statistical significance
test, or mean±std main experiment is present.

## 10. Representative Examples

### GDC `gdc_dou`, source `Histologic_Grade_FIGO`

Ground truth is `tumor_grade`. Source samples are `FIGO grade 2`, `FIGO grade 1`, `FIGO grade 3`.
The table below uses the real Magneto candidate cache and prediction files. `Jev P` is taken from
the controlled `dema_decision` run; it is not separately persisted inside the final DeMa output.
`DeMa score` is the final fused score slot after Jina reordering, not a probability.

| Candidate | Retrieval rank / score | Magneto score / rank | Jev P / rank | DeMa score / rank | GT |
|---|---:|---:|---:|---:|---|
| `tumor_grade` | 4 / 0.662075 | score not in Magneto Top-5 / rank 14 | 0.684012 / 3 | 0.541164 / 1 | yes |
| `who_nte_grade` | 7 / 0.587044 | below Top-5 | 0.695176 / 2 | 0.306147 / 2 | no |
| `enneking_msts_grade` | 8 / 0.586585 | below Top-5 | 0.765367 / 1 | 0.278070 / 3 | no |
| `who_cns_grade` | 1 / 0.680149 | 0.45 / 2 | 0.104596 / 5 | 0.041839 / 5 | no |
| `figo_stage` | 5 / 0.603675 | 0.85 / 1 | below Jev Top-5 | below DeMa Top-5 | no |
| `ajcc_pathologic_stage` | 9 / 0.576644 | 0.40 / 3 | below Jev Top-5 | below DeMa Top-5 | no |

Selected target samples: `tumor_grade`: G1/G2/G3/G4/GB/GX/High Grade/Intermediate Grade/Low
Grade/Unknown; `who_nte_grade`: G1/G2/G3/GX/Unknown/Not Reported; `figo_stage`: Stage 0/Stage
I/Stage IA/Stage IA1/Stage IA2/Stage IB/… . Magneto ranks the stage field first and the correct
grade field 14th; final DeMa ranks the correct field first.

## 11. Failure Cases

The following are factual rank comparisons using best rank among all GT targets. “Wrong” means
best-GT rank > 1; it does not imply the correct target is absent.

### DeMa rank 1 / Magneto rank > 1

| Dataset/case | Source | GT | DeMa rank | Magneto rank |
|---|---|---|---:|---:|
| OpenData / `opendata_joinable_miller2_vertical_70_ac4_ev` | `Enviromental sustainability (marker)` | `miller2_EnvSustain` | 1 | 2 |
| OpenData / same case | `Regional program (marker)` | `miller2_RePr` | 1 | 2 |
| GDC / `gdc_dou` | `Histologic_Grade_FIGO` | `tumor_grade` | 1 | 14 |

### Magneto rank 1 / DeMa rank > 1

| Dataset/case | Source | GT | DeMa rank | Magneto rank |
|---|---|---|---:|---:|
| WikiData / `wikidata_musicians_semjoinable` | `musicianLabel` | `musicianName` | 2 | 1 |
| GDC / `gdc_dou` | `Proteomics_Tumor_Normal` | `tissue_type` | 2 | 1 |
| GDC / `gdc_dou` | `Treatment_naive` | `treatment_or_therapy` | 2 | 1 |

### Both rank GT below 1

| Dataset/case | Source | GT | DeMa rank | Magneto rank |
|---|---|---|---:|---:|
| TPC-DI / `tpc_di_joinable_prospect_vertical_70_ac4_ev` | `NumberCars` | `prospect_NuCa` | 2 | 2 |
| WikiData / `wikidata_musicians_unionable` | `musicianLabel` | `musicianName` | 2 | 2 |
| GDC / `gdc_dou` | `tumor_Stage-Pathological` | `ajcc_pathologic_stage`, `uicc_pathologic_stage` | 2 | 7 |

### Retrieval miss: GT outside reranked candidate set

| Dataset/case | Source | GT | Final best rank (both) |
|---|---|---|---:|
| OpenData / `opendata_semantically_joinable_miller2_vertical_70_ac1_av` | `Division name` | `miller2_DivisionName` | 40 |
| GDC / `gdc_dou` | `Histologic_type` | `primary_diagnosis` | 640 |
| GDC / `gdc_dou` | `Path_Stage_Primary_Tumor-pT` | `ajcc_pathologic_t`, `uicc_pathologic_t` | 572 |

For these rows, final DeMa prediction stores no `reranker_score` for any GT target, confirming that
the decision/fusion/Jina stages could not recover it.

## 12. Key Quantitative Findings

1. DeMa case-macro MRR is 0.961135 versus 0.950083 for Magneto-Qwen: +0.011053 absolute.
2. DeMa case-macro Recall@GT is 0.802018 versus 0.825850 for Magneto-Qwen: −0.023832.
3. DeMa mean case runtime is 17.6187 s versus 133.7193 s for Magneto-Qwen: 7.59x lower.
4. On GDC, DeMa MRR is 0.847655 versus Magneto-Qwen 0.776765: +0.070890.
5. On OpenData, DeMa MRR is 0.994429 versus Magneto-Qwen 0.988714: +0.005714.
6. On WikiData, DeMa MRR is 0.923958 versus Magneto-Qwen 0.939583: −0.015625.
7. Removing both COMA++ and Jina lowers MRR from 0.961135 to 0.884503: −0.076632.
8. Single Jev context reduces mean runtime from 66.1748 s to 16.5885 s relative to shared context,
   while MRR rises from 0.772615 to 0.884503.
9. Dynamic Jev weighting produces MRR 0.932366 versus 0.934325 for fixed fusion and increases mean
   runtime from 16.8853 s to 19.2988 s.
10. DeMa and Magneto have identical case-macro Recall@20, 0.968338, because they share the same
    candidate generator on this result set.

## 13. Negative / Unexpected Results

- Final DeMa has lower Recall@GT than Magneto overall and on ChEMBL, GDC and OpenData.
- On WikiData, Magneto has higher MRR (0.939583 vs 0.923958); COMA++ also has higher MRR
  (0.945833) than DeMa.
- On TPC-DI, COMA++ MRR is 1.000000 while DeMa and Magneto are both 0.993333.
- DeMa−R has slightly higher Recall@GT than full DeMa (0.804973 vs 0.802018), so Jina improves
  ordering metrics but does not improve this pooled score metric.
- Distribution has no completed GDC results and only 24/29 cases overall.
- Retrieval misses remain unrecoverable by all rerankers; examples have GT at ranks 40, 572, 640.
- Shared Jev context is slower and less accurate than single context in the saved experiment.
- The dynamic-weight experiment is slower and lower-scoring than fixed fusion.
- Similarity Flooding and ISResMat have high runtime variance: p95 173.51 s and 226.88 s,
  respectively.
- Qwen/vLLM logs contain failed earlier launches caused by FlashInfer/CUDA toolchain mismatch. The
  serving script now disables the FlashInfer sampler. These failed launches are not included in the
  successful 29-case metric table.
- The saved scalability summary does not match the current scalability config and lacks final DeMa.

## 14. Reproducibility Information

### Repository state and output provenance

- Current HEAD while preparing this package: `252d8703d78165c11745c32be4c1dd4d7a596d55`.
- Working tree is dirty; therefore this hash alone does not identify the current implementation.
- The full-DeMa historical run manifest records commit
  `1c6efe2ac583abb8448794f943b2b40f743445bc`, also with `git_dirty: true`, run id
  `20260926-064415-dema_jina_rerank-9f29ac`.
- Result directories were subsequently renamed to the final method names. Historical run ids and
  manifests intentionally retain old names for provenance.
- A clean final commit/tag is still required before claiming an immutable reproduction revision.

### Commands

Main development experiment using current names:

```bash
source scripts/_env.sh
$PYTHON -m dema run --config configs/experiment_dev.yaml \
  --methods coma coma_plus distribution similarity_flooding isresmat unicorn \
  magneto_qwen dema dema_no_rerank dema_no_struct dema_decision dema_shared \
  --gpus 1 --server-timeout 900

$PYTHON -m dema.metrics.evaluator --config configs/experiment_dev.yaml \
  --methods coma coma_plus distribution similarity_flooding isresmat unicorn \
  magneto_qwen dema dema_no_rerank dema_no_struct dema_decision dema_shared
```

The top-level equivalent is:

```bash
EXPERIMENT_CONFIG=configs/experiment_dev.yaml bash scripts/run_all.sh
```

### Models/checkpoints

- Retrieval: `sentence-transformers/all-mpnet-base-v2`.
- Qwen: `Qwen/Qwen3.5-9B`; exact revision is not pinned in config: **UNCONFIRMED**.
- Open-Jev package: `ZefanCai/Open-Jev-9B` revision
  `47e966881e489511c0c7f5633a9e1960a676a551`.
- Open-Jev base model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Jina: `jinaai/jina-reranker-v3.5` revision
  `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.
- Unicorn: `RUC-DataLab/unicorn-plus-v1` revision
  `8fcc213b6d0c7f315c6044b4837810b32434c775`.
- ISResMat encoder: `bert-base-uncased`; exact revision not pinned.

### Artifact locations

- Predictions: `saves/predictions/<method>/<dataset>/<case>.json`.
- Runtime: `saves/runtime/<method>/<dataset>/<case>.json`.
- Completion/hash status: `saves/status/<method>/<dataset>/<case>.json`.
- Run provenance: `saves/manifests/*.json`.
- Metrics: `metrics/overall.csv`, `per_dataset.csv`, `per_case.csv`, `completeness.csv`.
- Human-readable case/server logs: `logs/`.

## 15. Relevant Files

| Purpose | Path |
|---|---|
| Final framework registry | `src/dema/model/registry.py` |
| Retrieve/rerank orchestration | `src/dema/model/pipeline.py` |
| Magneto candidate adapter/cache | `src/dema/model/magneto_retrieval.py` |
| Jev request and probability parsing | `src/dema/model/decision.py` |
| Fixed fusion | `src/dema/model/fusion.py` |
| Jina and dynamic-weight experiments | `src/dema/model/experimental_rerank.py` |
| Final ranking behavior | `src/dema/model/ranking.py` |
| DeMa question templates and model IDs | `configs/models.yaml` |
| Magneto generative prompt | `vendor/magneto/magneto/llm_reranker.py` |
| Magneto patch record | `vendor/magneto/PATCHES.md` |
| Benchmark source/layout config | `configs/paths.yaml` |
| Benchmark preparation/integrity | `src/dema/data/prepare.py` |
| Full benchmark manifest | `data/manifests/all.jsonl` |
| Actual 29-case experiment config | `configs/experiment_dev.yaml` |
| Intended full experiment config | `configs/experiment.yaml` |
| Qwen serving | `scripts/serve_qwen.sh` |
| Open-Jev serving | `scripts/serve_decision.sh` |
| Metric definitions | `src/dema/metrics/metrics.py` |
| Aggregation definitions | `src/dema/metrics/aggregate.py` |
| Evaluator | `src/dema/metrics/evaluator.py` |
| Overall main results | `metrics/overall.csv` |
| Per-dataset results | `metrics/per_dataset.csv` |
| Per-case/error-analysis source | `metrics/per_case.csv` |
| Completeness | `metrics/completeness.csv` |
| Efficiency inputs | `saves/runtime/` |
| Representative DeMa prediction | `saves/predictions/dema/GDC/gdc_dou.json` |
| Representative Magneto prediction | `saves/predictions/magneto_qwen/GDC/gdc_dou.json` |
| Representative Jev-only probability | `saves/predictions/dema_decision/GDC/gdc_dou.json` |
| Representative retrieval cache | `cache/candidates_magneto/GDC/gdc_dou/cfde21135c139b6222a17fa6.json` |
| Historical scalability summary | `saves/scalability/summary.csv` |
