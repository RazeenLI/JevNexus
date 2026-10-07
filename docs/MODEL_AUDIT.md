# JevNexus model and performance audit

## Confirmed causes of runtime cost

The complete manifest contains 561 cases and 13,522 source columns. The old
configuration submitted 20 binary questions to Open-Jev for every source
column. With `candidate_context: shared`, each independent question repeated
all 20 candidates, while the server used `--batch-size 1 --no-prefix-cache`.
Recorded JevNexus runtimes show about 35,180 seconds for 13,522 requests, including
about 34,748 seconds of reranking, or 2.57 seconds per request. Combined with
Magneto/Qwen, neural baselines, and repeated legacy scalability runs, the
greater-than-24-hour runtime was caused by configuration rather than data I/O.

The following changes were applied:

- decision context was changed to `single`, so each question contains only its
  own candidate;
- the default Open-Jev batch size was increased to 20 and prefix caching was
  enabled;
- candidate caching was limited to methods that actually use the JevNexus
  retriever;
- the development configuration was limited to five cases per dataset, and
  scalability to two cases per dataset with three repetitions;
- logs, saved state, metrics, and caches were separated, and resume no longer
  depends on log files.

On the same RTX PRO 6000 and Open-Jev process with 20 synthetic candidates, the
old `shared` request used 35,990 input tokens and 5.098 seconds. The new
`single` request used 4,270 input tokens and 0.482 seconds. This isolated test
reduced input tokens by about 88% and wall time by 10.6x. Speedups on real data
vary with the length of column values.

The end-to-end sanity case `magellan_amazon_google_exp` (4×4 columns) also
passed. New JevNexus reranking took 1.194 seconds for four requests and 3,684 input
tokens with no retries; the complete matcher took 1.245 seconds. Upstream
Magneto with Qwen took 12.931 seconds on the same case. JevNexus achieved MRR 0.875
and Magneto MRR 1.0 on this case, confirming that the runtime problem improved
while effectiveness still requires training or a predefined validation
strategy.

The server reported that `causal_conv1d` and `flash-linear-attention` were not
installed and therefore used reference PyTorch kernels. Results remain valid,
but performance is lower than with compatible optimized kernels. CUDA,
PyTorch, and model versions must be checked before installing those kernels.

Without vLLM, the Qwen service falls back to the built-in Transformers server.
The fallback exists for portability and serializes generation with a global
lock, so full experiments should use a compatible vLLM installation.

## Current primary method and ablations

The primary `dema` method retains Magneto Top-20 candidates, scores Open-Jev
single-candidate decisions and COMA++ evidence in parallel, combines their raw
scores with a fixed formula, and sends the fused Top-3 to
`jinaai/jina-reranker-v3.5` for listwise reranking:

```text
final_score = 0.4 × jev_score + 0.6 × coma_plus_score
```

Pairs omitted by COMA++ after bidirectional filtering receive a score of zero.
The retrieval tail outside the Top-20 is unchanged. The formula does not vary
by dataset, the Jina revision is pinned, and Jina is not trained on this
benchmark.

R denotes the Jina reranker and S denotes the COMA++ structured matcher:

| Code | Paper name | Pipeline |
|---|---|---|
| `dema` | JevNexus | Magneto candidates → Jev + COMA++ → Jina |
| `dema_no_rerank` | JevNexus−R | Magneto candidates → Jev + COMA++ |
| `dema_no_struct` | JevNexus−S | Magneto candidates → Jev → Jina |
| `dema_decision` | JevNexus−R−S | Magneto candidates → Jev |
| `dema_shared` | JevNexus-Shared | Magneto candidates → Jev shared context |

The 29 valid cases selected by `experiment_dev.yaml` produced these case-macro
results:

| Method | MRR | Recall@GT | Recall@20 | NDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|
| Magneto-Qwen | 0.9501 | 0.8259 | 0.9683 | 0.9500 | 0.9431 |
| **JevNexus** | **0.9611** | **0.8020** | **0.9683** | **0.9593** | **0.9540** |
| JevNexus−R | 0.9343 | 0.8050 | 0.9683 | 0.9401 | 0.9287 |
| JevNexus−S | 0.9375 | 0.6990 | 0.9683 | 0.9409 | 0.9298 |
| COMA++ | 0.9162 | 0.7382 | 0.9000 | 0.9089 | 0.9037 |
| JevNexus−R−S | 0.8845 | 0.6778 | 0.9683 | 0.9023 | 0.8781 |

All four JevNexus variants completed through the shared runner. Mean measured
runtime was 17.62 seconds per case for JevNexus, 16.89 for JevNexus−R, 16.85 for
JevNexus−S, and 16.59 for JevNexus−R−S. Magneto-Qwen averaged 133.72 seconds.

Per-dataset MRR shows the largest remaining ablation differences in ChEMBL:

| Method | ChEMBL | GDC | Magellan | OpenData | TPC-DI | WikiData |
|---|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 1.0000 | 0.7768 | 1.0000 | 0.9887 | 0.9933 | 0.9396 |
| JevNexus | 1.0000 | 0.8477 | 1.0000 | 0.9944 | 0.9933 | 0.9240 |
| JevNexus−R | 0.8604 | 0.8133 | 1.0000 | 0.9887 | 1.0000 | 0.9458 |
| JevNexus−S | 0.9373 | 0.8230 | 0.9857 | 0.9804 | 1.0000 | 0.8885 |
| COMA++ | 0.9758 | 0.5972 | 1.0000 | 0.9840 | 1.0000 | 0.9458 |
| JevNexus−R−S | 0.7894 | 0.7578 | 0.9217 | 0.9737 | 0.9933 | 0.8677 |

Relative to Magneto-Qwen, the primary method increases case-macro MRR from
0.9501 to 0.9611 while using about 13.2% of its mean runtime. The ablations show
that both the structured matcher and listwise reranker contribute. Recall@GT
remains slightly below Magneto, so further work should first examine candidate
coverage and the recall/ranking trade-off introduced by Jina rather than adding
unvalidated fusion branches.

The old names `dema_jina_rerank`, `dema_fusion`, `dema_jina_no_coma`, and
`dema_single` remain compatibility aliases. The earlier JevNexus-owned retriever
with Jev-Single is now `dema_legacy`; existing historical result directories are
not moved or overwritten.
