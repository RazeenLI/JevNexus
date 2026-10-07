# Baseline implementation audit

Audit date: 2026-09-25. The purpose of this audit is not to claim that methods
with the same name are numerically identical. It records the source, pinned
version, and known differences of every implementation.

## Reference versions

| Project | Pinned commit | Use |
|---|---|---|
| [magneto-matcher](https://github.com/VIDA-NYU/magneto-matcher) | `6620623265fc7feac0f053996e62b68a13a72a57` | Magneto implementation |
| [data-harmonization-benchmark](https://github.com/VIDA-NYU/data-harmonization-benchmark) | `3207e37e898af06211ef1a47316f9c07dc3572cd` | COMA, COMAInst, and ISResMat experiment wrappers |
| [Valentine](https://github.com/delftdata/valentine) | `5d5163f04da304985bd51a476ccf7653de3979c3` | Traditional matcher API |
| [ISResMat](https://github.com/duxyad/ISResMat) | `7db84986ac87aac84f2296a6d6e09550dd0f420e` | ISResMat reference implementation |
| [Unicorn](https://github.com/ruc-datalab/Unicorn) | `5424e585f6c739db3c1ca610e0db079392dfbfd6` | Unicorn reference implementation |

## Findings by method

| Method | Alignment | Finding |
|---|---|---|
| `magneto_qwen` | Upstream source with documented patch | `vendor/magneto` matches the pinned commit except that `litellm.completion` is replaced by a local endpoint and request statistics are recorded. See `vendor/magneto/PATCHES.md`. Replacing GPT-4o-mini with Qwen3.5-9B makes this a controlled variant, not a reproduction of the original paper's numbers. |
| `coma` | Parameters aligned; runtime differs | The wrapper aligns `max_n=20`, schema-only matching, and COMA's default `delta=0.15`. Valentine 1.x now uses a Python implementation, while the reference benchmark used Java COMA, so score-by-score equivalence is not claimed. |
| `coma_plus` | Parameters aligned and sampling reproducible | The implementation follows `ComaInst`: at most 500 randomly sampled rows per table, the instance matcher enabled, and `max_n=20`. This repository fixes `sample_seed=42`; the upstream wrapper did not fix a seed. |
| `distribution` | Current Valentine defaults | `threshold1=0.15`, `threshold2=0.15`, `quantiles=256`, and a single process match the current implementation defaults. |
| `similarity_flooding` | Current Valentine defaults | Uses inverse-average coefficients, formula C, and the prefix/suffix string matcher. |
| `unicorn` | Inference reimplementation | Uses the released UnicornPlus checkpoint, DeBERTa-base, six experts, `[ATT]/[VAL]` serialization, the first 20 unique values, 128 tokens, and the match logit. It does not directly execute the upstream repository. Dependency upgrades require the slow state-dict compatibility test. |
| `isresmat` | Approximate reimplementation | The main hyperparameters match the reference wrapper, but the implementation omits the complete validation/early-stopping procedure and some augmentation behavior. Results must be labeled as a reimplementation rather than official numbers. |

## Ranking adaptation

The shared evaluator requires a complete ordering of every target column for
each source column. When a baseline returns only selected pairs,
`dema.model.ranking.ranking_from_scores` places unreturned pairs after scored
pairs and breaks ties stably in target-schema order. This does not expose
ground truth to the matcher, but it differs from the output protocol of scripts
that evaluate only a top-k list.

## Maintenance rules

1. Document every functional change to `vendor/magneto` in `PATCHES.md`.
2. Update this audit and rerun baseline tests after changing Valentine, a
   checkpoint, or a pinned reference commit.
3. Distinguish `upstream code`, `ported wrapper`, and `reimplementation` when
   reporting results; do not describe all baselines as official implementations.
