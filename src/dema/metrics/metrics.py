"""Ranking metrics shared by every method.

Notation for one case: ``rankings[s]`` is the ordered list of target columns for
source column ``s`` (rank 1 first); ``GT`` is the set of ground-truth pairs
``(s, t)``. A *query* is a source column that has at least one GT target.

* **MRR** — mean over queries of ``1 / r_s`` where ``r_s`` is the best (lowest)
  rank of any correct target of ``s``; ``1/r_s = 0`` if no correct target is
  ranked. (Magneto's GDC "adjusted" MRR; with one target per source column it
  equals Magneto's Valentine MRR.)
* **Recall@GT** — Valentine's ``RecallAtSizeofGroundTruth`` used by Magneto:
  pool all predicted pairs of the case, sort them globally by score, keep the
  top ``|GT|`` and report ``|kept ∩ GT| / |GT|``. Ties in score are broken by
  per-source rank, then source-column order, then rank order (deterministic).
* **Hits@K** — fraction of queries with at least one correct target in the top K.
* **Recall@K** — pair-level: fraction of GT pairs ``(s, t)`` with ``t`` in the
  top K of ``s``.
* **NDCG@K** — binary relevance, mean over queries,
  ``IDCG`` uses ``min(|gold_s|, K)`` ideal hits.
* **MAP** — mean over queries of average precision over the complete ranking.

GT pairs that reference a column absent from the tables (known raw-benchmark
defects) can never be predicted; they stay in every denominator (count as misses).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Ranking = dict[str, list[str]]
ScoredRanking = dict[str, list[tuple[str, float]]]

HITS_KS = (1, 5, 10)
RECALL_KS = (1, 5, 10, 20)
NDCG_KS = (5, 10)


def gold_by_source(ground_truth: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    for s, t in ground_truth:
        gold.setdefault(s, set()).add(t)
    return gold


def best_rank(ranked: Sequence[str] | None, gold: set[str]) -> float:
    if not ranked:
        return math.inf
    for i, t in enumerate(ranked, start=1):
        if t in gold:
            return float(i)
    return math.inf


def mrr(rankings: Ranking, ground_truth: Iterable[tuple[str, str]]) -> float:
    gold = gold_by_source(ground_truth)
    if not gold:
        return math.nan
    return sum(1.0 / best_rank(rankings.get(s), g) for s, g in gold.items()) / len(gold)


def hits_at_k(rankings: Ranking, ground_truth: Iterable[tuple[str, str]], k: int) -> float:
    gold = gold_by_source(ground_truth)
    if not gold:
        return math.nan
    return sum(best_rank(rankings.get(s), g) <= k for s, g in gold.items()) / len(gold)


def recall_at_k(rankings: Ranking, ground_truth: Iterable[tuple[str, str]], k: int) -> float:
    pairs = set(ground_truth)
    if not pairs:
        return math.nan
    hit = sum(1 for s, t in pairs if t in (rankings.get(s) or [])[:k])
    return hit / len(pairs)


def recall_at_gt(scored: ScoredRanking, ground_truth: Iterable[tuple[str, str]]) -> float:
    pairs = set(ground_truth)
    if not pairs:
        return math.nan
    pooled = []
    for s_index, (s, entries) in enumerate(scored.items()):
        for rank, (t, score) in enumerate(entries, start=1):
            pooled.append((-float(score), rank, s_index, s, t))
    pooled.sort(key=lambda x: (x[0], x[1], x[2]))
    top = {(s, t) for _, _, _, s, t in pooled[: len(pairs)]}
    return len(top & pairs) / len(pairs)


def ndcg_at_k(rankings: Ranking, ground_truth: Iterable[tuple[str, str]], k: int) -> float:
    gold = gold_by_source(ground_truth)
    if not gold:
        return math.nan
    total = 0.0
    for s, g in gold.items():
        ranked = (rankings.get(s) or [])[:k]
        dcg = sum(1.0 / math.log2(i + 1) for i, t in enumerate(ranked, start=1) if t in g)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(g), k) + 1))
        total += dcg / idcg
    return total / len(gold)


def mean_average_precision(rankings: Ranking, ground_truth: Iterable[tuple[str, str]]) -> float:
    gold = gold_by_source(ground_truth)
    if not gold:
        return math.nan
    total = 0.0
    for s, g in gold.items():
        hits, ap = 0, 0.0
        for i, t in enumerate(rankings.get(s) or [], start=1):
            if t in g:
                hits += 1
                ap += hits / i
        total += ap / len(g)
    return total / len(gold)


def all_metrics(scored: ScoredRanking, ground_truth: Sequence[tuple[str, str]]) -> dict[str, float]:
    rankings = {s: [t for t, _ in entries] for s, entries in scored.items()}
    out = {"MRR": mrr(rankings, ground_truth), "Recall@GT": recall_at_gt(scored, ground_truth)}
    for k in HITS_KS:
        out[f"Hits@{k}"] = hits_at_k(rankings, ground_truth, k)
    for k in RECALL_KS:
        out[f"Recall@{k}"] = recall_at_k(rankings, ground_truth, k)
    for k in NDCG_KS:
        out[f"NDCG@{k}"] = ndcg_at_k(rankings, ground_truth, k)
    out["MAP"] = mean_average_precision(rankings, ground_truth)
    return out


METRIC_NAMES = tuple(
    ["MRR", "Recall@GT"] + [f"Hits@{k}" for k in HITS_KS] + [f"Recall@{k}" for k in RECALL_KS]
    + [f"NDCG@{k}" for k in NDCG_KS] + ["MAP"]
)
