"""Final ranking construction and validation.

For every source column::

    reranked = top-k candidates sorted by reranker score (desc),
               ties broken by retrieval rank
    tail     = remaining targets in retrieval order
    ranking  = reranked + tail

The stored ``score`` is the reranker score for the top-k and 0.0 for the tail;
``reranker_score`` / ``retrieval_score`` keep the raw values. No calibration
across the two scales is attempted — evaluation uses ranks.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from ..data.types import Candidate, Match

TAIL_SCORE = 0.0


class RankingError(ValueError):
    pass


def build_final_ranking(
    full_ordering: Sequence[Candidate],
    rerank_scores: dict[str, float],
    top_k: int,
    score_details: dict[str, dict] | None = None,
) -> list[Match]:
    k = min(int(top_k), len(full_ordering))
    top, tail = list(full_ordering[:k]), list(full_ordering[k:])
    missing = [c.target_column for c in top if c.target_column not in rerank_scores]
    if missing:
        raise RankingError(f"reranker scores missing for candidates {missing}")
    reranked = sorted(top, key=lambda c: (-rerank_scores[c.target_column], c.retrieval_rank))
    tail = sorted(tail, key=lambda c: c.retrieval_rank)
    matches = []
    score_details = score_details or {}
    for rank, cand in enumerate(reranked, start=1):
        s = float(rerank_scores[cand.target_column])
        detail = score_details.get(cand.target_column, {})
        matches.append(Match(
            cand.source_column, cand.target_column, s, rank, s, cand.retrieval_score,
            retrieval_rank=cand.retrieval_rank,
            jev_score=detail.get("jev_score"),
            coma_plus_score=detail.get("coma_plus_score"),
            fusion_score=detail.get("fusion_score"),
            fusion_rank=detail.get("fusion_rank"),
            jina_applied=detail.get("jina_applied"),
            jina_score=detail.get("jina_score"),
            jina_rank=detail.get("jina_rank"),
            gate_activated=detail.get("gate_activated"),
            gate_disagreement=detail.get("gate_disagreement"),
            gate_margin=detail.get("gate_margin"),
            gate_threshold=detail.get("gate_threshold"),
        ))
    for rank, cand in enumerate(tail, start=len(reranked) + 1):
        matches.append(
            Match(
                cand.source_column, cand.target_column, TAIL_SCORE, rank, None,
                cand.retrieval_score, retrieval_rank=cand.retrieval_rank,
            )
        )
    return matches


def ranking_from_scores(
    source_columns: Sequence[str],
    target_columns: Sequence[str],
    scores: dict[tuple[str, str], float],
) -> list[Match]:
    """Complete ranking from a (possibly partial) pair->score mapping.

    Used by the traditional baselines. Pairs a method did not score are placed
    after all scored pairs (score ``-inf`` is not stored; they receive the
    minimum stored score and keep target-schema order). Ties are broken by
    target column position, so the ranking is deterministic.
    """
    position = {t: i for i, t in enumerate(target_columns)}
    out: list[Match] = []
    for src in source_columns:
        scored = []
        unscored = []
        for t in target_columns:
            value = scores.get((src, t))
            if value is None or (isinstance(value, float) and math.isnan(value)):
                unscored.append(t)
            else:
                scored.append((t, float(value)))
        scored.sort(key=lambda item: (-item[1], position[item[0]]))
        floor = min((v for _, v in scored), default=0.0)
        floor = min(floor, 0.0)
        ordered = scored + [(t, floor) for t in unscored]
        out.extend(Match(src, t, v, r) for r, (t, v) in enumerate(ordered, start=1))
    return out


def group_by_source(matches: Iterable[Match]) -> dict[str, list[Match]]:
    grouped: dict[str, list[Match]] = {}
    for m in matches:
        grouped.setdefault(m.source_column, []).append(m)
    for items in grouped.values():
        items.sort(key=lambda m: m.rank)
    return grouped


def validate_ranking(
    matches: Sequence[Match], source_columns: Sequence[str], target_columns: Sequence[str]
) -> None:
    """Every source column ranks every target exactly once with ranks 1..n."""
    grouped = group_by_source(matches)
    if set(grouped) != set(source_columns):
        raise RankingError(
            f"source columns mismatch: missing={sorted(set(source_columns) - set(grouped))[:5]} "
            f"extra={sorted(set(grouped) - set(source_columns))[:5]}"
        )
    expected = list(target_columns)
    expected_set = set(expected)
    for src, items in grouped.items():
        targets = [m.target_column for m in items]
        if len(targets) != len(expected) or set(targets) != expected_set:
            raise RankingError(f"ranking for {src!r} does not contain every target exactly once")
        if [m.rank for m in items] != list(range(1, len(expected) + 1)):
            raise RankingError(f"ranks for {src!r} are not 1..{len(expected)}")
        for m in items:
            if not math.isfinite(m.score):
                raise RankingError(f"non-finite score for ({src!r}, {m.target_column!r})")
