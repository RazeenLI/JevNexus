"""Helpers shared by the traditional baselines.

Every baseline implements :class:`dema.models.base.BaseMatcher`, reads the same
processed benchmark through the runner and returns a complete ranking built by
:func:`dema.models.ranking.ranking_from_scores`. Baselines never see ground
truth and never compute metrics themselves.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..data.types import Match
from ..evaluation.runtime import GpuMemoryProbe, RuntimeStats, timed
from ..models.base import BaseMatcher
from ..models.ranking import ranking_from_scores


class ScoreMatrixBaseline(BaseMatcher):
    """Baseline that produces a pair -> score mapping; ranking is shared code."""

    name = "score_baseline"

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = dict(cfg or {})

    def describe(self) -> dict[str, Any]:
        return {"method": self.name, "config": self.cfg}

    def compute_scores(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> dict[tuple[str, str], float]:
        raise NotImplementedError

    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Match]:
        stats = RuntimeStats()
        self.last_runtime = stats
        with GpuMemoryProbe() as probe:
            with timed(stats, "matching_seconds"):
                scores = self.compute_scores(source_df, target_df)
            with timed(stats, "ranking_seconds"):
                matches = ranking_from_scores(
                    [str(c) for c in source_df.columns], [str(c) for c in target_df.columns], scores
                )
        stats.peak_gpu_memory_mb = probe.peak_mb()
        stats.finalize()
        return matches


class ValentineBaseline(ScoreMatrixBaseline):
    """Wraps a matcher from the ``valentine`` package (stable library dependency)."""

    instance_sample_size_key = "instance_sample_size"

    def build_matcher(self):
        raise NotImplementedError

    def compute_scores(self, source_df, target_df):
        from valentine import valentine_match

        matcher = self.build_matcher()
        # Default table names ("aaa"/"bbb") share no characters, so table names
        # cannot influence name-based similarity.
        results = valentine_match(
            [source_df, target_df],
            matcher,
            instance_sample_size=self.cfg.get(self.instance_sample_size_key, 1000),
        )
        scores: dict[tuple[str, str], float] = {}
        for pair, score in results.items():
            key = (str(pair.source_column), str(pair.target_column))
            # valentine_match over two tables yields source->target pairs only.
            scores[key] = max(float(score), scores.get(key, float("-inf")))
        return scores
