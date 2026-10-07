"""Fixed, training-free fusion of Jev and COMA++ scores for JevNexus."""

from __future__ import annotations

import time
from typing import Any, Sequence

import pandas as pd

from ..baselines.coma_plus import COMAPlusMatcher
from ..data.types import ColumnProfile, Match
from ..metrics.runtime import RuntimeStats
from .decision import DecisionReranker
from .jevnexus import JevNexusMatcher


class FixedScoreFusionReranker:
    """Combine raw Jev and COMA++ scores without a trained ranker.

    COMA++ returns a sparse score matrix after its bidirectional selection.
    Pairs absent from that matrix receive score zero, exactly as in the offline
    experiment that established the fixed-fusion ablation.
    """

    def __init__(
        self,
        decision: DecisionReranker,
        jev_weight: float,
        coma_plus_weight: float,
    ):
        if jev_weight < 0 or coma_plus_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if abs(jev_weight + coma_plus_weight - 1.0) > 1e-9:
            raise ValueError("fusion weights must sum to 1")
        self.decision = decision
        self.jev_weight = float(jev_weight)
        self.coma_plus_weight = float(coma_plus_weight)
        self.coma_scores: dict[tuple[str, str], float] = {}

    @property
    def context(self) -> str:
        return self.decision.context

    @property
    def include_dtype(self) -> bool:
        return self.decision.include_dtype

    def describe(self) -> dict[str, Any]:
        return {
            "type": "fixed_raw_score_fusion",
            "jev_weight": self.jev_weight,
            "coma_plus_weight": self.coma_plus_weight,
            "missing_coma_plus_score": 0.0,
            "decision": self.decision.describe(),
        }

    def score(
        self,
        source: ColumnProfile,
        candidates: Sequence[tuple[str, ColumnProfile]],
        stats: RuntimeStats,
        debug=None,
    ) -> dict[str, float]:
        jev_scores = self.decision.score(source, candidates, stats, debug=debug)
        output = {
            candidate_id: (
                self.jev_weight * float(jev_scores[candidate_id])
                + self.coma_plus_weight
                * float(self.coma_scores.get((source.name, profile.name), 0.0))
            )
            for candidate_id, profile in candidates
        }
        fusion_order = sorted(output, key=lambda cid: -output[cid])
        fusion_ranks = {candidate_id: rank for rank, candidate_id in enumerate(fusion_order, 1)}
        self.last_score_details = {
            candidate_id: {
                "jev_score": float(jev_scores[candidate_id]),
                "coma_plus_score": float(
                    self.coma_scores.get((source.name, profile.name), 0.0)
                ),
                "fusion_score": float(output[candidate_id]),
                "fusion_rank": fusion_ranks[candidate_id],
                "jina_applied": False,
            }
            for candidate_id, profile in candidates
        }
        return output


class JevNexusFusionMatcher(JevNexusMatcher):
    """Magneto candidates ranked by fixed Jev + COMA++ score fusion."""

    name = "jevnexus_fusion"

    def __init__(
        self,
        representation_cfg: dict[str, Any],
        retriever,
        decision_cfg: dict[str, Any],
        coma_plus_cfg: dict[str, Any],
        fusion_cfg: dict[str, Any],
        top_k: int,
        reranking_cfg: dict[str, Any] | None = None,
    ):
        decision_cfg = dict(decision_cfg)
        decision_cfg["candidate_context"] = "single"
        decision = DecisionReranker(decision_cfg, include_dtype=False)
        self.fusion_reranker = FixedScoreFusionReranker(
            decision,
            float(fusion_cfg.get("jev_weight", 0.4)),
            float(fusion_cfg.get("coma_plus_weight", 0.6)),
        )
        self.coma_plus = COMAPlusMatcher(coma_plus_cfg)
        super().__init__(
            representation_cfg,
            retriever,
            decision_cfg,
            top_k,
            reranking_cfg=reranking_cfg,
            reranker=self.fusion_reranker,
            candidate_context="single",
            include_dtype=False,
            name=self.name,
        )

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description["coma_plus"] = self.coma_plus.describe()
        return description

    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Match]:
        start = time.perf_counter()
        self.fusion_reranker.coma_scores = self.coma_plus.compute_scores(source_df, target_df)
        coma_seconds = time.perf_counter() - start
        try:
            matches = super().match(source_df, target_df)
        finally:
            # Never retain a potentially large per-case score matrix.
            self.fusion_reranker.coma_scores = {}
        self.last_runtime.matching_seconds += coma_seconds
        self.last_runtime.finalize()
        return matches


# Import compatibility for the former method name.
DeMaFusionMatcher = JevNexusFusionMatcher
