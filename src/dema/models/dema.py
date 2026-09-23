"""DeMa: shared retrieval + decision-model scoring (independent binary judgments)."""

from __future__ import annotations

import pandas as pd

from ..data.types import Match
from .decision_reranker import DecisionReranker
from .retriever import CandidateRetriever
from .two_stage import RetrieveRerankMatcher


class DeMaMatcher(RetrieveRerankMatcher):
    """column representation -> candidate discovery -> decision scoring -> complete ranking.

    Uses the same representation, retriever, candidate size and candidate sets as
    :class:`~dema.models.magneto_qwen.MagnetoQwenMatcher`; only the scoring model
    and formulation (one P(match) per candidate from a decision model) differ.
    """

    name = "dema"

    def __init__(self, representation_cfg, retriever: CandidateRetriever, decision_cfg, top_k, reranking_cfg=None,
                 reranker: DecisionReranker | None = None):
        reranker = reranker or DecisionReranker(
            decision_cfg, include_dtype=bool(representation_cfg.get("include_dtype", True))
        )
        super().__init__(representation_cfg, retriever, reranker, top_k, reranking_cfg)

    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Match]:
        # profiles -> candidates -> decision scores -> ranking (see RetrieveRerankMatcher.match)
        return super().match(source_df, target_df)
