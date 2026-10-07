"""JevNexus: shared retrieval and typed correspondence decisions."""

from __future__ import annotations

import pandas as pd

from ..data.types import Match
from .decision import DecisionReranker
from .pipeline import RetrieveRerankMatcher
from .retrieval import CandidateRetriever


class JevNexusMatcher(RetrieveRerankMatcher):
    """Discover candidates, score correspondence decisions, and rank targets."""

    name = "jevnexus"

    def __init__(
        self,
        representation_cfg,
        retriever: CandidateRetriever,
        decision_cfg,
        top_k,
        reranking_cfg=None,
        reranker: DecisionReranker | None = None,
        candidate_context: str | None = None,
        include_dtype: bool | None = None,
        name: str | None = None,
    ):
        decision_cfg = dict(decision_cfg)
        if candidate_context is not None:
            decision_cfg["candidate_context"] = candidate_context
        if include_dtype is None:
            include_dtype = bool(representation_cfg.get("include_dtype", True))
        reranker = reranker or DecisionReranker(
            decision_cfg, include_dtype=include_dtype
        )
        super().__init__(representation_cfg, retriever, reranker, top_k, reranking_cfg)
        if name is not None:
            self.name = name

    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Match]:
        return super().match(source_df, target_df)
