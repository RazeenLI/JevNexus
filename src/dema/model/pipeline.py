"""DeMa's retrieve-and-rerank pipeline.

    column representation -> candidate retrieval -> reranker scoring -> complete ranking

The upstream Magneto baseline intentionally does not use this pipeline: it
keeps Magneto's own profiling, retrieval, and reranking implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import pandas as pd

from ..data.types import ColumnProfile, Match
from ..metrics.runtime import GpuMemoryProbe, RuntimeStats, timed
from .representation.profile import profile_table
from .base import BaseMatcher
from .prompting import present_candidates
from .ranking import build_final_ranking
from .retrieval import CandidateRetriever


class Reranker(Protocol):
    def score(
        self, source: ColumnProfile, candidates: Sequence[tuple[str, ColumnProfile]],
        stats: RuntimeStats, debug=None,
    ) -> dict[str, float]:
        ...

    def describe(self) -> dict[str, Any]:
        ...


class RetrieveRerankMatcher(BaseMatcher):
    name = "retrieve_rerank"

    def __init__(
        self,
        representation_cfg: dict[str, Any],
        retriever: CandidateRetriever,
        reranker: Reranker,
        top_k: int,
        reranking_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.rep_cfg = representation_cfg
        self.retrieval = retriever
        self.reranker = reranker
        self.top_k = int(top_k)
        self.reranking_cfg = reranking_cfg or {}

    def load(self) -> None:
        self.retrieval.load()

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "embedding_model": self.retrieval.model_name,
            "top_k": self.top_k,
            "representation": dict(self.rep_cfg),
            "retriever": self.retrieval.describe() if hasattr(self.retrieval, "describe") else {
                "embedding_model": self.retrieval.model_name,
            },
            "reranker": self.reranker.describe(),
        }

    # Exposed separately so tests can check both matchers see identical inputs.
    def profiles(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
        return profile_table(source_df, self.rep_cfg), profile_table(target_df, self.rep_cfg)

    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Match]:
        stats = RuntimeStats()
        self.last_runtime = stats
        ctx = self.case_context
        with GpuMemoryProbe() as probe:
            with timed(stats, "representation_seconds"):
                if hasattr(self.retrieval, "profile_tables"):
                    src_profiles, tgt_profiles = self.retrieval.profile_tables(source_df, target_df)
                else:
                    src_profiles, tgt_profiles = self.profiles(source_df, target_df)

            if hasattr(self.retrieval, "rank_dataframes"):
                full = self.retrieval.rank_dataframes(
                    source_df, target_df, self.top_k,
                    dataset=ctx.dataset if ctx else None, case_id=ctx.case_id if ctx else None,
                )
            else:
                full = self.retrieval.rank_all(
                    src_profiles, tgt_profiles, self.top_k,
                    dataset=ctx.dataset if ctx else None, case_id=ctx.case_id if ctx else None,
                )
            stats.retrieval_seconds = self.retrieval.last_compute_seconds
            stats.retrieval_cache_hit = self.retrieval.last_cache_hit

            tgt_by_name = {p.name: p for p in tgt_profiles}
            position = {p.name: i for i, p in enumerate(tgt_profiles)}
            order = self.reranking_cfg.get("candidate_order", "target_schema")
            prefix = self.reranking_cfg.get("candidate_id_prefix", "c")

            rerank_scores: dict[str, dict[str, float]] = {}
            rerank_details: dict[str, dict[str, dict[str, Any]]] = {}
            rerank_counts: dict[str, int] = {}
            with timed(stats, "reranking_seconds"):
                for sp in src_profiles:
                    if hasattr(self.retrieval, "rerank_count"):
                        k = self.retrieval.rerank_count(sp.name, len(tgt_profiles), self.top_k)
                    else:
                        k = min(self.top_k, len(tgt_profiles))
                    rerank_counts[sp.name] = k
                    if k == 0:
                        rerank_scores[sp.name] = {}
                        rerank_details[sp.name] = {}
                        continue
                    presented = present_candidates(full[sp.name][:k], tgt_by_name, position, order, prefix)
                    scores = self.reranker.score(
                        sp, [(cid, prof) for cid, _, prof in presented], stats, debug=self.debug_sink
                    )
                    rerank_scores[sp.name] = {col: scores[cid] for cid, col, _ in presented}
                    details = getattr(self.reranker, "last_score_details", {})
                    rerank_details[sp.name] = {
                        col: dict(details.get(cid, {})) for cid, col, _ in presented
                    }

            matches: list[Match] = []
            with timed(stats, "ranking_seconds"):
                for sp in src_profiles:
                    matches.extend(build_final_ranking(
                        full[sp.name], rerank_scores[sp.name], rerank_counts[sp.name],
                        rerank_details[sp.name],
                    ))
        stats.peak_gpu_memory_mb = probe.peak_mb()
        stats.finalize()
        return matches
