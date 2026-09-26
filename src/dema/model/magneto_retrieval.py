"""Magneto's upstream candidate generator exposed to DeMa rerankers.

The baseline keeps running the vendored Magneto pipeline unchanged.  This
adapter exists for controlled reranker comparisons: ``dema_shared`` and
DeMa and its controlled ablations receive the same cleaned data, MPNet serialization, sampling,
exact-name matches, threshold and per-dataset encoding used by Magneto before
its Qwen reranker.
"""

from __future__ import annotations

import hashlib
import sys
import time
from typing import Any

import pandas as pd

from ..data.types import Candidate, ColumnProfile
from ..utils.config import REPO_ROOT
from ..utils.io import atomic_write_json, stable_hash
from .retrieval import CandidateCache

VENDOR_DIR = REPO_ROOT / "vendor" / "magneto"


def _import_upstream():
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    from magneto.embedding_matcher import EmbeddingMatcher
    from magneto.magneto import Magneto
    from magneto.utils.utils import clean_df, get_samples, remove_invalid_characters

    return Magneto, EmbeddingMatcher, clean_df, get_samples, remove_invalid_characters


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash every value that can affect Magneto's cleaning or sampling."""
    digest = hashlib.sha256()
    digest.update(stable_hash([str(c) for c in frame.columns], length=64).encode())
    digest.update(stable_hash([str(d) for d in frame.dtypes], length=64).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()[:24]


class MagnetoCandidateRetriever:
    """Persistent, cacheable form of Magneto's pre-reranking candidate stage."""

    def __init__(self, magneto_cfg: dict[str, Any], cache: CandidateCache | None = None):
        self.cfg = magneto_cfg
        self.cache = cache
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self._magneto_cls = None
        self._embedding_cls = None
        self._embedding = None
        self._clean_df = None
        self._get_samples = None
        self._remove_invalid_characters = None
        self.last_cache_hit = False
        self.last_compute_seconds = 0.0
        self._candidate_counts: dict[str, int] = {}

    def params_for(self, dataset: str | None) -> dict[str, Any]:
        self._ensure_imports()
        params = dict(self._magneto_cls.DEFAULT_PARAMS)
        params.update(self.cfg.get("params") or {})
        params.update((self.cfg.get("params_by_dataset") or {}).get(dataset or "", {}) or {})
        return params

    def _ensure_imports(self) -> None:
        if self._magneto_cls is not None:
            return
        (self._magneto_cls, self._embedding_cls, self._clean_df,
         self._get_samples, self._remove_invalid_characters) = _import_upstream()

    def load(self) -> None:
        self._ensure_imports()
        if self._embedding is None:
            self._embedding = self._embedding_cls(self.params_for(None))

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "vendored Magneto candidate generator",
            "embedding_model": self.model_name,
            "config": self.cfg,
        }

    def profile_tables(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> tuple[list[ColumnProfile], list[ColumnProfile]]:
        """Use the exact name/value fields presented by Magneto to Qwen.

        Magneto's LLM prompt does not contain a dtype.  The empty dtype is kept
        only to satisfy the shared ``ColumnProfile`` type; DeMa disables dtype
        rendering for these controlled variants.
        """
        self._ensure_imports()

        def profiles(frame: pd.DataFrame) -> list[ColumnProfile]:
            return [
                ColumnProfile(str(col), "", tuple(self._get_samples(frame[col], 10)))
                for col in frame.columns
            ]

        return profiles(source_df), profiles(target_df)

    def _cache_meta(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame, top_k: int,
        dataset: str | None, case_id: str | None,
    ) -> dict[str, Any]:
        params = self.params_for(dataset)
        relevant = {
            key: params[key] for key in (
                "embedding_model", "encoding_mode", "sampling_mode", "sampling_size",
                "embedding_threshold", "include_embedding_matches", "include_equal_matches",
                "topk",
            ) if key in params
        }
        return {
            "provider": "magneto_upstream_candidates_v1",
            "dataset": dataset,
            "case_id": case_id,
            "top_k": int(top_k),
            "params": relevant,
            "source_fingerprint": _frame_fingerprint(source_df),
            "target_fingerprint": _frame_fingerprint(target_df),
        }

    def rank_dataframes(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame, top_k: int,
        dataset: str | None = None, case_id: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """Return Magneto candidates first and the untouched schema-order tail."""
        meta = self._cache_meta(source_df, target_df, top_k, dataset, case_id)
        if self.cache is not None:
            entry = self.cache.load(meta)
            if entry is not None:
                self.last_cache_hit = True
                self.last_compute_seconds = float(entry["compute_seconds"])
                self._candidate_counts = {
                    str(src): int(count) for src, count in entry["candidate_counts"].items()
                }
                return {
                    src: [Candidate.from_dict(c) for c in candidates]
                    for src, candidates in entry["rankings"].items()
                }

        self.load()
        start = time.perf_counter()
        params = self.params_for(dataset)
        # The model/tokenizer are persistent; only serialization parameters vary
        # by dataset (GDC uses Magneto's benchmark-specific encoding mode).
        self._embedding.params = params
        cleaned_source = self._clean_df(source_df)
        cleaned_target = self._clean_df(target_df)
        score_map: dict[Any, dict[Any, float]] = {col: {} for col in cleaned_source.columns}

        if params.get("include_embedding_matches", True):
            pairs = self._embedding.get_embedding_similarity_candidates(cleaned_source, cleaned_target)
            for (source_col, target_col), score in pairs.items():
                score_map[source_col][target_col] = float(score)

        if params.get("include_equal_matches", True):
            source_names = {
                col: self._remove_invalid_characters(str(col).strip().lower())
                for col in cleaned_source.columns
            }
            target_names = {
                col: self._remove_invalid_characters(str(col).strip().lower())
                for col in cleaned_target.columns
            }
            for source_col, source_name in source_names.items():
                for target_col, target_name in target_names.items():
                    if source_name == target_name:
                        score_map[source_col][target_col] = 1.0

        requested_k = min(int(top_k), int(params.get("topk", top_k)), len(target_df.columns))
        target_columns = list(target_df.columns)
        rankings: dict[str, list[Candidate]] = {}
        counts: dict[str, int] = {}
        for source_col in source_df.columns:
            # Python's stable sort preserves Magneto's insertion-order tie break.
            selected = sorted(
                score_map.get(source_col, {}).items(), key=lambda item: item[1], reverse=True
            )[:requested_k]
            selected_names = {target for target, _ in selected}
            tail = [target for target in target_columns if target not in selected_names]
            ordered = selected + [(target, 0.0) for target in tail]
            source_name = str(source_col)
            counts[source_name] = len(selected)
            rankings[source_name] = [
                Candidate(source_name, str(target), float(score), rank)
                for rank, (target, score) in enumerate(ordered, start=1)
            ]

        elapsed = time.perf_counter() - start
        self.last_cache_hit = False
        self.last_compute_seconds = elapsed
        self._candidate_counts = counts
        if self.cache is not None:
            atomic_write_json(self.cache.path(meta), {
                "meta": meta,
                "compute_seconds": elapsed,
                "candidate_counts": counts,
                "rankings": {
                    source: [candidate.to_dict() for candidate in candidates]
                    for source, candidates in rankings.items()
                },
            })
        return rankings

    def rerank_count(self, source_column: str, total_targets: int, top_k: int) -> int:
        return int(self._candidate_counts.get(source_column, min(int(top_k), total_targets)))
