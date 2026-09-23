"""Shared zero-shot candidate retriever (sentence embeddings + cosine) and cache.

The retriever always computes the *complete* ordering of target columns for
every source column: the first ``top_k`` entries are the candidates handed to a
reranker, the rest form the ranking tail. Ordering is by descending cosine
similarity with ties broken by target column position, so it is deterministic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..data.types import Candidate, ColumnProfile
from ..representation.serializer import representation_signature, serialize_profile
from ..utils.device import resolve_device
from ..utils.io import atomic_write_json, read_json, safe_name, stable_hash

Encoder = Callable[[list[str]], np.ndarray]


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    an[an == 0] = 1.0
    bn[bn == 0] = 1.0
    return (a / an) @ (b / bn).T


def order_targets(scores: np.ndarray) -> np.ndarray:
    """Per row: target indices by descending score, ties by ascending index."""
    n_targets = scores.shape[1]
    idx = np.arange(n_targets)
    return np.stack([np.lexsort((idx, -row)) for row in scores]) if scores.size else np.zeros(
        (scores.shape[0], 0), dtype=int
    )


def profiles_fingerprint(profiles: Sequence[ColumnProfile]) -> str:
    return stable_hash([p.to_dict() for p in profiles], length=24)


class CandidateCache:
    """JSON cache of complete retrieval orderings.

    Key: dataset, case_id, embedding model, representation signature, top_k and
    fingerprints of the source/target profiles (so a stale entry is never reused).
    """

    def __init__(self, root: Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def key(self, meta: dict[str, Any]) -> str:
        return stable_hash(meta, length=24)

    def path(self, meta: dict[str, Any]) -> Path:
        return (
            self.root
            / safe_name(str(meta.get("dataset") or "_nodataset"))
            / safe_name(str(meta.get("case_id") or "_nocase"))
            / f"{self.key(meta)}.json"
        )

    def load(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.path(meta)
        if not path.is_file():
            return None
        entry = read_json(path)
        if entry.get("meta") != meta:
            return None
        return entry

    def save(self, meta: dict[str, Any], entry: dict[str, Any]) -> None:
        if self.enabled:
            atomic_write_json(self.path(meta), {"meta": meta, **entry})


class CandidateRetriever:
    def __init__(
        self,
        retriever_cfg: dict[str, Any],
        representation_cfg: dict[str, Any],
        cache: CandidateCache | None = None,
        encoder: Encoder | None = None,
    ):
        self.cfg = retriever_cfg
        self.rep_cfg = representation_cfg
        self.cache = cache
        self.model_name = retriever_cfg["model"]
        self.batch_size = int(retriever_cfg.get("batch_size", 64))
        self._encoder = encoder
        self._model = None
        self.last_cache_hit = False
        self.last_compute_seconds = 0.0

    # --------------------------------------------------------------- model
    def load(self) -> None:
        if self._encoder is not None or self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        device = resolve_device(self.cfg.get("device", "auto"))
        self._model = SentenceTransformer(self.model_name, device=device)
        self._model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._encoder is not None:
            return np.asarray(self._encoder(texts), dtype=np.float64)
        self.load()
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=bool(self.cfg.get("normalize", True)),
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float64)

    # ----------------------------------------------------------- retrieval
    def score_matrix(
        self, source_profiles: Sequence[ColumnProfile], target_profiles: Sequence[ColumnProfile]
    ) -> np.ndarray:
        src = [serialize_profile(p, self.rep_cfg) for p in source_profiles]
        tgt = [serialize_profile(p, self.rep_cfg) for p in target_profiles]
        if not src or not tgt:
            return np.zeros((len(src), len(tgt)))
        return cosine_matrix(self.embed(src), self.embed(tgt))

    def cache_meta(
        self,
        source_profiles: Sequence[ColumnProfile],
        target_profiles: Sequence[ColumnProfile],
        top_k: int,
        dataset: str | None,
        case_id: str | None,
    ) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "case_id": case_id,
            "embedding_model": self.model_name,
            "representation": representation_signature(self.rep_cfg),
            "top_k": int(top_k),
            "source_fingerprint": profiles_fingerprint(source_profiles),
            "target_fingerprint": profiles_fingerprint(target_profiles),
        }

    def rank_all(
        self,
        source_profiles: Sequence[ColumnProfile],
        target_profiles: Sequence[ColumnProfile],
        top_k: int,
        dataset: str | None = None,
        case_id: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """Complete ordering of all targets for every source column (cached)."""
        meta = self.cache_meta(source_profiles, target_profiles, top_k, dataset, case_id)
        if self.cache is not None:
            entry = self.cache.load(meta)
            if entry is not None:
                self.last_cache_hit = True
                self.last_compute_seconds = float(entry["compute_seconds"])
                return {
                    src: [Candidate.from_dict(c) for c in cands]
                    for src, cands in entry["rankings"].items()
                }
        start = time.perf_counter()
        scores = self.score_matrix(source_profiles, target_profiles)
        order = order_targets(scores)
        rankings: dict[str, list[Candidate]] = {}
        for i, sp in enumerate(source_profiles):
            rankings[sp.name] = [
                Candidate(
                    source_column=sp.name,
                    target_column=target_profiles[j].name,
                    retrieval_score=float(scores[i, j]),
                    retrieval_rank=r + 1,
                )
                for r, j in enumerate(order[i])
            ]
        elapsed = time.perf_counter() - start
        self.last_cache_hit = False
        self.last_compute_seconds = elapsed
        if self.cache is not None:
            self.cache.save(
                meta,
                {
                    "compute_seconds": elapsed,
                    "rankings": {s: [c.to_dict() for c in cs] for s, cs in rankings.items()},
                },
            )
        return rankings

    def retrieve(
        self,
        source_profiles: list[ColumnProfile],
        target_profiles: list[ColumnProfile],
        top_k: int,
        dataset: str | None = None,
        case_id: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """Top-k candidates per source column: exactly min(top_k, |targets|) unique targets."""
        full = self.rank_all(source_profiles, target_profiles, top_k, dataset, case_id)
        k = min(int(top_k), len(target_profiles))
        return {src: cands[:k] for src, cands in full.items()}
