"""Per-case runtime accounting.

Definitions (documented in README):

* ``representation_seconds`` – column profiling of source + target.
* ``retrieval_seconds`` – embedding + similarity search. On a candidate-cache
  hit this is the compute time recorded when the entry was created, so cached
  and uncached runs report comparable numbers; ``retrieval_cache_hit`` flags it.
* ``reranking_seconds`` – all reranker requests, including retries.
* ``matching_seconds`` – the core algorithm of a traditional baseline (for
  ISResMat this includes its per-case in-situ training).
* ``ranking_seconds`` – final ranking construction.
* ``total_seconds`` – sum of the components above. One-time model/server start-up
  is never included.
* ``failures`` – failed model attempts (invalid output, timeout, connection error),
  including attempts that were retried successfully.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass


@dataclass
class RuntimeStats:
    representation_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    reranking_seconds: float = 0.0
    matching_seconds: float = 0.0
    ranking_seconds: float = 0.0
    total_seconds: float = 0.0
    model_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    failures: int = 0
    gate_evaluations: int = 0
    gate_activations: int = 0
    jina_requests: int = 0
    retrieval_cache_hit: bool = False
    peak_gpu_memory_mb: float | None = None

    def finalize(self) -> "RuntimeStats":
        self.total_seconds = (
            self.representation_seconds
            + self.retrieval_seconds
            + self.reranking_seconds
            + self.matching_seconds
            + self.ranking_seconds
        )
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def timed(stats: RuntimeStats, field: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(stats, field, getattr(stats, field) + time.perf_counter() - start)


class GpuMemoryProbe:
    """Peak CUDA memory allocated by *this* process during a block."""

    def __init__(self) -> None:
        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
        except ImportError:  # pragma: no cover
            self._torch = None

    def __enter__(self) -> "GpuMemoryProbe":
        # Never create a CUDA context for CPU-only methods.
        if self._torch is not None and self._torch.cuda.is_initialized():
            self._torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        return None

    def peak_mb(self) -> float | None:
        if self._torch is None or not self._torch.cuda.is_initialized():
            return None
        return float(self._torch.cuda.max_memory_allocated()) / (1024 * 1024)
