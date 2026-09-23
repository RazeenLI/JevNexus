"""Magneto-style baseline: shared retrieval + Qwen3.5-9B generative reranking."""

from __future__ import annotations

from .generative_reranker import QwenReranker
from .retriever import CandidateRetriever
from .two_stage import RetrieveRerankMatcher


class MagnetoQwenMatcher(RetrieveRerankMatcher):
    """Independent re-implementation of the Magneto retrieve-then-LLM-rerank design.

    Differences to the original Magneto: a zero-shot (not fine-tuned) retriever,
    DeMa's own column serialization, and Qwen3.5-9B instead of GPT-4o-mini.
    """

    name = "magneto_qwen"

    def __init__(self, representation_cfg, retriever: CandidateRetriever, qwen_cfg, top_k, reranking_cfg=None,
                 reranker: QwenReranker | None = None):
        reranker = reranker or QwenReranker(qwen_cfg, include_dtype=bool(representation_cfg.get("include_dtype", True)))
        super().__init__(representation_cfg, retriever, reranker, top_k, reranking_cfg)
