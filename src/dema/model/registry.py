"""Method name -> matcher construction (one matcher per process)."""

from __future__ import annotations

from typing import Callable

from ..utils.config import Config
from .base import BaseMatcher
from .retrieval import CandidateCache, CandidateRetriever

PRIMARY_METHODS = (
    "coma", "coma_plus", "distribution", "similarity_flooding",
    "isresmat", "unicorn", "magneto_qwen",
    "dema", "dema_no_rerank", "dema_no_struct", "dema_decision", "dema_shared",
)
OPTIONAL_METHODS = (
    "dema_own_retrieval", "dema_jev_weight", "dema_legacy",
    # Backward-compatible aliases for result directories and old commands.
    "dema_fusion", "dema_jina_rerank", "dema_jina_no_coma", "dema_single",
    "jaccard", "levenshtein",
)
ALL_METHODS = PRIMARY_METHODS + OPTIONAL_METHODS
DEMA_METHODS = (
    "dema", "dema_no_rerank", "dema_no_struct", "dema_decision", "dema_shared",
    "dema_own_retrieval", "dema_jev_weight", "dema_legacy",
    "dema_fusion", "dema_jina_rerank", "dema_jina_no_coma", "dema_single",
)
CACHED_RETRIEVAL_METHODS = DEMA_METHODS


def candidate_cache(config: Config, enabled: bool = True) -> CandidateCache:
    return CandidateCache(config.cache_dir / "candidates", enabled=enabled)


def magneto_candidate_cache(config: Config, enabled: bool = True) -> CandidateCache:
    return CandidateCache(config.cache_dir / "candidates_magneto", enabled=enabled)


def make_retriever(config: Config, use_cache: bool = True) -> CandidateRetriever:
    return CandidateRetriever(
        config.section("retriever"), config.section("representation"), cache=candidate_cache(config, use_cache)
    )


def make_magneto_retriever(config: Config, use_cache: bool = True):
    from .magneto_retrieval import MagnetoCandidateRetriever

    return MagnetoCandidateRetriever(
        config.section("magneto"), cache=magneto_candidate_cache(config, use_cache)
    )


def build_matcher(method: str, config: Config, use_cache: bool = True) -> BaseMatcher:
    builders: dict[str, Callable[[], BaseMatcher]] = {
        "coma": lambda: _baseline("coma", "COMAMatcher", config),
        "coma_plus": lambda: _baseline("coma_plus", "COMAPlusMatcher", config),
        "distribution": lambda: _baseline("distribution", "DistributionMatcher", config),
        "similarity_flooding": lambda: _baseline("similarity_flooding", "SimilarityFloodingMatcher", config),
        "isresmat": lambda: _baseline("isresmat", "ISResMatMatcher", config),
        "unicorn": lambda: _baseline("unicorn", "UnicornMatcher", config),
        "jaccard": lambda: _baseline("simple", "JaccardMatcher", config, cfg_name="jaccard"),
        "levenshtein": lambda: _baseline("simple", "LevenshteinMatcher", config, cfg_name="levenshtein"),
        "magneto_qwen": lambda: _magneto(config, use_cache),
        "dema": lambda: _dema_experimental(config, use_cache, "dema"),
        "dema_no_rerank": lambda: _dema_fusion(config, use_cache, "dema_no_rerank"),
        "dema_no_struct": lambda: _dema_experimental(config, use_cache, "dema_no_struct"),
        "dema_decision": lambda: _dema_magneto(config, use_cache, "single", "dema_decision"),
        "dema_jev_weight": lambda: _dema_experimental(config, use_cache, "dema_jev_weight"),
        "dema_jina_rerank": lambda: _dema_experimental(config, use_cache, "dema_jina_rerank"),
        "dema_jina_no_coma": lambda: _dema_experimental(config, use_cache, "dema_jina_no_coma"),
        "dema_fusion": lambda: _dema_fusion(config, use_cache, "dema_fusion"),
        "dema_shared": lambda: _dema_magneto(config, use_cache, "shared", "dema_shared"),
        "dema_single": lambda: _dema_magneto(config, use_cache, "single", "dema_single"),
        "dema_own_retrieval": lambda: _dema_own(config, use_cache, "shared", "dema_own_retrieval"),
        "dema_legacy": lambda: _dema_own(config, use_cache, "single", "dema_legacy"),
    }
    if method not in builders:
        raise ValueError(f"unknown method {method!r}; available: {sorted(builders)}")
    return builders[method]()


def _baseline(module: str, cls: str, config: Config, cfg_name: str | None = None) -> BaseMatcher:
    import importlib

    mod = importlib.import_module(f"dema.baselines.{module}")
    return getattr(mod, cls)(config.baseline(cfg_name or module))


def _magneto(config: Config, use_cache: bool) -> BaseMatcher:
    # Upstream Magneto code (vendor/magneto) with only its LLM transport swapped.
    from ..baselines.magneto_original import MagnetoOriginalMatcher

    return MagnetoOriginalMatcher(config.section("magneto"), config.section("qwen"))


def _dema_own(config: Config, use_cache: bool, context: str, name: str) -> BaseMatcher:
    from .dema import DeMaMatcher

    return DeMaMatcher(
        config.section("representation"), make_retriever(config, use_cache), config.section("decision"),
        top_k=config.section("retriever")["top_k"], reranking_cfg=config.section("reranking"),
        candidate_context=context, name=name,
    )


def _dema_magneto(config: Config, use_cache: bool, context: str, name: str) -> BaseMatcher:
    from .dema import DeMaMatcher

    reranking_cfg = dict(config.section("reranking"))
    # Magneto presents candidates in retrieval order; preserve the same signal
    # for the controlled Jev comparison.
    reranking_cfg["candidate_order"] = "retrieval"
    retriever = make_magneto_retriever(config, use_cache)
    return DeMaMatcher(
        config.section("representation"), retriever, config.section("decision"),
        top_k=config.section("retriever")["top_k"], reranking_cfg=reranking_cfg,
        candidate_context=context, include_dtype=False, name=name,
    )


def _dema_fusion(config: Config, use_cache: bool, name: str) -> BaseMatcher:
    from .fusion import DeMaFusionMatcher

    reranking_cfg = dict(config.section("reranking"))
    reranking_cfg["candidate_order"] = "retrieval"
    matcher = DeMaFusionMatcher(
        config.section("representation"),
        make_magneto_retriever(config, use_cache),
        config.section("decision"),
        config.baseline("coma_plus"),
        config.section("fusion"),
        top_k=config.section("retriever")["top_k"],
        reranking_cfg=reranking_cfg,
    )
    matcher.name = name
    return matcher


def _dema_experimental(config: Config, use_cache: bool, method: str) -> BaseMatcher:
    from .experimental_rerank import (
        DeMaJevWeightMatcher,
        DeMaJinaNoComaMatcher,
        DeMaJinaRerankMatcher,
    )

    reranking_cfg = dict(config.section("reranking"))
    reranking_cfg["candidate_order"] = "retrieval"
    common = dict(
        representation_cfg=config.section("representation"),
        retriever=make_magneto_retriever(config, use_cache),
        decision_cfg=config.section("decision"),
        coma_plus_cfg=config.baseline("coma_plus"),
        fusion_cfg=config.section("fusion"),
        top_k=config.section("retriever")["top_k"],
        reranking_cfg=reranking_cfg,
    )
    if method == "dema_jev_weight":
        return DeMaJevWeightMatcher(
            **common, dynamic_weight_cfg=config.section("dynamic_weight")
        )
    if method in ("dema_no_struct", "dema_jina_no_coma"):
        matcher = DeMaJinaNoComaMatcher(
            representation_cfg=common["representation_cfg"],
            retriever=common["retriever"],
            decision_cfg=common["decision_cfg"],
            jina_cfg=config.section("jina_rerank"),
            top_k=common["top_k"],
            reranking_cfg=common["reranking_cfg"],
        )
    else:
        matcher = DeMaJinaRerankMatcher(**common, jina_cfg=config.section("jina_rerank"))
    matcher.name = method
    return matcher
