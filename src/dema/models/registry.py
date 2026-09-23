"""Method name -> matcher construction (one matcher per process)."""

from __future__ import annotations

from typing import Callable

from ..utils.config import Config
from .base import BaseMatcher
from .retriever import CandidateCache, CandidateRetriever

PRIMARY_METHODS = (
    "coma", "coma_plus", "distribution", "similarity_flooding",
    "isresmat", "unicorn", "magneto_qwen", "dema",
)
OPTIONAL_METHODS = ("jaccard", "levenshtein")
ALL_METHODS = PRIMARY_METHODS + OPTIONAL_METHODS
NEURAL_RERANK_METHODS = ("magneto_qwen", "dema")


def candidate_cache(config: Config, enabled: bool = True) -> CandidateCache:
    return CandidateCache(config.outputs_dir / "cache" / "candidates", enabled=enabled)


def make_retriever(config: Config, use_cache: bool = True) -> CandidateRetriever:
    return CandidateRetriever(
        config.section("retriever"), config.section("representation"), cache=candidate_cache(config, use_cache)
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
        "dema": lambda: _dema(config, use_cache),
    }
    if method not in builders:
        raise ValueError(f"unknown method {method!r}; available: {sorted(builders)}")
    return builders[method]()


def _baseline(module: str, cls: str, config: Config, cfg_name: str | None = None) -> BaseMatcher:
    import importlib

    mod = importlib.import_module(f"dema.baselines.{module}")
    return getattr(mod, cls)(config.baseline(cfg_name or module))


def _magneto(config: Config, use_cache: bool) -> BaseMatcher:
    from .magneto_qwen import MagnetoQwenMatcher

    return MagnetoQwenMatcher(
        config.section("representation"), make_retriever(config, use_cache), config.section("qwen"),
        top_k=config.section("retriever")["top_k"], reranking_cfg=config.section("reranking"),
    )


def _dema(config: Config, use_cache: bool) -> BaseMatcher:
    from .dema import DeMaMatcher

    return DeMaMatcher(
        config.section("representation"), make_retriever(config, use_cache), config.section("decision"),
        top_k=config.section("retriever")["top_k"], reranking_cfg=config.section("reranking"),
    )
