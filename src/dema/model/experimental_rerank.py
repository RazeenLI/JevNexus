"""Optional post-fusion experiments; none are part of the default experiment."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import requests

from ..data.types import ColumnProfile
from ..metrics.runtime import RuntimeStats
from .contracts import RerankerFailure, RerankerOutputError
from .decision import parse_decision_response
from .decision import DecisionReranker
from .jevnexus import JevNexusMatcher
from .fusion import JevNexusFusionMatcher, FixedScoreFusionReranker
from .prompting import column_fields, describe_column


def _candidate_evidence(
    candidates: Sequence[tuple[str, ColumnProfile]],
    jev_scores: dict[str, float],
    coma_scores: dict[str, float],
    fusion_scores: dict[str, float],
    limit: int,
) -> str:
    ordered = sorted(candidates, key=lambda item: -fusion_scores[item[0]])[:limit]
    blocks = []
    for candidate_id, profile in ordered:
        blocks.append(
            f"[{candidate_id}]\n{describe_column(profile, include_dtype=False)}\n"
            f"Jev score: {jev_scores[candidate_id]:.8f}\n"
            f"COMA+ score: {coma_scores[candidate_id]:.8f}\n"
            f"fixed fusion score: {fusion_scores[candidate_id]:.8f}"
        )
    return "\n\n".join(blocks)


class JevDynamicWeightReranker:
    """Ask Jev for one source-specific confidence, then map it to a safe weight."""

    def __init__(self, base: FixedScoreFusionReranker, cfg: dict[str, Any]):
        self.base = base
        self.min_jev_weight = float(cfg.get("min_jev_weight", 0.1))
        self.max_jev_weight = float(cfg.get("max_jev_weight", 0.7))
        self.evidence_top_n = int(cfg.get("evidence_top_n", 5))
        if not 0 <= self.min_jev_weight <= self.max_jev_weight <= 1:
            raise ValueError("dynamic Jev weight bounds must satisfy 0 <= min <= max <= 1")
        if self.evidence_top_n <= 0:
            raise ValueError("evidence_top_n must be positive")

    @property
    def context(self) -> str:
        return self.base.context

    @property
    def include_dtype(self) -> bool:
        return self.base.include_dtype

    @property
    def coma_scores(self) -> dict[tuple[str, str], float]:
        return self.base.coma_scores

    @coma_scores.setter
    def coma_scores(self, value: dict[tuple[str, str], float]) -> None:
        self.base.coma_scores = value

    def describe(self) -> dict[str, Any]:
        return {
            "type": "jev_dynamic_weight",
            "min_jev_weight": self.min_jev_weight,
            "max_jev_weight": self.max_jev_weight,
            "evidence_top_n": self.evidence_top_n,
            "base": self.base.describe(),
        }

    def _weight_probability(
        self,
        source: ColumnProfile,
        evidence: str,
        stats: RuntimeStats,
        debug: Callable[[dict[str, Any]], None] | None,
    ) -> float:
        decision = self.base.decision
        state = {"source_column": column_fields(source, include_dtype=False)}
        questions = {
            "weight": {
                "type": "noul",
                "instructions": (
                    "Review the candidate evidence below. Should the Jev semantic scores receive "
                    "more influence than the COMA+ schema-and-instance scores when ranking these "
                    "candidates? Answer true only when the semantic evidence is more reliable.\n\n"
                    + evidence
                ),
                "criteria": {
                    "true": "Jev semantic evidence is more reliable for this source column.",
                    "false": "COMA+ schema and instance evidence is at least as reliable.",
                },
            }
        }
        errors: list[str] = []
        for attempt in range(decision.max_retries + 1):
            if attempt:
                stats.retries += 1
            stats.model_requests += 1
            record: dict[str, Any] = {
                "model": "decision_weight",
                "attempt": attempt,
                "request": {"state": state, "questions": questions},
            }
            try:
                response = decision.backend.predict(state, questions)
                record["response"] = response
                usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
                stats.input_tokens += int(usage.get("input_tokens") or 0)
                stats.output_tokens += int(usage.get("output_tokens") or 0)
                probability = parse_decision_response(response, ["weight"])["weight"]
                if debug:
                    debug(record)
                return probability
            except (requests.RequestException, RerankerOutputError, KeyError, TypeError, ValueError) as exc:
                stats.failures += 1
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                record["error"] = errors[-1]
                if debug:
                    debug(record)
                if attempt < decision.max_retries and isinstance(exc, requests.RequestException):
                    time.sleep(min(2.0 * (attempt + 1), 10.0))
        raise RerankerFailure(
            f"dynamic weight judgment failed for source column {source.name!r}: " + " | ".join(errors)
        )

    def score(self, source, candidates, stats, debug=None) -> dict[str, float]:
        jev_scores = self.base.decision.score(source, candidates, stats, debug=debug)
        coma_scores = {
            candidate_id: float(self.coma_scores.get((source.name, profile.name), 0.0))
            for candidate_id, profile in candidates
        }
        fixed_scores = {
            candidate_id: self.base.jev_weight * jev_scores[candidate_id]
            + self.base.coma_plus_weight * coma_scores[candidate_id]
            for candidate_id, _ in candidates
        }
        evidence = _candidate_evidence(
            candidates, jev_scores, coma_scores, fixed_scores, self.evidence_top_n
        )
        probability = self._weight_probability(source, evidence, stats, debug)
        jev_weight = self.min_jev_weight + (
            self.max_jev_weight - self.min_jev_weight
        ) * probability
        return {
            candidate_id: jev_weight * jev_scores[candidate_id]
            + (1.0 - jev_weight) * coma_scores[candidate_id]
            for candidate_id, _ in candidates
        }


class JinaListwiseModel:
    """Lazy adapter for jinaai/jina-reranker-v3.5's native listwise API."""

    def __init__(self, cfg: dict[str, Any], model=None):
        self.cfg = dict(cfg)
        self.model = model

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModel

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.cfg.get("dtype", "auto"),
            "device_map": self.cfg.get("device_map", "auto"),
        }
        if self.cfg.get("revision"):
            kwargs["revision"] = self.cfg["revision"]
        self.model = AutoModel.from_pretrained(self.cfg["model"], **kwargs)
        self.model.eval()

    def rerank(self, query: str, documents: list[str]) -> list[dict[str, Any]]:
        self.load()
        return self.model.rerank(query, documents, top_n=len(documents))


class JinaTopReranker:
    """Use Jina's native listwise scorer to reorder only fixed-fusion Top-N."""

    def __init__(
        self,
        base: FixedScoreFusionReranker,
        cfg: dict[str, Any],
        listwise_model: JinaListwiseModel | None = None,
        gate_cfg: dict[str, Any] | None = None,
    ):
        self.base = base
        self.cfg = dict(cfg)
        self.top_n = int(cfg.get("top_n", 3))
        self.include_scores = bool(cfg.get("include_scores", True))
        self.listwise_model = listwise_model or JinaListwiseModel(cfg)
        self.gate_cfg = dict(gate_cfg) if gate_cfg is not None else None
        self.margin_threshold = (
            float(self.gate_cfg.get("margin_threshold", 0.02))
            if self.gate_cfg is not None else None
        )
        if self.top_n <= 1:
            raise ValueError("Jina top_n must be greater than one")
        if self.margin_threshold is not None and self.margin_threshold < 0:
            raise ValueError("selective-refinement margin_threshold must be non-negative")

    @property
    def context(self) -> str:
        return self.base.context

    @property
    def include_dtype(self) -> bool:
        return self.base.include_dtype

    @property
    def coma_scores(self) -> dict[tuple[str, str], float]:
        return self.base.coma_scores

    @coma_scores.setter
    def coma_scores(self, value: dict[tuple[str, str], float]) -> None:
        self.base.coma_scores = value

    def load(self) -> None:
        self.listwise_model.load()

    def describe(self) -> dict[str, Any]:
        return {
            "type": "gated_jina_listwise_top_n" if self.gate_cfg is not None else "jina_listwise_top_n",
            "model": self.cfg["model"],
            "revision": self.cfg.get("revision"),
            "top_n": self.top_n,
            "include_scores": self.include_scores,
            "gate": dict(self.gate_cfg) if self.gate_cfg is not None else None,
            "base": self.base.describe(),
        }

    def score(self, source, candidates, stats, debug=None) -> dict[str, float]:
        jev_scores = self.base.decision.score(source, candidates, stats, debug=debug)
        coma_scores = {
            candidate_id: float(self.coma_scores.get((source.name, profile.name), 0.0))
            for candidate_id, profile in candidates
        }
        fusion_scores = {
            candidate_id: self.base.jev_weight * jev_scores[candidate_id]
            + self.base.coma_plus_weight * coma_scores[candidate_id]
            for candidate_id, _ in candidates
        }
        profile_by_id = dict(candidates)
        top_ids = sorted(fusion_scores, key=lambda cid: -fusion_scores[cid])[: self.top_n]
        fusion_order = sorted(fusion_scores, key=lambda cid: -fusion_scores[cid])
        fusion_ranks = {candidate_id: rank for rank, candidate_id in enumerate(fusion_order, 1)}

        disagreement = False
        margin = None
        gate_activated = True
        if self.gate_cfg is not None:
            stats.gate_evaluations += 1
            jev_top = max(jev_scores, key=jev_scores.get)
            # COMA+ returns zero for pairs it did not select. If every score is
            # zero it has expressed no preference, so disagreement is undefined
            # and refinement is conservatively bypassed.
            if any(score > 0.0 for score in coma_scores.values()):
                coma_top = max(coma_scores, key=coma_scores.get)
                disagreement = jev_top != coma_top
            if len(fusion_order) >= 2:
                margin = fusion_scores[fusion_order[0]] - fusion_scores[fusion_order[1]]
            gate_activated = bool(
                disagreement
                and margin is not None
                and margin < self.margin_threshold
            )

        if not gate_activated:
            self.last_score_details = {
                candidate_id: {
                    "jev_score": float(jev_scores[candidate_id]),
                    "coma_plus_score": float(coma_scores[candidate_id]),
                    "fusion_score": float(fusion_scores[candidate_id]),
                    "fusion_rank": fusion_ranks[candidate_id],
                    "jina_applied": False,
                    "jina_score": None,
                    "jina_rank": None,
                    "gate_activated": False,
                    "gate_disagreement": disagreement,
                    "gate_margin": float(margin) if margin is not None else None,
                    "gate_threshold": self.margin_threshold,
                }
                for candidate_id, _ in candidates
            }
            if debug:
                debug({
                    "model": "selective_refinement_gate",
                    "source_column": source.name,
                    "activated": False,
                    "disagreement": disagreement,
                    "margin": margin,
                    "threshold": self.margin_threshold,
                })
            return fusion_scores

        if self.gate_cfg is not None:
            stats.gate_activations += 1
        query = describe_column(source, include_dtype=False)
        documents = []
        for candidate_id in top_ids:
            document = describe_column(profile_by_id[candidate_id], include_dtype=False)
            if self.include_scores:
                document += (
                    f"\nJev score: {jev_scores[candidate_id]:.8f}"
                    f"\nCOMA+ score: {coma_scores[candidate_id]:.8f}"
                    f"\nfixed fusion score: {fusion_scores[candidate_id]:.8f}"
                )
            documents.append(document)
        stats.model_requests += 1
        stats.jina_requests += 1
        try:
            results = self.listwise_model.rerank(query, documents)
        except Exception as exc:  # noqa: BLE001 - third-party custom model boundary
            stats.failures += 1
            raise RerankerFailure(
                f"Jina reranking failed for source column {source.name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        indices = [result.get("index") for result in results]
        if len(indices) != len(top_ids) or set(indices) != set(range(len(top_ids))):
            stats.failures += 1
            raise RerankerOutputError(
                f"Jina returned invalid indices {indices!r}; expected 0..{len(top_ids) - 1}"
            )
        ranked_ids = [top_ids[int(index)] for index in indices]
        # Preserve the fixed fusion score distribution (important for pooled
        # Recall@GT) and only permute those scores according to Jina's order.
        score_slots = sorted((fusion_scores[cid] for cid in top_ids), reverse=True)
        output = dict(fusion_scores)
        for candidate_id, score_value in zip(ranked_ids, score_slots):
            output[candidate_id] = score_value
        result_by_id = {
            top_ids[int(result["index"])]: result for result in results
        }
        jina_ranks = {candidate_id: rank for rank, candidate_id in enumerate(ranked_ids, 1)}
        self.last_score_details = {}
        for candidate_id, profile in candidates:
            result = result_by_id.get(candidate_id)
            raw_jina_score = None
            if result is not None:
                raw_jina_score = result.get("relevance_score", result.get("score"))
            detail = {
                "jev_score": float(jev_scores[candidate_id]),
                "coma_plus_score": float(
                    self.coma_scores.get((source.name, profile.name), 0.0)
                ),
                "fusion_score": float(fusion_scores[candidate_id]),
                "fusion_rank": fusion_ranks[candidate_id],
                "jina_applied": candidate_id in result_by_id,
                "jina_score": float(raw_jina_score) if raw_jina_score is not None else None,
                "jina_rank": jina_ranks.get(candidate_id),
            }
            if self.gate_cfg is not None:
                detail.update({
                    "gate_activated": gate_activated,
                    "gate_disagreement": disagreement,
                    "gate_margin": float(margin) if margin is not None else None,
                    "gate_threshold": self.margin_threshold,
                })
            self.last_score_details[candidate_id] = detail
        if debug:
            debug({
                "model": "jina_reranker",
                "request": {"query": query, "documents": documents, "candidate_ids": top_ids},
                "response": results,
            })
        return output


class JevJinaTopReranker:
    """Ablation: Jev scores followed by Jina Top-N, with no COMA++ signal."""

    def __init__(
        self,
        decision,
        cfg: dict[str, Any],
        listwise_model: JinaListwiseModel | None = None,
    ):
        self.decision = decision
        self.cfg = dict(cfg)
        self.top_n = int(cfg.get("top_n", 3))
        self.include_scores = bool(cfg.get("include_scores", True))
        self.listwise_model = listwise_model or JinaListwiseModel(cfg)
        if self.top_n <= 1:
            raise ValueError("Jina top_n must be greater than one")

    @property
    def context(self) -> str:
        return self.decision.context

    @property
    def include_dtype(self) -> bool:
        return self.decision.include_dtype

    def load(self) -> None:
        self.listwise_model.load()

    def describe(self) -> dict[str, Any]:
        return {
            "type": "jev_then_jina_top_n_no_coma",
            "model": self.cfg["model"],
            "revision": self.cfg.get("revision"),
            "top_n": self.top_n,
            "include_scores": self.include_scores,
            "decision": self.decision.describe(),
        }

    def score(self, source, candidates, stats, debug=None) -> dict[str, float]:
        jev_scores = self.decision.score(source, candidates, stats, debug=debug)
        profile_by_id = dict(candidates)
        top_ids = sorted(jev_scores, key=lambda cid: -jev_scores[cid])[: self.top_n]
        query = describe_column(source, include_dtype=False)
        documents = []
        for candidate_id in top_ids:
            document = describe_column(profile_by_id[candidate_id], include_dtype=False)
            if self.include_scores:
                document += f"\nJev score: {jev_scores[candidate_id]:.8f}"
            documents.append(document)
        stats.model_requests += 1
        try:
            results = self.listwise_model.rerank(query, documents)
        except Exception as exc:  # noqa: BLE001 - third-party custom model boundary
            stats.failures += 1
            raise RerankerFailure(
                f"Jina reranking failed for source column {source.name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        indices = [result.get("index") for result in results]
        if len(indices) != len(top_ids) or set(indices) != set(range(len(top_ids))):
            stats.failures += 1
            raise RerankerOutputError(
                f"Jina returned invalid indices {indices!r}; expected 0..{len(top_ids) - 1}"
            )
        ranked_ids = [top_ids[int(index)] for index in indices]
        score_slots = sorted((jev_scores[cid] for cid in top_ids), reverse=True)
        output = dict(jev_scores)
        for candidate_id, score_value in zip(ranked_ids, score_slots):
            output[candidate_id] = score_value
        if debug:
            debug({
                "model": "jina_reranker_no_coma",
                "request": {"query": query, "documents": documents, "candidate_ids": top_ids},
                "response": results,
            })
        return output


class JevNexusJevWeightMatcher(JevNexusFusionMatcher):
    name = "jevnexus_jev_weight"

    def __init__(self, *args, dynamic_weight_cfg: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_reranker = JevDynamicWeightReranker(
            self.fusion_reranker, dynamic_weight_cfg
        )
        self.reranker = self.fusion_reranker
        self.name = type(self).name


class JevNexusJinaRerankMatcher(JevNexusFusionMatcher):
    name = "jevnexus_jina_rerank"

    def __init__(self, *args, jina_cfg: dict[str, Any], listwise_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_reranker = JinaTopReranker(
            self.fusion_reranker, jina_cfg, listwise_model=listwise_model
        )
        self.reranker = self.fusion_reranker
        self.name = type(self).name

    def load(self) -> None:
        super().load()
        self.fusion_reranker.load()


class JevNexusGatedJinaMatcher(JevNexusFusionMatcher):
    """Canonical JevNexus: refine only low-margin Jev/COMA+ disagreements."""

    name = "jevnexus"

    def __init__(
        self, *args, jina_cfg: dict[str, Any], gate_cfg: dict[str, Any],
        listwise_model=None, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fusion_reranker = JinaTopReranker(
            self.fusion_reranker,
            jina_cfg,
            listwise_model=listwise_model,
            gate_cfg=gate_cfg,
        )
        self.reranker = self.fusion_reranker
        self.name = type(self).name

    def load(self) -> None:
        super().load()
        self.fusion_reranker.load()


class JevNexusJinaNoComaMatcher(JevNexusMatcher):
    """Controlled ablation: Magneto candidates -> Jev-Single -> Jina Top-3."""

    name = "jevnexus_jina_no_coma"

    def __init__(
        self,
        representation_cfg,
        retriever,
        decision_cfg,
        jina_cfg,
        top_k,
        reranking_cfg=None,
        listwise_model=None,
    ):
        decision_cfg = dict(decision_cfg)
        decision_cfg["candidate_context"] = "single"
        decision = DecisionReranker(decision_cfg, include_dtype=False)
        reranker = JevJinaTopReranker(
            decision, jina_cfg, listwise_model=listwise_model
        )
        super().__init__(
            representation_cfg,
            retriever,
            decision_cfg,
            top_k,
            reranking_cfg=reranking_cfg,
            reranker=reranker,
            candidate_context="single",
            include_dtype=False,
            name=self.name,
        )

    def load(self) -> None:
        super().load()
        self.reranker.load()


# Import compatibility for historical experiment code.
DeMaJevWeightMatcher = JevNexusJevWeightMatcher
DeMaJinaRerankMatcher = JevNexusJinaRerankMatcher
DeMaGatedJinaMatcher = JevNexusGatedJinaMatcher
DeMaJinaNoComaMatcher = JevNexusJinaNoComaMatcher
