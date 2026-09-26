"""Decision-model reranking (DeMa).

Each candidate receives an *independent binary* judgment
``p_i = P(T_i matches S)`` — one ``noul`` (yes/no) question per candidate, not a
single multiclass choice. The backend abstraction keeps DeMa independent of a
specific provider/checkpoint; the default ``system_one`` backend speaks the
System One HTTP protocol (Jev / Open-Jev ``POST /v1/systemone``)::

    request : {"state": <text>, "questions": {id: {"type": "noul", ...}}}  (+ optional "model")
    response: {"answers": {id: {"type": "noul", "noul": P(true)}}, "usage": {...}}

The decision input contains only column name, type and sampled values — the
same information given to Qwen — and never retrieval scores/ranks, ground truth
or dataset hints.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Protocol, Sequence

import requests

from ..data.types import ColumnProfile
from ..metrics.runtime import RuntimeStats
from .contracts import (
    RerankerFailure,
    RerankerOutputError,
    load_json_strict,
    validate_probability_map,
)
from .prompting import column_fields, describe_column


class DecisionBackend(Protocol):
    def predict(self, state: dict, questions: dict) -> dict:
        """Return the raw backend response (must contain ``answers``)."""
        ...

    def describe(self) -> dict[str, Any]:
        ...


def render_state(state: dict[str, Any]) -> str:
    """Plain-text rendering of the structured decision state."""
    lines = ["Source column:", _render_fields(state["source_column"])]
    candidates = state.get("candidates")
    if candidates:
        lines += ["", "Candidate target columns:"]
        for cid, fields in candidates.items():
            lines.append(f"[{cid}]")
            lines.append(_render_fields(fields))
    return "\n".join(lines)


def _render_fields(fields: dict[str, Any]) -> str:
    out = [f"name: {fields['name']}"]
    if "type" in fields:
        out.append(f"type: {fields['type']}")
    out.append("values: " + " | ".join(fields["values"]))
    return "\n".join(out)


class SystemOneBackend:
    """HTTP client for a System One decision server (Jev-compatible)."""

    def __init__(self, cfg: dict[str, Any], session: requests.Session | None = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.url = cfg["base_url"].rstrip("/") + cfg.get("endpoint", "/v1/systemone")
        self.timeout = float(cfg.get("timeout", 300))
        self.last_raw_text: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"backend": "system_one", "model": self.cfg["model"], "base_url": self.cfg["base_url"],
                "endpoint": self.cfg.get("endpoint", "/v1/systemone")}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.cfg.get("api_key_env") or "", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def payload(self, state: dict, questions: dict) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": render_state(state), "questions": questions}
        # Servers reject a "model" that differs from the loaded one; send it only if configured.
        if self.cfg.get("request_model"):
            payload["model"] = self.cfg["request_model"]
        return payload

    def predict(self, state: dict, questions: dict) -> dict:
        resp = self.session.post(
            self.url, data=json.dumps(self.payload(state, questions), allow_nan=False),
            headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        self.last_raw_text = resp.text
        # Strict parsing: duplicate keys / NaN are contract violations.
        return load_json_strict(resp.text)


BACKENDS: dict[str, Callable[[dict[str, Any]], DecisionBackend]] = {
    "system_one": SystemOneBackend,
}


def make_backend(cfg: dict[str, Any]) -> DecisionBackend:
    name = cfg.get("backend", "system_one")
    if name not in BACKENDS:
        raise ValueError(f"unknown decision backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name](cfg)


def parse_decision_response(response: Any, candidate_ids: Sequence[str]) -> dict[str, float]:
    """Exactly one probability in [0,1] per candidate id."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise RerankerOutputError("decision response has no 'answers' object")
    probs: dict[str, Any] = {}
    for qid, answer in response["answers"].items():
        if isinstance(answer, dict):
            if answer.get("type", "noul") != "noul":
                raise RerankerOutputError(f"answer {qid} has type {answer.get('type')!r}, expected 'noul'")
            if "noul" not in answer:
                raise RerankerOutputError(f"answer {qid} has no 'noul' probability")
            probs[qid] = answer["noul"]
        else:
            probs[qid] = answer
    return validate_probability_map(probs, candidate_ids, what="probability")


class DecisionReranker:
    def __init__(self, cfg: dict[str, Any], include_dtype: bool = True, backend: DecisionBackend | None = None):
        self.cfg = cfg
        self.include_dtype = include_dtype
        self.backend = backend or make_backend(cfg)
        self.max_retries = int(cfg.get("max_retries", 3))
        self.context = cfg.get("candidate_context", "shared")
        if self.context not in ("shared", "single"):
            raise ValueError("decision.candidate_context must be 'shared' or 'single'")
        self.prompt = cfg["prompt"]

    def describe(self) -> dict[str, Any]:
        return {**self.backend.describe(), "candidate_context": self.context, "max_retries": self.max_retries}

    def build_request(
        self, source: ColumnProfile, candidates: Sequence[tuple[str, ColumnProfile]]
    ) -> tuple[dict, dict]:
        state: dict[str, Any] = {"source_column": column_fields(source, self.include_dtype)}
        questions: dict[str, dict[str, Any]] = {}
        criteria = self.prompt.get("criteria")
        if self.context == "shared":
            state["candidates"] = {cid: column_fields(p, self.include_dtype) for cid, p in candidates}
        for cid, profile in candidates:
            if self.context == "shared":
                text = self.prompt["question"].format(cid=cid)
            else:
                text = self.prompt["question_single"].format(
                    cid=cid, candidate=describe_column(profile, self.include_dtype)
                )
            question: dict[str, Any] = {"type": "noul", "instructions": text}
            if criteria:
                question["criteria"] = dict(criteria)
            questions[cid] = question
        return state, questions

    def score(
        self,
        source: ColumnProfile,
        candidates: Sequence[tuple[str, ColumnProfile]],
        stats: RuntimeStats,
        debug: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, float]:
        ids = [cid for cid, _ in candidates]
        state, questions = self.build_request(source, candidates)
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            if attempt:
                stats.retries += 1
            stats.model_requests += 1
            record: dict[str, Any] = {"model": "decision", "attempt": attempt,
                                      "request": {"state": state, "questions": questions}}
            try:
                response = self.backend.predict(state, questions)
                record["response"] = response
                usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
                stats.input_tokens += int(usage.get("input_tokens") or 0)
                stats.output_tokens += int(usage.get("output_tokens") or 0)
                probs = parse_decision_response(response, ids)
                if debug:
                    debug(record)
                return probs
            except (requests.RequestException, RerankerOutputError, KeyError, TypeError, ValueError) as exc:
                stats.failures += 1
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                record["error"] = errors[-1]
                if debug:
                    debug(record)
                if attempt < self.max_retries and isinstance(exc, requests.RequestException):
                    time.sleep(min(2.0 * (attempt + 1), 10.0))
        raise RerankerFailure(
            f"decision reranking failed for source column {source.name!r} after "
            f"{self.max_retries + 1} attempt(s): " + " | ".join(errors)
        )
