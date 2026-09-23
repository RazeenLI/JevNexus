"""Magneto-style generative reranking with Qwen3.5-9B (OpenAI-compatible API).

One request per source column containing every retrieved candidate. The model
must return a JSON object with exactly one score in [0, 1] per candidate id.
Invalid responses are retried up to ``max_retries`` times; after the final
failure a :class:`RerankerFailure` is raised — there is never a fallback to
retrieval scores.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Callable, Sequence

import requests

from ..data.types import ColumnProfile
from ..evaluation.runtime import RuntimeStats
from .prompting import describe_column


class RerankerOutputError(ValueError):
    """The model answered, but the answer violates the output contract."""


class RerankerFailure(RuntimeError):
    """All attempts failed (invalid output, timeout, connection error, ...)."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _reject_constant(value: str):
    raise RerankerOutputError(f"non-finite number {value!r} in response")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise RerankerOutputError(f"duplicate candidate id(s) {sorted(dup)}")
    return dict(pairs)


def load_json_strict(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise RerankerOutputError(f"invalid JSON: {exc}") from exc


def extract_json_object(text: str) -> str:
    """Strip thinking blocks / code fences and return the JSON object text."""
    if text is None:
        raise RerankerOutputError("empty response")
    cleaned = _THINK_RE.sub("", text).strip()
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RerankerOutputError("no JSON object in response")
    return cleaned[start : end + 1]


def validate_probability_map(values: Any, candidate_ids: Sequence[str], what: str = "score") -> dict[str, float]:
    if not isinstance(values, dict):
        raise RerankerOutputError(f"expected a JSON object of {what}s")
    expected = list(candidate_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("candidate ids must be unique")
    missing = [c for c in expected if c not in values]
    extra = [k for k in values if k not in set(expected)]
    if missing:
        raise RerankerOutputError(f"missing candidate id(s) {missing}")
    if extra:
        raise RerankerOutputError(f"unexpected candidate id(s) {extra}")
    out: dict[str, float] = {}
    for cid in expected:
        value = values[cid]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RerankerOutputError(f"{what} for {cid} is not a number: {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise RerankerOutputError(f"{what} for {cid} is not finite")
        if not 0.0 <= value <= 1.0:
            raise RerankerOutputError(f"{what} for {cid} outside [0,1]: {value}")
        out[cid] = value
    return out


def parse_score_response(text: str, candidate_ids: Sequence[str]) -> dict[str, float]:
    return validate_probability_map(load_json_strict(extract_json_object(text)), candidate_ids)


class QwenReranker:
    def __init__(self, cfg: dict[str, Any], include_dtype: bool = True, session: requests.Session | None = None):
        self.cfg = cfg
        self.include_dtype = include_dtype
        self.session = session or requests.Session()
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.model = cfg.get("served_model_name") or cfg["model"]
        self.max_retries = int(cfg.get("max_retries", 3))
        self.timeout = float(cfg.get("timeout", 300))
        self.prompt = cfg["prompt"]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.cfg["model"],
            "served_model_name": self.model,
            "base_url": self.cfg["base_url"],
            "temperature": self.cfg.get("temperature", 0),
            "enable_thinking": self.cfg.get("enable_thinking", False),
            "max_retries": self.max_retries,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.cfg.get("api_key_env") or "", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def build_messages(
        self, source: ColumnProfile, candidates: Sequence[tuple[str, ColumnProfile]]
    ) -> list[dict[str, str]]:
        ids = ", ".join(cid for cid, _ in candidates)
        blocks = [
            "Source column:",
            describe_column(source, self.include_dtype),
            "",
            "Candidate target columns:",
        ]
        for cid, profile in candidates:
            blocks.append(f"[{cid}]")
            blocks.append(describe_column(profile, self.include_dtype))
        blocks.append("")
        blocks.append(self.prompt["task"].format(ids=ids))
        return [
            {"role": "system", "content": self.prompt["system"]},
            {"role": "user", "content": "\n".join(blocks)},
        ]

    def request_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.cfg.get("temperature", 0)),
            "max_tokens": int(self.cfg.get("max_tokens", 1024)),
            "chat_template_kwargs": {"enable_thinking": bool(self.cfg.get("enable_thinking", False))},
        }

    def score(
        self,
        source: ColumnProfile,
        candidates: Sequence[tuple[str, ColumnProfile]],
        stats: RuntimeStats,
        debug: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, float]:
        ids = [cid for cid, _ in candidates]
        payload = self.request_payload(self.build_messages(source, candidates))
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            if attempt:
                stats.retries += 1
            stats.model_requests += 1
            record: dict[str, Any] = {"model": "qwen", "attempt": attempt, "request": payload}
            try:
                resp = self.session.post(self.url, json=payload, headers=self._headers(), timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                usage = body.get("usage") or {}
                stats.input_tokens += int(usage.get("prompt_tokens") or 0)
                stats.output_tokens += int(usage.get("completion_tokens") or 0)
                text = body["choices"][0]["message"].get("content")
                record["response"] = text
                scores = parse_score_response(text, ids)
                if debug:
                    debug(record)
                return scores
            except (requests.RequestException, RerankerOutputError, KeyError, IndexError, TypeError, ValueError) as exc:
                stats.failures += 1
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                record["error"] = errors[-1]
                if debug:
                    debug(record)
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * (attempt + 1), 10.0) if isinstance(exc, requests.RequestException) else 0)
        raise RerankerFailure(
            f"Qwen reranking failed for source column {source.name!r} after "
            f"{self.max_retries + 1} attempt(s): " + " | ".join(errors)
        )
