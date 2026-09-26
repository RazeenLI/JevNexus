"""Strict response contracts shared by model clients."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence


class RerankerOutputError(ValueError):
    """The model answered, but its response violates the output contract."""


class RerankerFailure(RuntimeError):
    """All configured inference attempts failed."""


def _reject_constant(value: str):
    raise RerankerOutputError(f"non-finite number {value!r} in response")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise RerankerOutputError(f"duplicate candidate id(s) {sorted(duplicates)}")
    return dict(pairs)


def load_json_strict(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise RerankerOutputError(f"invalid JSON: {exc}") from exc


def validate_probability_map(
    values: Any, candidate_ids: Sequence[str], what: str = "score"
) -> dict[str, float]:
    """Require exactly one finite probability in ``[0, 1]`` for every id."""
    if not isinstance(values, dict):
        raise RerankerOutputError(f"expected a JSON object of {what}s")
    expected = list(candidate_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("candidate ids must be unique")
    expected_set = set(expected)
    missing = [candidate_id for candidate_id in expected if candidate_id not in values]
    extra = [key for key in values if key not in expected_set]
    if missing:
        raise RerankerOutputError(f"missing candidate id(s) {missing}")
    if extra:
        raise RerankerOutputError(f"unexpected candidate id(s) {extra}")

    result: dict[str, float] = {}
    for candidate_id in expected:
        value = values[candidate_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RerankerOutputError(f"{what} for {candidate_id} is not a number: {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise RerankerOutputError(f"{what} for {candidate_id} is not finite")
        if not 0.0 <= value <= 1.0:
            raise RerankerOutputError(f"{what} for {candidate_id} outside [0,1]: {value}")
        result[candidate_id] = value
    return result
