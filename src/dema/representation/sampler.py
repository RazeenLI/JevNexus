"""Deterministic frequency-based value sampling.

remove nulls -> normalized string -> count unique -> sort by descending
frequency -> tie-break lexicographically -> keep top N.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import pandas as pd


def is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def normalize_value(value: Any) -> str:
    """Canonical string form of a non-null cell value."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) or (hasattr(value, "dtype") and getattr(value.dtype, "kind", "") == "f"):
        value = float(value)
        if math.isfinite(value) and value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return repr(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return " ".join(str(value).split())


def non_null_strings(values: Iterable[Any]) -> list[str]:
    return [normalize_value(v) for v in values if not is_null(v)]


def sample_values(values: Iterable[Any], max_values: int, max_chars: int | None = None) -> list[str]:
    counts = Counter(non_null_strings(values))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    out: list[str] = []
    seen: set[str] = set()
    for value, _ in ordered:
        if max_chars is not None and len(value) > max_chars:
            value = value[:max_chars]
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max_values:
            break
    return out
