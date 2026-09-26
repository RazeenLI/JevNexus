"""Column profiling: name, basic data type, representative values."""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

from ...data.types import ColumnProfile
from .sampling import non_null_strings, sample_values

TYPES = ("integer", "float", "boolean", "datetime", "string", "mixed")

_INT_RE = re.compile(r"^[+-]?\d+$")
_BOOL_VALUES = {"true", "false", "yes", "no", "t", "f", "y", "n"}
# A datetime candidate must contain a separator or a month name; bare numbers
# (years, ids) are never treated as dates.
_DATE_HINT_RE = re.compile(
    r"(\d[-/.:]\d|\d{1,2}\s+[A-Za-z]{3,}|[A-Za-z]{3,}\s+\d{1,2}|\dT\d)",
)


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def infer_type(values: list[str], series_dtype: str | None = None, cfg: dict[str, Any] | None = None) -> str:
    """Deterministic type inference over normalized non-null string values."""
    cfg = cfg or {}
    min_frac = float(cfg.get("type_min_fraction", 1.0))
    date_frac = float(cfg.get("datetime_min_fraction", 0.9))
    mixed_frac = float(cfg.get("mixed_min_numeric_fraction", 0.2))
    if not values:
        return "string"
    if series_dtype == "bool":
        return "boolean"
    if series_dtype is not None and series_dtype.startswith("datetime"):
        return "datetime"
    n = len(values)
    unique = sorted(set(values))
    if sum(v.lower() in _BOOL_VALUES for v in values) / n >= min_frac:
        return "boolean"
    n_int = sum(bool(_INT_RE.match(v)) for v in values)
    n_num = sum(_is_float(v) for v in values)
    if n_int / n >= min_frac:
        return "integer"
    if n_num / n >= min_frac:
        return "float"
    hinted = [v for v in unique if _DATE_HINT_RE.search(v) and not _is_float(v)]
    if hinted and len(hinted) / len(unique) >= date_frac:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(pd.Series(hinted), errors="coerce", format="mixed")
        if parsed.notna().sum() / len(unique) >= date_frac:
            return "datetime"
    if mixed_frac <= n_num / n < 1.0:
        return "mixed"
    return "string"


def profile_column(name: Any, series: pd.Series, cfg: dict[str, Any]) -> ColumnProfile:
    values = non_null_strings(series.tolist())
    dtype = infer_type(values, str(series.dtype), cfg)
    sampled = sample_values(values, int(cfg["max_values"]), cfg.get("max_value_chars"))
    return ColumnProfile(name=str(name), dtype=dtype, values=tuple(sampled))


def profile_table(df: pd.DataFrame, cfg: dict[str, Any]) -> list[ColumnProfile]:
    """Profiles in column order. ``cfg`` is models.yaml ``representation``."""
    if cfg.get("sampling", "frequency") != "frequency":
        raise ValueError("only frequency sampling is supported")
    return [profile_column(name, df.iloc[:, i], cfg) for i, name in enumerate(df.columns)]
