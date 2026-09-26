"""Text serialization of a column profile (used by the embedding retriever)."""

from __future__ import annotations

from typing import Any

from ...data.types import ColumnProfile


def serialize_profile(profile: ColumnProfile, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or {}
    lines = [f"Column: {profile.name}"]
    if cfg.get("include_dtype", True):
        lines.append(f"Type: {profile.dtype}")
    lines.append("Values: " + " | ".join(profile.values))
    return "\n".join(lines)


def representation_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    """The part of the representation config that affects outputs (cache key)."""
    keys = ("version", "max_values", "sampling", "include_dtype", "max_value_chars",
            "datetime_min_fraction", "type_min_fraction", "mixed_min_numeric_fraction")
    return {k: cfg.get(k) for k in keys}
