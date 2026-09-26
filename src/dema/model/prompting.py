"""Shared, model-agnostic presentation of source/candidate columns.

Both rerankers use these helpers, so the generative (Qwen) and decision (DeMa)
models receive exactly the same semantic information: column name, type and
sampled values. Retrieval scores/ranks and ground truth are never included.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..data.types import Candidate, ColumnProfile


def column_fields(profile: ColumnProfile, include_dtype: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"name": profile.name}
    if include_dtype:
        out["type"] = profile.dtype
    out["values"] = list(profile.values)
    return out


def describe_column(profile: ColumnProfile, include_dtype: bool = True) -> str:
    lines = [f"name: {profile.name}"]
    if include_dtype:
        lines.append(f"type: {profile.dtype}")
    lines.append("values: " + " | ".join(profile.values))
    return "\n".join(lines)


def present_candidates(
    candidates: Sequence[Candidate],
    target_profiles: dict[str, ColumnProfile],
    target_position: dict[str, int],
    order: str = "target_schema",
    id_prefix: str = "c",
) -> list[tuple[str, str, ColumnProfile]]:
    """Return ``(candidate_id, target_column, profile)`` in presentation order."""
    if order == "target_schema":
        ordered = sorted(candidates, key=lambda c: target_position[c.target_column])
    elif order == "retrieval":
        ordered = sorted(candidates, key=lambda c: c.retrieval_rank)
    else:
        raise ValueError(f"unknown candidate_order {order!r}")
    names = [c.target_column for c in ordered]
    if len(set(names)) != len(names):
        raise ValueError("duplicate candidate target columns")
    return [
        (f"{id_prefix}{i}", c.target_column, target_profiles[c.target_column])
        for i, c in enumerate(ordered)
    ]
