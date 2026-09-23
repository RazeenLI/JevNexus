"""Core data types shared by every component."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BenchmarkCase:
    dataset: str
    case_id: str
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    ground_truth: list[tuple[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnProfile":
        return cls(name=data["name"], dtype=data["dtype"], values=tuple(data["values"]))


@dataclass(frozen=True)
class Candidate:
    source_column: str
    target_column: str
    retrieval_score: float
    retrieval_rank: int  # 1-based

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            source_column=data["source_column"],
            target_column=data["target_column"],
            retrieval_score=float(data["retrieval_score"]),
            retrieval_rank=int(data["retrieval_rank"]),
        )


@dataclass(frozen=True)
class Match:
    """One ranked target for one source column.

    ``score`` is the method's own score for the pair. For the retrieve-and-rerank
    matchers ``score`` is the reranker score for top-k candidates and 0.0 for the
    tail (see ``models.ranking``); the raw values are kept in ``reranker_score``
    and ``retrieval_score``. Evaluation relies on ``rank``.
    """

    source_column: str
    target_column: str
    score: float
    rank: int  # 1-based, per source column
    reranker_score: float | None = None
    retrieval_score: float | None = None
