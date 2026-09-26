"""Matcher interface shared by every method (baselines included)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ..data.types import Match
from ..metrics.runtime import RuntimeStats

DebugSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CaseContext:
    """Identifies the case being matched (used for cache keys and logs only).

    It never carries ground truth.
    """

    dataset: str
    case_id: str


class BaseMatcher(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.last_runtime = RuntimeStats()
        self.case_context: CaseContext | None = None
        self.debug_sink: DebugSink | None = None

    # ------------------------------------------------------------ lifecycle
    def load(self) -> None:
        """Load models once per process. Never timed as per-case inference."""

    def set_case_context(self, context: CaseContext | None) -> None:
        self.case_context = context

    def describe(self) -> dict[str, Any]:
        """Model/checkpoint information for logs and run manifests."""
        return {"method": self.name}

    # ------------------------------------------------------------- matching
    @abstractmethod
    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Match]:
        """Return a complete ranking: every target column once per source column."""
