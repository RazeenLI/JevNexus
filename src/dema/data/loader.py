"""Load processed benchmark cases."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from ..utils.io import read_json
from .manifest import CaseRecord
from .types import BenchmarkCase


def read_table(path: Path) -> pd.DataFrame:
    # Same parsing as the Magneto/Valentine benchmark scripts (pd.read_csv defaults).
    return pd.read_csv(path, low_memory=False)


def read_ground_truth(path: Path) -> list[tuple[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["source_column", "target_column"]:
            raise ValueError(f"unexpected ground-truth header in {path}: {header}")
        return [(row[0], row[1]) for row in reader]


def load_case(record: CaseRecord) -> BenchmarkCase:
    source_df = read_table(record.abs("source_path"))
    target_df = read_table(record.abs("target_path"))
    source_df.columns = [str(c) for c in source_df.columns]
    target_df.columns = [str(c) for c in target_df.columns]
    return BenchmarkCase(
        dataset=record.dataset,
        case_id=record.case_id,
        source_df=source_df,
        target_df=target_df,
        ground_truth=read_ground_truth(record.abs("ground_truth_path")),
        metadata=read_json(record.abs("metadata_path")),
    )
