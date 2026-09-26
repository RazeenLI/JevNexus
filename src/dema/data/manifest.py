"""Benchmark manifest: one JSONL row per matching case."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..utils.config import REPO_ROOT
from ..utils.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class CaseRecord:
    dataset: str
    case_id: str
    source_path: str        # repository-relative
    target_path: str
    ground_truth_path: str
    metadata_path: str
    n_source_columns: int
    n_target_columns: int
    n_ground_truth: int

    def abs(self, key: str, root: Path = REPO_ROOT) -> Path:
        value = Path(getattr(self, key))
        return value if value.is_absolute() else root / value


def write_manifest(path: Path, records: Iterable[CaseRecord]) -> None:
    rows = sorted((asdict(r) for r in records), key=lambda r: (r["dataset"], r["case_id"]))
    write_jsonl(path, rows)


def read_manifest(path: Path) -> list[CaseRecord]:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"manifest not found: {path}. Run scripts/prepare_data.sh first."
        )
    return [CaseRecord(**row) for row in read_jsonl(path)]


def spread_per_dataset(records: list[CaseRecord], n: int | None) -> list[CaseRecord]:
    """At most ``n`` cases per dataset, evenly spaced over the sorted case list
    (covers all relatedness types, deterministic). ``None`` keeps everything."""
    if not n:
        return records
    by: dict[str, list[CaseRecord]] = {}
    for r in records:
        by.setdefault(r.dataset, []).append(r)
    out = []
    for rows in by.values():
        rows = sorted(rows, key=lambda r: r.case_id)
        if len(rows) <= n:
            out += rows
        else:
            out += [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)] if n > 1 else rows[:1]
    return out


def select_cases(
    records: list[CaseRecord],
    datasets: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
) -> list[CaseRecord]:
    dataset_set = set(datasets) if datasets else None
    case_set = set(case_ids) if case_ids else None
    known = {r.dataset for r in records}
    if dataset_set:
        unknown = dataset_set - known
        if unknown:
            raise ValueError(f"unknown dataset(s) {sorted(unknown)}; manifest has {sorted(known)}")
    selected = [
        r
        for r in records
        if (dataset_set is None or r.dataset in dataset_set)
        and (case_set is None or r.case_id in case_set)
    ]
    if case_set:
        missing = case_set - {r.case_id for r in selected}
        if missing:
            raise ValueError(f"case id(s) not found: {sorted(missing)}")
    return selected
