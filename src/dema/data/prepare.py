"""Convert the raw benchmark into the unified processed format.

Output per case::

    data/processed/<dataset>/<case_id>/
        source.csv  target.csv  ground_truth.csv  metadata.json

Processing only normalizes layout, encoding (to UTF-8) and ground-truth
representation. Table files are byte copies whenever the raw file is already
valid UTF-8; otherwise the decoded text is re-written as UTF-8. Column names,
cell values, rows and ground-truth pairs are never changed. Cases whose ground
truth is empty are kept and flagged (``n_ground_truth == 0``); the evaluator
skips them, as Magneto's benchmark scripts do.

The command is idempotent and ends with integrity checks
(``verify_processed``); it exits non-zero when any check fails.

Usage::

    python -m dema.data.prepare [--datasets GDC OpenData ...] [--verify-only]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..utils.config import REPO_ROOT, Config, load_config
from ..utils.io import atomic_write_json, read_json, sha256_file
from ..utils.logging import get_console_logger
from .manifest import CaseRecord, read_manifest, write_manifest

log = get_console_logger("dema.prepare")

csv.field_size_limit(sys.maxsize)

ALL_DATASETS = ("GDC", "ChEMBL", "Magellan", "OpenData", "TPC-DI", "WikiData")


@dataclass
class RawCase:
    dataset: str
    case_id: str
    source_file: Path
    target_file: Path
    ground_truth: list[tuple[str, str]]
    ground_truth_file: Path
    source_table: str
    target_table: str
    category: str | None = None


class IntegrityError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def _slug(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def read_text(path: Path) -> tuple[str, str]:
    """Decode a raw file; returns (text, encoding)."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise IntegrityError(f"cannot decode {path}")  # pragma: no cover (latin-1 never fails)


def csv_shape(text: str) -> tuple[list[str], int]:
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        return [], 0
    n_rows = sum(1 for _ in reader)
    return header, n_rows


def copy_table(src: Path, dst: Path) -> str:
    """Copy a CSV, normalizing only the encoding. Returns the raw encoding."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_bytes()
    try:
        raw.decode("utf-8")
        shutil.copyfile(src, dst)
        return "utf-8"
    except UnicodeDecodeError:
        text, encoding = read_text(src)
        dst.write_text(text, encoding="utf-8", newline="")
        return encoding


def write_ground_truth(path: Path, pairs: list[tuple[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["source_column", "target_column"])
    writer.writerows(pairs)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def read_ground_truth(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["source_column", "target_column"]:
            raise IntegrityError(f"unexpected ground-truth header in {path}: {header}")
        return [(row[0], row[1]) for row in reader]


# -------------------------------------------------------------- discovery
def discover_gdc(config: Config) -> list[RawCase]:
    layout = config.paths["raw_layout"]
    root = config.raw_dir / layout["gdc_root"]
    target = root / layout["gdc_target"]
    gt_dir = root / "ground-truth"
    src_dir = root / "source-tables"
    if not gt_dir.is_dir() or not target.is_file():
        raise FileNotFoundError(f"raw GDC benchmark missing under {root}; run scripts/download_data.sh")
    cases = []
    for gt_file in sorted(gt_dir.glob("*.csv")):
        text, _ = read_text(gt_file)
        rows = list(csv.reader(io.StringIO(text, newline="")))
        header, body = rows[0], rows[1:]
        if len(header) != 2:
            raise IntegrityError(f"unexpected GDC ground-truth header {header} in {gt_file}")
        # Magneto drops GT rows with a missing value (gt_df.dropna()); none exist in
        # GDC-SM, and we fail loudly rather than silently dropping anything.
        if any(len(r) != 2 or not r[0] or not r[1] for r in body):
            raise IntegrityError(f"incomplete ground-truth row in {gt_file}")
        source_file = src_dir / gt_file.name
        if not source_file.is_file():
            raise FileNotFoundError(f"GDC source table missing: {source_file}")
        cases.append(
            RawCase(
                dataset="GDC",
                case_id=f"gdc_{_slug(gt_file.stem)}",
                source_file=source_file,
                target_file=target,
                ground_truth=[(r[0], r[1]) for r in body],
                ground_truth_file=gt_file,
                source_table=gt_file.stem,
                target_table=target.stem,
            )
        )
    return cases


def discover_valentine(config: Config, dataset: str) -> list[RawCase]:
    layout = config.paths["raw_layout"]
    root = config.raw_dir / layout["valentine_root"] / layout["valentine_datasets"][dataset]
    if not root.is_dir():
        raise FileNotFoundError(f"raw Valentine dataset missing: {root}; run scripts/download_data.sh")
    cases = []
    for mapping in sorted(root.rglob("*_mapping.json")):
        stem = mapping.name[: -len("_mapping.json")]
        source_file = mapping.with_name(f"{stem}_source.csv")
        target_file = mapping.with_name(f"{stem}_target.csv")
        if not source_file.is_file() or not target_file.is_file():
            raise FileNotFoundError(f"incomplete Valentine case: {mapping.parent}")
        text, _ = read_text(mapping)
        matches = json.loads(text)["matches"]
        rel_parts = mapping.parent.relative_to(root).parts
        category = rel_parts[0] if len(rel_parts) > 1 else None
        tables = {(m["source_table"], m["target_table"]) for m in matches}
        source_table, target_table = (
            next(iter(tables)) if len(tables) == 1 else (source_file.stem, target_file.stem)
        )
        # Drop directory levels repeated in the case folder name
        # (e.g. Wikidata/Musicians/Musicians_joinable -> wikidata_musicians_joinable).
        parts = [p for i, p in enumerate(rel_parts)
                 if i == len(rel_parts) - 1 or not rel_parts[-1].lower().startswith(p.lower())]
        case_id = _slug(f"{dataset}_{'_'.join(parts)}")
        cases.append(
            RawCase(
                dataset=dataset,
                case_id=case_id,
                source_file=source_file,
                target_file=target_file,
                ground_truth=[(m["source_column"], m["target_column"]) for m in matches],
                ground_truth_file=mapping,
                source_table=source_table,
                target_table=target_table,
                category=category,
            )
        )
    return cases


def discover(config: Config, datasets: Iterable[str]) -> list[RawCase]:
    cases: list[RawCase] = []
    for dataset in datasets:
        if dataset == "GDC":
            cases.extend(discover_gdc(config))
        elif dataset in config.paths["raw_layout"]["valentine_datasets"]:
            cases.extend(discover_valentine(config, dataset))
        else:
            raise ValueError(f"unknown dataset {dataset}")
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise IntegrityError("case ids are not unique")
    return cases


# ---------------------------------------------------------------- convert
def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def known_defects(config: Config) -> set[tuple[str, str, str]]:
    return {
        (d["case_id"], d["source_column"], d["target_column"])
        for d in config.paths.get("known_gt_defects", []) or []
    }


def invalid_gt_pairs(
    pairs: list[tuple[str, str]], source_cols: list[str], target_cols: list[str]
) -> list[tuple[str, str]]:
    s, t = set(source_cols), set(target_cols)
    return [(a, b) for a, b in pairs if a not in s or b not in t]


def convert_case(config: Config, case: RawCase) -> CaseRecord:
    out_dir = config.processed_dir / case.dataset / case.case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    src_enc = copy_table(case.source_file, out_dir / "source.csv")
    tgt_enc = copy_table(case.target_file, out_dir / "target.csv")
    write_ground_truth(out_dir / "ground_truth.csv", case.ground_truth)

    src_header, src_rows = csv_shape(read_text(out_dir / "source.csv")[0])
    tgt_header, tgt_rows = csv_shape(read_text(out_dir / "target.csv")[0])
    invalid = invalid_gt_pairs(case.ground_truth, src_header, tgt_header)
    metadata = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "source_table": case.source_table,
        "target_table": case.target_table,
        "category": case.category,
        "raw_source": _rel(case.source_file),
        "raw_target": _rel(case.target_file),
        "raw_ground_truth": _rel(case.ground_truth_file),
        "raw_source_sha256": sha256_file(case.source_file),
        "raw_target_sha256": sha256_file(case.target_file),
        "raw_ground_truth_sha256": sha256_file(case.ground_truth_file),
        "raw_source_encoding": src_enc,
        "raw_target_encoding": tgt_enc,
        "n_source_columns": len(src_header),
        "n_target_columns": len(tgt_header),
        "n_source_rows": src_rows,
        "n_target_rows": tgt_rows,
        "n_ground_truth": len(case.ground_truth),
        "invalid_ground_truth_pairs": [list(p) for p in invalid],
    }
    atomic_write_json(out_dir / "metadata.json", metadata)
    return CaseRecord(
        dataset=case.dataset,
        case_id=case.case_id,
        source_path=_rel(out_dir / "source.csv"),
        target_path=_rel(out_dir / "target.csv"),
        ground_truth_path=_rel(out_dir / "ground_truth.csv"),
        metadata_path=_rel(out_dir / "metadata.json"),
        n_source_columns=len(src_header),
        n_target_columns=len(tgt_header),
        n_ground_truth=len(case.ground_truth),
    )


# ------------------------------------------------------------- integrity
def verify_case(config: Config, raw: RawCase, record: CaseRecord) -> list[str]:
    """Return a list of integrity violations for one case (empty = OK)."""
    import pandas as pd

    errors: list[str] = []
    meta = read_json(record.abs("metadata_path"))
    for key, raw_path in (("source", raw.source_file), ("target", raw.target_file)):
        processed = record.abs(f"{key}_path")
        raw_text, _ = read_text(raw_path)
        proc_text, _ = read_text(processed)
        if raw_text.lstrip("﻿") != proc_text.lstrip("﻿"):
            errors.append(f"{key}.csv content differs from raw file")
        if sha256_file(raw_path) != meta[f"raw_{key}_sha256"]:
            errors.append(f"raw {key} file changed since processing")
        raw_header, raw_rows = csv_shape(raw_text)
        header, rows = csv_shape(proc_text)
        if raw_header != header:
            errors.append(f"{key} column names changed")
        if raw_rows != rows:
            errors.append(f"{key} row count changed ({raw_rows} -> {rows})")
        if len(set(header)) != len(header):
            errors.append(f"{key} has duplicate column names")
        loaded = list(pd.read_csv(processed, nrows=0).columns.astype(str))
        if loaded != [str(h) for h in header]:
            errors.append(f"{key} columns differ when loaded with pandas")
    gt = read_ground_truth(record.abs("ground_truth_path"))
    if gt != raw.ground_truth:
        errors.append("ground-truth pairs changed")
    if len(gt) != record.n_ground_truth:
        errors.append("ground-truth count mismatch with manifest")
    src_header = csv_shape(read_text(record.abs("source_path"))[0])[0]
    tgt_header = csv_shape(read_text(record.abs("target_path"))[0])[0]
    allowed = known_defects(config)
    for a, b in invalid_gt_pairs(gt, src_header, tgt_header):
        if (record.case_id, a, b) not in allowed:
            errors.append(f"ground-truth pair ({a!r}, {b!r}) references a missing column")
    return errors


def verify_processed(config: Config, raw_cases: list[RawCase], records: list[CaseRecord]) -> list[str]:
    errors: list[str] = []
    by_id = {r.case_id: r for r in records}
    raw_ids = {c.case_id for c in raw_cases}
    if raw_ids != set(by_id):
        errors.append(
            f"raw/processed case sets differ: missing={sorted(raw_ids - set(by_id))} "
            f"extra={sorted(set(by_id) - raw_ids)}"
        )
    for raw in raw_cases:
        record = by_id.get(raw.case_id)
        if record is None:
            continue
        errors.extend(f"{raw.case_id}: {e}" for e in verify_case(config, raw, record))
    return errors


# ------------------------------------------------------------------- main
def run(config: Config, datasets: Iterable[str], verify_only: bool = False) -> list[CaseRecord]:
    datasets = list(datasets)
    raw_cases = discover(config, datasets)
    if verify_only:
        records = [r for r in read_manifest(config.manifest_path) if r.dataset in datasets]
    else:
        records = []
        for case in raw_cases:
            records.append(convert_case(config, case))
        existing = []
        if config.manifest_path.is_file():
            existing = [r for r in read_manifest(config.manifest_path) if r.dataset not in datasets]
        write_manifest(config.manifest_path, existing + records)
        log.info("wrote %d cases to %s", len(records), config.manifest_path)
    errors = verify_processed(config, raw_cases, records)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.dataset] = counts.get(r.dataset, 0) + 1
    log.info("case counts: %s", counts)
    empty = [r.case_id for r in records if r.n_ground_truth == 0]
    if empty:
        log.warning("%d case(s) with empty ground truth (skipped by the evaluator): %s", len(empty), empty)
    if errors:
        for e in errors:
            log.error("integrity: %s", e)
        raise IntegrityError(f"{len(errors)} integrity violation(s)")
    log.info("integrity checks passed")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)
    config = load_config(config_dir=args.config_dir)
    try:
        run(config, args.datasets, verify_only=args.verify_only)
    except (IntegrityError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
