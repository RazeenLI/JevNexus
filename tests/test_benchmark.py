import csv
import json

import pytest

from dema.data.loader import load_case
from dema.data.manifest import read_manifest
from dema.data.prepare import (
    IntegrityError, csv_shape, discover, invalid_gt_pairs, read_text, run as prepare_run, verify_processed,
)
from dema.utils.config import load_config


def test_counts_names_and_ground_truth_preserved(mini_benchmark):
    config, records = mini_benchmark
    raw_cases = discover(config, ["GDC", "OpenData"])
    assert len(raw_cases) == len(records) == len(read_manifest(config.manifest_path)) == 2
    assert verify_processed(config, raw_cases, records) == []
    by_id = {r.case_id: r for r in records}
    for raw in raw_cases:
        rec = by_id[raw.case_id]
        for key, raw_file in (("source_path", raw.source_file), ("target_path", raw.target_file)):
            assert csv_shape(read_text(raw_file)[0]) == csv_shape(read_text(rec.abs(key))[0])
        case = load_case(rec)
        assert case.ground_truth == raw.ground_truth
        assert len(case.ground_truth) == rec.n_ground_truth
        assert invalid_gt_pairs(case.ground_truth, list(case.source_df.columns), list(case.target_df.columns)) == []


def test_encoding_normalized_but_content_identical(mini_benchmark):
    config, records = mini_benchmark
    rec = next(r for r in records if r.dataset == "OpenData")
    meta = json.loads(rec.abs("metadata_path").read_text())
    assert meta["raw_source_encoding"] == "latin-1"
    case = load_case(rec)
    assert list(case.source_df.columns) == ["city", "café_name", "rating"]
    assert case.source_df["city"].tolist() == ["Paris", "Oslo"]


def test_integrity_check_detects_tampering(mini_benchmark):
    config, records = mini_benchmark
    rec = next(r for r in records if r.dataset == "GDC")
    path = rec.abs("ground_truth_path")
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n")  # drop one GT pair
    errors = verify_processed(config, discover(config, ["GDC"]), [rec])
    assert any("ground-truth" in e for e in errors)


def test_unknown_invalid_gt_reference_fails(config, tmp_path):
    from conftest import write_mini_raw

    raw = tmp_path / "raw"
    write_mini_raw(raw)
    gt = raw / "gdc" / "data" / "ground-truth" / "Toy.csv"
    gt.write_text(gt.read_text() + "Missing Column,gender\n")
    config.paths.update(raw_data=str(raw), processed_data=str(tmp_path / "p"), manifest=str(tmp_path / "m.jsonl"))
    with pytest.raises(IntegrityError):
        prepare_run(config, ["GDC"])


def test_idempotent(mini_benchmark):
    config, records = mini_benchmark
    again = prepare_run(config, ["GDC", "OpenData"])
    assert again == records


# ------------------------------------------------------- real benchmark (if prepared)
REAL = load_config()


@pytest.mark.skipif(not REAL.manifest_path.is_file(), reason="real benchmark not prepared")
def test_real_benchmark_manifest_consistent():
    records = read_manifest(REAL.manifest_path)
    counts = {}
    for r in records:
        counts[r.dataset] = counts.get(r.dataset, 0) + 1
    assert counts == {"GDC": 10, "ChEMBL": 180, "Magellan": 7, "OpenData": 180, "TPC-DI": 180, "WikiData": 4}
    allowed = {(d["case_id"], d["source_column"], d["target_column"]) for d in REAL.paths["known_gt_defects"]}
    for r in records:
        with open(r.abs("ground_truth_path"), newline="", encoding="utf-8") as fh:
            gt = [tuple(row) for row in csv.reader(fh)][1:]
        assert len(gt) == r.n_ground_truth
        src, _ = csv_shape(read_text(r.abs("source_path"))[0])
        tgt, _ = csv_shape(read_text(r.abs("target_path"))[0])
        assert len(src) == r.n_source_columns and len(tgt) == r.n_target_columns
        for a, b in invalid_gt_pairs(gt, src, tgt):
            assert (r.case_id, a, b) in allowed
