"""Evaluate saved predictions (no model inference happens here).

    python -m dema.evaluation.evaluator [--methods ...] [--datasets ...]

Writes ``outputs/metrics/{per_case,per_dataset,overall,completeness}.csv``.
Only units whose status is complete (see ``dema.runner.unit_state``) are
evaluated; everything else is reported in ``completeness.csv`` and in the log.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from ..data.loader import read_ground_truth
from ..data.manifest import CaseRecord, read_manifest, select_cases
from ..runner import OutputPaths, unit_state
from ..utils.config import Config, load_config
from ..utils.io import read_json
from ..utils.logging import get_console_logger
from .aggregate import aggregate_overall, aggregate_per_dataset
from .metrics import all_metrics

log = get_console_logger("dema.evaluate")

RUNTIME_FIELDS = (
    "total_seconds", "representation_seconds", "retrieval_seconds", "reranking_seconds",
    "matching_seconds", "ranking_seconds", "model_requests", "input_tokens", "output_tokens",
    "retries", "failures", "peak_gpu_memory_mb",
)


def load_scored_ranking(doc: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    out = {}
    for src, entries in doc["predictions"].items():
        ordered = sorted(entries, key=lambda e: e["rank"])
        out[src] = [(e["target_column"], float(e["score"])) for e in ordered]
    return out


def evaluate_case(paths: OutputPaths, method: str, record: CaseRecord) -> dict[str, Any]:
    doc = read_json(paths.prediction(method, record.dataset, record.case_id))
    gt = read_ground_truth(record.abs("ground_truth_path"))
    row: dict[str, Any] = {
        "method": method, "dataset": record.dataset, "case_id": record.case_id,
        "n_source_columns": record.n_source_columns, "n_target_columns": record.n_target_columns,
        "n_ground_truth": len(set(gt)), "n_queries": len({s for s, _ in gt}),
    }
    row.update(all_metrics(load_scored_ranking(doc), gt))
    rt_path = paths.runtime(method, record.dataset, record.case_id)
    runtime = read_json(rt_path) if rt_path.is_file() else {}
    for f in RUNTIME_FIELDS:
        row[f] = runtime.get(f)
    return row


def evaluate(config: Config, methods: list[str], datasets: list[str]) -> dict[str, pd.DataFrame]:
    paths = OutputPaths(config.outputs_dir)
    records = select_cases(read_manifest(config.manifest_path), datasets)
    records = [r for r in records if r.n_ground_truth > 0]
    rows, completeness = [], []
    for method in methods:
        for record in records:
            state = unit_state(paths, method, record)
            completeness.append({"method": method, "dataset": record.dataset, "case_id": record.case_id,
                                 "state": state})
            if state == "complete":
                rows.append(evaluate_case(paths, method, record))
    per_case = pd.DataFrame(rows)
    comp = pd.DataFrame(completeness)
    expected = comp.groupby(["method", "dataset"]).size().rename("n_cases_expected").reset_index()
    per_dataset = aggregate_per_dataset(per_case, expected)
    overall = aggregate_overall(per_case, per_dataset, expected)
    return {"per_case": per_case, "per_dataset": per_dataset, "overall": overall, "completeness": comp}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--config-dir", default=None)
    args = p.parse_args(argv)
    config = load_config(args.config, args.config_dir)
    methods = args.methods or list(config.experiment["methods"])
    datasets = args.datasets or list(config.experiment["datasets"])
    tables = evaluate(config, methods, datasets)
    out = config.outputs_dir / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    comp = tables["completeness"]
    incomplete = comp[comp["state"] != "complete"]
    if len(incomplete):
        summary = incomplete.groupby(["method", "dataset", "state"]).size()
        log.warning("incomplete method/dataset units (not evaluated):\n%s", summary.to_string())
    else:
        log.info("all %d method x case units complete", len(comp))
    if len(tables["overall"]):
        cols = [c for c in ("method", "weighting", "n_cases", "complete", "MRR", "Recall@GT", "Recall@20")
                if c in tables["overall"].columns]
        log.info("overall:\n%s", tables["overall"][cols].to_string(index=False))
    log.info("metrics written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
