"""Generate no-inference diagnostics used by the DeMa paper.

Outputs candidate localization, gate routing, and paired bootstrap comparisons
from completed prediction artifacts. Missing method/case units are reported both
on the common-case subset and with failures conservatively scored as zero.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.loader import read_ground_truth
from ..metrics.evaluator import evaluate_case
from ..metrics.aggregate import aggregate_overall, aggregate_per_dataset
from ..utils.config import load_config
from ..utils.io import read_json
from ..utils.logging import get_console_logger
from .runner import ExperimentPaths, selected_records, unit_state

log = get_console_logger("dema.publication_analysis")


def candidate_coverage(config, method: str, datasets: list[str]) -> pd.DataFrame:
    paths = ExperimentPaths.from_config(config)
    rows = []
    for record in selected_records(config, datasets):
        if record.n_ground_truth <= 0 or unit_state(paths, method, record) != "complete":
            continue
        doc = read_json(paths.prediction(method, record.dataset, record.case_id))
        gt = set(read_ground_truth(record.abs("ground_truth_path")))
        counts: Counter[str] = Counter()
        for source, target in gt:
            entries = doc["predictions"].get(source, [])
            entry = next((item for item in entries if item["target_column"] == target), None)
            if entry is None:
                counts["invalid_or_absent_target"] += 1
            elif entry.get("jev_score") is None:
                counts["outside_candidate_top20"] += 1
            elif int(entry["rank"]) > 3:
                counts["candidate_top20_not_final_top3"] += 1
            elif int(entry["rank"]) > 1:
                counts["final_top3_not_top1"] += 1
            else:
                counts["correct_top1"] += 1
        for category in (
            "correct_top1", "final_top3_not_top1", "candidate_top20_not_final_top3",
            "outside_candidate_top20", "invalid_or_absent_target",
        ):
            rows.append({
                "method": method,
                "dataset": record.dataset,
                "case_id": record.case_id,
                "category": category,
                "count": counts[category],
                "n_ground_truth": len(gt),
                "rate": counts[category] / len(gt) if gt else np.nan,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    totals = frame.groupby(["method", "dataset", "category"], as_index=False).agg(
        count=("count", "sum"), n_ground_truth=("n_ground_truth", "sum")
    )
    totals["rate"] = totals["count"] / totals["n_ground_truth"]
    totals["case_id"] = "ALL"
    return pd.concat([frame, totals[frame.columns]], ignore_index=True)


def gate_routing(config, method: str, datasets: list[str]) -> pd.DataFrame:
    paths = ExperimentPaths.from_config(config)
    rows = []
    for record in selected_records(config, datasets):
        if unit_state(paths, method, record) != "complete":
            continue
        runtime = read_json(paths.runtime(method, record.dataset, record.case_id))
        evaluations = int(runtime.get("gate_evaluations") or 0)
        activations = int(runtime.get("gate_activations") or 0)
        rows.append({
            "method": method,
            "dataset": record.dataset,
            "case_id": record.case_id,
            "gate_evaluations": evaluations,
            "gate_activations": activations,
            "jina_requests": int(runtime.get("jina_requests") or 0),
            "gate_activation_rate": activations / evaluations if evaluations else np.nan,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    totals = frame.groupby(["method", "dataset"], as_index=False)[
        ["gate_evaluations", "gate_activations", "jina_requests"]
    ].sum()
    totals["case_id"] = "ALL"
    totals["gate_activation_rate"] = totals["gate_activations"] / totals["gate_evaluations"]
    return pd.concat([frame, totals[frame.columns]], ignore_index=True)


def _bootstrap_dataset_macro(
    differences: dict[str, np.ndarray], samples: int, seed: int
) -> tuple[float, float, float, float]:
    observed = float(np.mean([values.mean() for values in differences.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        draws[index] = np.mean([
            rng.choice(values, size=len(values), replace=True).mean()
            for values in differences.values()
        ])
    low, high = np.percentile(draws, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))))
    return observed, float(low), float(high), p_value


def paired_bootstrap(
    config, method_a: str, method_b: str, datasets: list[str], samples: int, seed: int
) -> pd.DataFrame:
    paths = ExperimentPaths.from_config(config)
    records = [record for record in selected_records(config, datasets) if record.n_ground_truth > 0]
    cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
    for method in (method_a, method_b):
        for record in records:
            key = method, record.dataset, record.case_id
            cache[key] = (
                evaluate_case(paths, method, record)
                if unit_state(paths, method, record) == "complete" else None
            )

    rows = []
    for missing_policy in ("common_cases", "failure_as_zero"):
        for metric in ("MRR", "Hits@1", "Recall@GT"):
            differences: dict[str, list[float]] = {}
            for record in records:
                left = cache[(method_a, record.dataset, record.case_id)]
                right = cache[(method_b, record.dataset, record.case_id)]
                if missing_policy == "common_cases" and (left is None or right is None):
                    continue
                left_value = float(left[metric]) if left is not None else 0.0
                right_value = float(right[metric]) if right is not None else 0.0
                differences.setdefault(record.dataset, []).append(left_value - right_value)
            arrays = {dataset: np.asarray(values) for dataset, values in differences.items() if values}
            if not arrays:
                continue
            observed, low, high, p_value = _bootstrap_dataset_macro(arrays, samples, seed)
            rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "estimand": "dataset_macro_paired_difference_a_minus_b",
                "missing_policy": missing_policy,
                "n_cases": sum(len(values) for values in arrays.values()),
                "n_datasets": len(arrays),
                "difference": observed,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_p_two_sided": p_value,
                "bootstrap_samples": samples,
                "seed": seed,
            })
    return pd.DataFrame(rows)


def heldout_evaluation(config, dev_config, methods: list[str], datasets: list[str]) -> dict[str, pd.DataFrame]:
    """Evaluate full-run predictions after excluding every development case."""
    paths = ExperimentPaths.from_config(config)
    dev_ids = {
        (record.dataset, record.case_id)
        for record in selected_records(dev_config, datasets)
    }
    records = [
        record for record in selected_records(config, datasets)
        if record.n_ground_truth > 0 and (record.dataset, record.case_id) not in dev_ids
    ]
    expected_rows = []
    rows = []
    for method in methods:
        per_dataset = Counter(record.dataset for record in records)
        expected_rows.extend(
            {"method": method, "dataset": dataset, "n_cases_expected": count}
            for dataset, count in per_dataset.items()
        )
        for record in records:
            if unit_state(paths, method, record) == "complete":
                rows.append(evaluate_case(paths, method, record))
    per_case = pd.DataFrame(rows)
    expected = pd.DataFrame(expected_rows)
    per_dataset = aggregate_per_dataset(per_case, expected)
    overall = aggregate_overall(per_case, per_dataset, expected)
    return {"per_case": per_case, "per_dataset": per_dataset, "overall": overall}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dev-config", default="configs/experiment_dev.yaml")
    parser.add_argument("--method", default="dema")
    parser.add_argument("--compare", default="magneto_qwen")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    dev_config = load_config(args.dev_config)
    datasets = args.datasets or list(config.experiment["datasets"])
    destination = args.output_dir or config.metrics_dir
    destination.mkdir(parents=True, exist_ok=True)
    coverage = candidate_coverage(config, args.method, datasets)
    routing = gate_routing(config, args.method, datasets)
    bootstrap = paired_bootstrap(
        config, args.method, args.compare, datasets, args.bootstrap_samples, args.seed
    )
    heldout = heldout_evaluation(config, dev_config, [args.method, args.compare], datasets)
    coverage.to_csv(destination / "candidate_coverage.csv", index=False)
    routing.to_csv(destination / "gate_routing.csv", index=False)
    bootstrap.to_csv(destination / "paired_bootstrap.csv", index=False)
    for name, frame in heldout.items():
        frame.to_csv(destination / f"heldout_{name}.csv", index=False)
    log.info("publication diagnostics written to %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
