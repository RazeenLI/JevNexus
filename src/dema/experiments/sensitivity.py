"""Offline alpha/tau sensitivity from one ``dema_always`` run.

The Always-Jina artifact stores Jev, COMA+, fixed-fusion and Jina ranks for
every retrieved candidate. Alpha therefore changes only the no-refinement
fusion, while tau exactly replays the canonical m=3 gate without another model
call. This script performs no inference.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.loader import read_ground_truth
from ..metrics.metrics import METRIC_NAMES, all_metrics
from ..metrics.evaluator import evaluate
from ..utils.config import load_config
from ..utils.io import read_json
from ..utils.logging import get_console_logger
from .runner import ExperimentPaths, selected_records, unit_state

log = get_console_logger("dema.sensitivity")


def _candidate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [entry for entry in entries if entry.get("jev_score") is not None]
    if not candidates:
        raise ValueError(
            "prediction has no component scores; run method 'dema_always' with the current code"
        )
    return sorted(candidates, key=lambda entry: int(entry["retrieval_rank"]))


def _finish_ranking(
    entries: list[dict[str, Any]], candidate_scores: dict[str, float]
) -> list[tuple[str, float]]:
    candidates = _candidate_entries(entries)
    candidate_names = {entry["target_column"] for entry in candidates}
    ranked = sorted(
        candidates,
        key=lambda entry: (
            -candidate_scores[entry["target_column"]], int(entry["retrieval_rank"])
        ),
    )
    tail = sorted(
        (entry for entry in entries if entry["target_column"] not in candidate_names),
        key=lambda entry: int(entry["retrieval_rank"]),
    )
    return [
        (entry["target_column"], float(candidate_scores[entry["target_column"]]))
        for entry in ranked
    ] + [(entry["target_column"], 0.0) for entry in tail]


def replay_alpha(doc: dict[str, Any], alpha: float) -> dict[str, list[tuple[str, float]]]:
    """Replay fixed fusion with no Jina refinement."""
    scored = {}
    for source, entries in doc["predictions"].items():
        candidates = _candidate_entries(entries)
        scores = {
            entry["target_column"]: (
                alpha * float(entry["jev_score"])
                + (1.0 - alpha) * float(entry["coma_plus_score"])
            )
            for entry in candidates
        }
        scored[source] = _finish_ranking(entries, scores)
    return scored


def replay_tau(
    doc: dict[str, Any], threshold: float, top_n: int = 3
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, int]]:
    """Replay disagreement-and-margin gating using saved Always-Jina ranks."""
    scored = {}
    routing = {"evaluations": 0, "disagreements": 0, "activations": 0}
    for source, entries in doc["predictions"].items():
        candidates = _candidate_entries(entries)
        by_retrieval = {entry["target_column"]: int(entry["retrieval_rank"]) for entry in candidates}

        def ordered(field: str) -> list[dict[str, Any]]:
            return sorted(
                candidates,
                key=lambda entry: (-float(entry[field]), by_retrieval[entry["target_column"]]),
            )

        fusion_order = ordered("fusion_score")
        jev_top = ordered("jev_score")[0]["target_column"]
        coma_available = any(float(entry["coma_plus_score"]) > 0.0 for entry in candidates)
        coma_top = ordered("coma_plus_score")[0]["target_column"] if coma_available else None
        disagreement = coma_top is not None and jev_top != coma_top
        margin = (
            float(fusion_order[0]["fusion_score"]) - float(fusion_order[1]["fusion_score"])
            if len(fusion_order) >= 2 else math.inf
        )
        activate = disagreement and margin < threshold
        routing["evaluations"] += 1
        routing["disagreements"] += int(disagreement)
        routing["activations"] += int(activate)

        scores = {
            entry["target_column"]: float(entry["fusion_score"]) for entry in candidates
        }
        if activate:
            refine = fusion_order[: min(top_n, len(fusion_order))]
            if any(entry.get("jina_rank") is None for entry in refine):
                raise ValueError(
                    f"saved Always-Jina output for {source!r} lacks Top-{top_n} Jina ranks"
                )
            jina_order = sorted(refine, key=lambda entry: int(entry["jina_rank"]))
            slots = sorted((float(entry["fusion_score"]) for entry in refine), reverse=True)
            for entry, score in zip(jina_order, slots):
                scores[entry["target_column"]] = score
        scored[source] = _finish_ranking(entries, scores)
    return scored, routing


def _summarize(per_case: pd.DataFrame, parameter: str) -> pd.DataFrame:
    group_cols = [parameter, "dataset"]
    numeric = list(METRIC_NAMES) + ["gate_evaluations", "gate_disagreements", "gate_activations"]
    per_dataset = per_case.groupby(group_cols, as_index=False)[numeric].mean()
    per_dataset["weighting"] = "case_macro"
    overall = per_dataset.groupby(parameter, as_index=False)[list(METRIC_NAMES)].mean()
    totals = per_case.groupby(parameter, as_index=False)[
        ["gate_evaluations", "gate_disagreements", "gate_activations"]
    ].sum()
    overall = overall.merge(totals, on=parameter)
    overall["dataset"] = "ALL"
    overall["weighting"] = "dataset_macro"
    output = pd.concat([per_dataset, overall], ignore_index=True)
    output["gate_activation_rate"] = output["gate_activations"].div(
        output["gate_evaluations"].replace(0, pd.NA)
    )
    return output


def run_sensitivity(
    config_path: str,
    source_method: str,
    datasets: list[str] | None,
    alphas: list[float],
    taus: list[float],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    selected_datasets = datasets or list(config.experiment["datasets"])
    records = [
        record for record in selected_records(config, selected_datasets)
        if record.n_ground_truth > 0
    ]
    paths = ExperimentPaths.from_config(config)
    alpha_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    for record in records:
        if unit_state(paths, source_method, record) != "complete":
            raise FileNotFoundError(
                f"{source_method}/{record.dataset}/{record.case_id} is not complete; "
                f"run {source_method} first"
            )
        doc = read_json(paths.prediction(source_method, record.dataset, record.case_id))
        gt = read_ground_truth(record.abs("ground_truth_path"))
        for alpha in alphas:
            row = {
                "alpha": alpha,
                "dataset": record.dataset,
                "case_id": record.case_id,
                "gate_evaluations": 0,
                "gate_disagreements": 0,
                "gate_activations": 0,
            }
            row.update(all_metrics(replay_alpha(doc, alpha), gt))
            alpha_rows.append(row)
        for tau in taus:
            ranking, routing = replay_tau(doc, tau)
            row = {"tau": tau, "dataset": record.dataset, "case_id": record.case_id}
            row.update({f"gate_{key}": value for key, value in routing.items()})
            row.update(all_metrics(ranking, gt))
            tau_rows.append(row)

    alpha_case = pd.DataFrame(alpha_rows)
    tau_case = pd.DataFrame(tau_rows)
    alpha_summary = _summarize(alpha_case, "alpha")
    tau_summary = _summarize(tau_case, "tau")
    destination = output_dir or config.metrics_dir
    destination.mkdir(parents=True, exist_ok=True)
    alpha_case.to_csv(destination / "sensitivity_alpha_per_case.csv", index=False)
    tau_case.to_csv(destination / "sensitivity_tau_per_case.csv", index=False)
    alpha_summary.to_csv(destination / "sensitivity_alpha.csv", index=False)
    tau_summary.to_csv(destination / "sensitivity_tau.csv", index=False)
    log.info("offline sensitivity written to %s (%d cases)", destination, len(records))
    return alpha_summary, tau_summary


def write_m_summary(config_path: str, methods: list[str], output_dir: Path | None = None) -> pd.DataFrame:
    """Summarize online m=2/3/5 runs over the configured development split."""
    config = load_config(config_path)
    tables = evaluate(config, methods, list(config.experiment["datasets"]))
    mapping = {"dema_gate_m2": 2, "dema": 3, "dema_gate_m5": 5}
    frames = []
    for table_name in ("per_dataset", "overall"):
        frame = tables[table_name].copy()
        if frame.empty:
            continue
        frame["m"] = frame["method"].map(mapping)
        frame["scope"] = table_name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    destination = output_dir or config.metrics_dir
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "sensitivity_m.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_dev.yaml")
    parser.add_argument("--source-method", default="dema_always")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--m-methods", nargs="+", default=None,
        help="also summarize completed online m runs (normally dema_gate_m2 dema dema_gate_m5)",
    )
    args = parser.parse_args(argv)
    if any(not 0 <= alpha <= 1 for alpha in args.alphas):
        parser.error("every alpha must be in [0, 1]")
    if any(tau < 0 for tau in args.taus):
        parser.error("every tau must be non-negative")
    run_sensitivity(
        args.config, args.source_method, args.datasets, args.alphas, args.taus, args.output_dir
    )
    if args.m_methods:
        write_m_summary(args.config, args.m_methods, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
