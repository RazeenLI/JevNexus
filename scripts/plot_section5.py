"""Reproduce the compact two-panel figures used in Section 5."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "saves" / "predictions"
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "Writing" / "6ab7898ff4be0b3a8d5f4255" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = "#4C78A8", "#F58518", "#54A24B", "#E45756", "#BAB0AC"
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    padding = 1 / 220
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=padding)
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=padding)
    plt.close(fig)


def efficiency() -> None:
    per_dataset = pd.read_csv(METRICS / "per_dataset.csv")
    per_case = pd.read_csv(METRICS / "per_case.csv")
    methods = ["coma", "coma_plus", "similarity_flooding", "unicorn",
               "isresmat", "magneto_qwen", "dema"]
    labels = {
        "coma": "COMA", "coma_plus": "COMA++",
        "similarity_flooding": "SF", "unicorn": "Unicorn",
        "isresmat": "ISResMat", "magneto_qwen": "Magneto", "dema": "DeMa",
    }

    # The failed Magneto unit is part of the system-level comparison. Its
    # elapsed time is recovered from the status record; effectiveness is zero.
    failure_path = (ROOT / "saves" / "status" / "magneto_qwen" / "OpenData" /
                    "opendata_joinable_miller2_both_50_50_ac5_ev.json")
    status = json.loads(failure_path.read_text())
    failure_seconds = (
        datetime.fromisoformat(status["finished_at"]) -
        datetime.fromisoformat(status["started_at"])
    ).total_seconds()

    effectiveness, runtime = {}, {}
    for method in methods:
        rows = per_dataset[per_dataset["method"] == method].copy()
        adjusted = rows["MRR"] * rows["n_cases"] / rows["n_cases_expected"]
        effectiveness[method] = adjusted.mean()
        values = per_case[per_case["method"] == method]["total_seconds"]
        if method == "magneto_qwen":
            values = pd.concat([values, pd.Series([failure_seconds])], ignore_index=True)
        runtime[method] = values.mean()

    summaries = []
    for method in ["dema", "magneto_qwen"]:
        values = per_case[per_case["method"] == method]["total_seconds"]
        if method == "magneto_qwen":
            values = pd.concat([values, pd.Series([failure_seconds])], ignore_index=True)
        summaries.append([values.mean(), values.median(), values.quantile(0.95)])

    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.55))
    for method in methods:
        color = BLUE if method == "dema" else ORANGE if method == "magneto_qwen" else GRAY
        left.scatter(runtime[method], effectiveness[method], s=24, color=color,
                     edgecolor="white", linewidth=0.4, zorder=3)
        offset = {
            "coma": (2, -7), "coma_plus": (2, 3), "similarity_flooding": (2, -1),
            "unicorn": (2, 3), "isresmat": (-31, -8), "magneto_qwen": (-28, -8),
            "dema": (-17, 4),
        }[method]
        left.annotate(labels[method], (runtime[method], effectiveness[method]),
                      xytext=offset, textcoords="offset points", fontsize=5.2)
    left.set_xscale("log")
    left.set_xlabel("Mean latency (s, log)")
    left.set_ylabel("Dataset-macro MRR")
    left.set_title("(a) Effectiveness--runtime")
    left.grid(alpha=0.2, linewidth=0.5)
    left.set_axisbelow(True)

    positions, width = np.arange(3), 0.34
    right.bar(positions - width / 2, summaries[0], width, color=BLUE, label="DeMa")
    right.bar(positions + width / 2, summaries[1], width, color=ORANGE, label="Magneto")
    right.set_xticks(positions, ["Mean", "Median", "P95"])
    right.set_yscale("log")
    right.set_title("(b) Latency (s, log scale)")
    right.grid(axis="y", alpha=0.2, linewidth=0.5)
    right.set_axisbelow(True)
    right.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.27, wspace=0.32)
    save(fig, "efficiency")


def sensitivity() -> None:
    alpha = pd.read_csv(METRICS / "sensitivity_alpha.csv")
    tau = pd.read_csv(METRICS / "sensitivity_tau.csv")
    selected = ["ALL", "GDC", "OpenData"]
    labels = {"ALL": "Overall", "GDC": "GDC", "OpenData": "OpenData"}
    colors = {"ALL": "#111111", "GDC": RED, "OpenData": "#B279A2"}
    markers = {"ALL": "o", "GDC": "^", "OpenData": "s"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.65))
    for dataset in selected:
        rows = alpha[alpha["dataset"] == dataset].sort_values("alpha")
        left.plot(rows["alpha"], rows["Recall@GT"], marker=markers[dataset], markersize=2.5,
                  linewidth=1.4 if dataset == "ALL" else 0.9, color=colors[dataset],
                  label=labels[dataset])
    left.axvline(0.4, color="0.45", linestyle="--", linewidth=0.7)
    left.set_xlabel(r"Fusion weight $\alpha$")
    left.set_ylabel("Recall@GT")
    left.set_title("(a) Evidence fusion")
    left.grid(axis="y", alpha=0.2, linewidth=0.5)
    left.legend(frameon=False, loc="lower center")

    overall = tau[tau["dataset"] == "ALL"].sort_values("tau")
    right.plot(overall["tau"], overall["Recall@GT"], marker="o", markersize=2.5,
               color=BLUE)
    right.set_xlabel(r"Gate threshold $\tau$")
    right.set_ylabel("Recall@GT", color=BLUE)
    right.tick_params(axis="y", colors=BLUE)
    right.axvline(0.02, color="0.45", linestyle="--", linewidth=0.7)
    activation = right.twinx()
    activation.plot(overall["tau"], 100 * overall["gate_activation_rate"], marker="s",
                    markersize=2.5, color=ORANGE)
    activation.set_ylabel("Activation (\%)", color=ORANGE)
    activation.tick_params(axis="y", colors=ORANGE)
    right.set_title("(b) Selective routing")
    right.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.86, bottom=0.25, wspace=0.52)
    save(fig, "sensitivity")


def _load_predictions(method: str, dataset: str, case_id: str) -> dict:
    path = PREDICTIONS / method / dataset / f"{case_id}.json"
    return json.loads(path.read_text())["predictions"]


def _ground_truth(dataset: str, case_id: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with (DATA / dataset / case_id / "ground_truth.csv").open() as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["source_column"], set()).add(row["target_column"])
    return result


def _rr(entries: list[dict], targets: set[str]) -> float:
    ranks = [int(row.get("final_rank", row.get("rank", 10**9)))
             for row in entries if row["target_column"] in targets]
    return 1 / min(ranks) if ranks else 0.0


def _outcome(before: float, after: float) -> str:
    if after > before + 1e-12:
        return "Improved"
    if after < before - 1e-12:
        return "Harmed"
    return "Unchanged"


def mechanism_analysis() -> None:
    fusion, routed, bypassed = Counter(), Counter(), Counter()
    methods = ["dema_decision", "dema_no_rerank", "dema", "dema_always"]
    for path in (PREDICTIONS / "dema").glob("*/*.json"):
        dataset, case_id = path.parent.name, path.stem
        predictions = {method: _load_predictions(method, dataset, case_id) for method in methods}
        for source, targets in _ground_truth(dataset, case_id).items():
            if source not in predictions["dema"]:
                continue
            decision_rr = _rr(predictions["dema_decision"].get(source, []), targets)
            fused_rr = _rr(predictions["dema_no_rerank"].get(source, []), targets)
            fusion[_outcome(decision_rr, fused_rr)] += 1
            activated = any(row.get("gate_activated", False) for row in predictions["dema"][source])
            if activated:
                routed[_outcome(fused_rr, _rr(predictions["dema"][source], targets))] += 1
            else:
                bypassed[_outcome(fused_rr, _rr(predictions["dema_always"].get(source, []), targets))] += 1

    categories, colors = ["Improved", "Unchanged", "Harmed"], [GREEN, GRAY, RED]
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.45))

    def stacked(ax, counters, labels):
        left_edge = np.zeros(len(counters))
        for category, color in zip(categories, colors):
            values = np.array([100 * counter[category] / sum(counter.values()) for counter in counters])
            ax.barh(labels, values, left=left_edge, color=color, height=0.48, label=category)
            for index, value in enumerate(values):
                if value >= 7:
                    ax.text(left_edge[index] + value / 2, index, f"{value:.1f}",
                            ha="center", va="center", fontsize=5.5)
            left_edge += values
        ax.set_xlim(0, 100)
        ax.set_xlabel("GT-bearing queries (\%)")
        ax.grid(axis="x", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    stacked(left, [fusion], ["Fusion"])
    left.set_title("(a) Complementary evidence")
    stacked(right, [routed, bypassed], ["Routed", "Bypassed\n(always replay)"])
    right.set_title("(b) Refinement outcomes")
    handles, legend_labels = left.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), columnspacing=0.9, handlelength=1.2)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.82, bottom=0.35, wspace=0.52)
    save(fig, "mechanism_analysis")


def diagnostics() -> None:
    raw = pd.read_csv(METRICS / "candidate_coverage.csv")
    raw = raw[raw["case_id"] != "ALL"]
    counts = raw.groupby(["dataset", "category"], as_index=False)["count"].sum()
    overall = counts.groupby("category", as_index=False)["count"].sum()
    overall["dataset"] = "Overall"
    counts = pd.concat([overall, counts], ignore_index=True)
    categories = ["correct_top1", "final_top3_not_top1", "candidate_top20_not_final_top3",
                  "outside_candidate_top20", "invalid_or_absent_target"]
    category_labels = ["Top-1", "Top-3", "In Top-20", "Outside", "Invalid"]
    category_colors = [GREEN, "#72B7B2", "#E2B04A", RED, "#777777"]
    table = counts.pivot(index="dataset", columns="category", values="count").fillna(0)
    shares = table[categories].div(table[categories].sum(axis=1), axis=0) * 100

    coverage_counts = {1: 0, 5: 0, 10: 0, 20: 0}
    total = 0
    for path in (PREDICTIONS / "dema").glob("*/*.json"):
        dataset, case_id = path.parent.name, path.stem
        predictions = json.loads(path.read_text())["predictions"]
        for source, targets in _ground_truth(dataset, case_id).items():
            by_target = {row["target_column"]: row for row in predictions.get(source, [])}
            for target in targets:
                total += 1
                rank = by_target.get(target, {}).get("retrieval_rank", 10**9)
                for k in coverage_counts:
                    if rank is not None and rank <= k:
                        coverage_counts[k] += 1

    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.55))
    ks = list(coverage_counts)
    coverage = [100 * coverage_counts[k] / total for k in ks]
    left.plot(ks, coverage, marker="o", color=BLUE, linewidth=1.4, markersize=3)
    for x, y in zip(ks, coverage):
        left.text(x, y + 0.8, f"{y:.1f}", ha="center", fontsize=5.5)
    left.set_xticks(ks)
    left.set_ylim(70, 101)
    left.set_xlabel("Candidate limit $k$")
    left.set_ylabel("GT coverage (\%)")
    left.set_title("(a) Localization coverage")
    left.grid(axis="y", alpha=0.2, linewidth=0.5)

    shown = ["Overall", "GDC"]
    left_edge = np.zeros(len(shown))
    for category, label, color in zip(categories, category_labels, category_colors):
        values = shares.reindex(shown)[category].to_numpy()
        right.barh(shown, values, left=left_edge, color=color, height=0.5, label=label)
        left_edge += values
    right.invert_yaxis()
    right.set_xlim(0, 100)
    right.set_xlabel("GT pairs (\%)")
    right.set_title("(b) Stage-level errors")
    right.legend(ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.50),
                 columnspacing=0.6, handlelength=0.9, fontsize=5.0)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.85, bottom=0.25, wspace=0.48)
    save(fig, "diagnostics")


if __name__ == "__main__":
    efficiency()
    sensitivity()
    mechanism_analysis()
    diagnostics()
