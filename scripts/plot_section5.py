"""Reproduce the figures used in Section 5."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "saves" / "predictions"
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "Writing" / "6ab7898ff4be0b3a8d5f4255" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# AI / ML profile from the paper's shared academic color system.
BLUE = "#3B73AB"
ORANGE = "#D87C2C"
GREEN = "#3F925C"
RED = "#C14E57"
CYAN = "#289AA4"
GOLD = "#C79C23"
PURPLE = "#7F68AC"
GRAY = "#B8B7B0"
DARK_GRAY = "#8B8C85"
INK = "#252825"
SKY = "#4F9CBF"
BROWN = "#9F6850"

METHODS = ["coma", "coma_plus", "distribution", "similarity_flooding",
           "isresmat", "unicorn", "magneto_qwen", "dema"]
METHOD_LABELS = {
    "coma": "COMA", "coma_plus": "COMA++", "distribution": "Distribution",
    "similarity_flooding": "SF", "isresmat": "ISResMat", "unicorn": "Unicorn",
    "magneto_qwen": "Magneto", "dema": "JevNexus", "jevnexus": "JevNexus",
}
METHOD_COLORS = {
    "coma": "#678598", "coma_plus": "#3F8D81", "distribution": BROWN,
    "similarity_flooding": GOLD, "isresmat": PURPLE, "unicorn": SKY,
    "magneto_qwen": ORANGE, "dema": BLUE,
}
DATASETS = ["ChEMBL", "GDC", "Magellan", "OpenData", "TPC-DI", "WikiData"]
DATASET_COLORS = {
    "ChEMBL": BLUE, "GDC": RED, "Magellan": GREEN,
    "OpenData": PURPLE, "TPC-DI": GOLD, "WikiData": CYAN,
}
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


def _dataset_macro(metric: str) -> dict[str, float]:
    per_dataset = pd.read_csv(METRICS / "per_dataset.csv")
    result = {}
    for method in METHODS:
        rows = per_dataset[per_dataset["method"] == method].copy()
        adjusted = rows[metric] * rows["n_cases"] / rows["n_cases_expected"]
        result[method] = adjusted.mean()
    return result


def headline_results() -> None:
    metrics = [("MRR", "Dataset-macro MRR"), ("Hits@1", "Dataset-macro Hits@1")]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.30), sharey=True)
    positions = np.arange(len(METHODS))
    for index, (metric, title) in enumerate(metrics):
        values = _dataset_macro(metric)
        ax = axes[index]
        ax.barh(positions, [values[m] for m in METHODS],
                color=[METHOD_COLORS[m] for m in METHODS], height=0.68)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        if index == 0:
            ax.set_yticks(positions, [METHOD_LABELS[m] for m in METHODS])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0],
                          ["", "0.25", "0.50", "0.75", "1.00"])
    fig.subplots_adjust(left=0.25, right=0.995, top=0.98, bottom=0.25, wspace=0.12)
    save(fig, "headline_results")


def efficiency() -> None:
    per_dataset = pd.read_csv(METRICS / "per_dataset.csv")
    per_case = pd.read_csv(METRICS / "per_case.csv")

    # The failed Magneto unit is part of the system-level comparison. Its
    # elapsed time is recovered from the status record; effectiveness is zero.
    failure_path = (ROOT / "saves" / "status" / "magneto_qwen" / "OpenData" /
                    "opendata_joinable_miller2_both_50_50_ac5_ev.json")
    status = json.loads(failure_path.read_text())
    failure_seconds = (
        datetime.fromisoformat(status["finished_at"]) -
        datetime.fromisoformat(status["started_at"])
    ).total_seconds()

    effectiveness, summaries = {}, {}
    for method in METHODS:
        rows = per_dataset[per_dataset["method"] == method].copy()
        adjusted = rows["MRR"] * rows["n_cases"] / rows["n_cases_expected"]
        effectiveness[method] = adjusted.mean()
        values = per_case[per_case["method"] == method]["total_seconds"]
        if method == "magneto_qwen":
            values = pd.concat([values, pd.Series([failure_seconds])], ignore_index=True)
        summaries[method] = (values.mean(), values.median(), values.quantile(0.95))

    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.48))
    shown = [m for m in METHODS if summaries[m][0] >= 1.0]
    for method in shown:
        mean = summaries[method][0]
        left.scatter(mean, effectiveness[method], s=20, color=METHOD_COLORS[method],
                     edgecolor="white", linewidth=0.4, zorder=3)
    comparison_y = max(effectiveness["dema"], effectiveness["magneto_qwen"]) + 0.008
    left.annotate("", xy=(summaries["dema"][0], comparison_y),
                  xytext=(summaries["magneto_qwen"][0], comparison_y),
                  arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 0.6})
    left.text(np.sqrt(summaries["dema"][0] * summaries["magneto_qwen"][0]),
              comparison_y + 0.002, r"$7.75\times$", ha="center", va="bottom", fontsize=4.5)
    left.set_xscale("log")
    left.set_xlim(1, 220)
    left.set_ylim(0.645, 0.958)
    left.set_xlabel("Mean latency (s, log)")
    left.set_ylabel("Dataset-macro MRR")
    left.set_title("Effectiveness vs. runtime")
    left.grid(alpha=0.2, linewidth=0.5)
    left.set_axisbelow(True)

    y = np.arange(len(METHODS))
    for index, method in enumerate(METHODS):
        mean, median, p95 = summaries[method]
        color = METHOD_COLORS[method]
        right.plot([median, p95], [index, index], color=color, linewidth=1.2, zorder=1)
        right.scatter(median, index, marker="o", s=10, color=color, zorder=2)
        right.scatter(p95, index, marker="|", s=32, color=color, linewidth=1.0, zorder=2)
        if p95 < 1:
            value = f"{median:.2f}--{p95:.2f}"
        elif p95 < 10:
            value = f"{median:.2f}--{p95:.2f}"
        elif p95 < 100:
            value = f"{median:.1f}--{p95:.1f}"
        else:
            value = f"{median:.1f}--{p95:.0f}"
        right.text(p95 * 1.10, index, value, va="center", ha="left",
                   fontsize=3.5, color=color)
    right.set_xscale("log")
    right.set_xlim(0.025, 900)
    right.set_yticks([])
    right.set_ylim(-0.65, len(METHODS) - 0.35)
    right.invert_yaxis()
    right.set_xlabel("Latency (s, log)")
    right.set_title("Median--P95 latency")
    right.grid(axis="x", alpha=0.2, linewidth=0.5)
    right.set_axisbelow(True)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3.5,
                      markerfacecolor=METHOD_COLORS[m], markeredgecolor="none",
                      label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), fontsize=4.0,
               columnspacing=0.6, handletextpad=0.25)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.87, bottom=0.38, wspace=0.18)
    save(fig, "efficiency")


def sensitivity() -> None:
    alpha = pd.read_csv(METRICS / "sensitivity_alpha.csv")
    tau = pd.read_csv(METRICS / "sensitivity_tau.csv")
    width = pd.read_csv(METRICS / "sensitivity_m.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.48))
    fusion, gate_quality, gate_activation, refinement_width = axes
    markers = ["o", "^", "s", "D", "v", "P"]

    overall_alpha = alpha[alpha["dataset"] == "ALL"].sort_values("alpha")
    fusion.plot(overall_alpha["alpha"], overall_alpha["Recall@GT"], marker="o",
                markersize=2.4, linewidth=1.4, color=INK, label="Overall")
    for dataset, marker in zip(DATASETS, markers):
        rows = alpha[alpha["dataset"] == dataset].sort_values("alpha")
        fusion.plot(rows["alpha"], rows["Recall@GT"], marker=marker,
                    markersize=2.0, linewidth=0.8, color=DATASET_COLORS[dataset],
                    label=dataset)
    fusion.axvline(0.4, color=DARK_GRAY, linestyle="--", linewidth=0.7)
    fusion.set_xlabel(r"Fusion weight $\alpha$")
    fusion.set_ylabel("Recall@GT")
    fusion.set_title("Fusion weight")

    overall_tau = tau[tau["dataset"] == "ALL"].sort_values("tau")
    tau_x = np.arange(len(overall_tau))
    gate_quality.plot(tau_x, overall_tau["Recall@GT"], marker="o",
                      markersize=2.4, linewidth=1.4, color=INK)
    gate_activation.plot(tau_x, 100 * overall_tau["gate_activation_rate"], marker="o",
                         markersize=2.4, linewidth=1.4, color=INK)
    for dataset, marker in zip(DATASETS, markers):
        rows = tau[tau["dataset"] == dataset].sort_values("tau")
        gate_quality.plot(tau_x, rows["Recall@GT"], marker=marker,
                          markersize=2.0, linewidth=0.8, color=DATASET_COLORS[dataset])
        gate_activation.plot(tau_x, 100 * rows["gate_activation_rate"], marker=marker,
                             markersize=2.0, linewidth=0.8,
                             color=DATASET_COLORS[dataset])
    for ax in [gate_quality, gate_activation]:
        ax.axvline(2, color=DARK_GRAY, linestyle="--", linewidth=0.7)
        ax.set_xticks(tau_x, [".005", ".01", ".02", ".05", ".10"])
        ax.set_xlabel(r"Gate threshold $\tau$")
    gate_quality.set_ylabel("Recall@GT")
    gate_quality.set_title("Gate quality")
    gate_activation.set_ylabel("Activation (%)")
    gate_activation.set_title("Gate activation")

    overall_m = width[(width["scope"] == "overall") &
                      (width["weighting"] == "dataset_macro")].sort_values("m")
    refinement_width.plot(overall_m["m"], overall_m["Hits@1"], marker="o",
                          markersize=2.4, linewidth=1.4, color=INK)
    for dataset, marker in zip(DATASETS, markers):
        rows = width[(width["dataset"] == dataset) &
                     (width["scope"] == "per_dataset")].sort_values("m")
        refinement_width.plot(rows["m"], rows["Hits@1"], marker=marker,
                              markersize=2.0, linewidth=0.8,
                              color=DATASET_COLORS[dataset])
    refinement_width.axvline(3, color=DARK_GRAY, linestyle="--", linewidth=0.7)
    refinement_width.set_xticks([2, 3, 5])
    refinement_width.set_xlabel(r"Refinement width $m$")
    refinement_width.set_ylabel("Hits@1")
    refinement_width.set_title("Refinement width")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
    handles, labels = fusion.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), fontsize=4.3,
               columnspacing=0.7, handlelength=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.83, bottom=0.34, wspace=0.30)
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
    fusion = {name: Counter() for name in ["Overall", *DATASETS]}
    routed = {name: Counter() for name in ["Overall", *DATASETS]}
    bypassed = {name: Counter() for name in ["Overall", *DATASETS]}
    methods = ["dema_decision", "dema_no_rerank", "dema", "dema_always"]
    for path in (PREDICTIONS / "dema").glob("*/*.json"):
        dataset, case_id = path.parent.name, path.stem
        predictions = {method: _load_predictions(method, dataset, case_id) for method in methods}
        for source, targets in _ground_truth(dataset, case_id).items():
            if source not in predictions["dema"]:
                continue
            decision_rr = _rr(predictions["dema_decision"].get(source, []), targets)
            fused_rr = _rr(predictions["dema_no_rerank"].get(source, []), targets)
            outcome = _outcome(decision_rr, fused_rr)
            fusion[dataset][outcome] += 1
            fusion["Overall"][outcome] += 1
            activated = any(row.get("gate_activated", False) for row in predictions["dema"][source])
            if activated:
                outcome = _outcome(fused_rr, _rr(predictions["dema"][source], targets))
                routed[dataset][outcome] += 1
                routed["Overall"][outcome] += 1
            else:
                outcome = _outcome(fused_rr, _rr(predictions["dema_always"].get(source, []), targets))
                bypassed[dataset][outcome] += 1
                bypassed["Overall"][outcome] += 1

    categories, colors = ["Improved", "Unchanged", "Harmed"], [GREEN, GRAY, RED]
    shown = ["Overall", *DATASETS]
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.42), sharey=True)

    def stacked(ax, counters):
        left_edge = np.zeros(len(shown))
        for category, color in zip(categories, colors):
            values = np.array([100 * counters[name][category] / sum(counters[name].values())
                               for name in shown])
            ax.barh(shown, values, left=left_edge, color=color, height=0.64,
                    label=category)
            left_edge += values
        ax.set_xlim(0, 100)
        ax.set_xlabel("Queries (%)")
        ax.grid(axis="x", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    stacked(left, fusion)
    left.set_title("Complementary evidence")
    left.invert_yaxis()

    y = np.arange(len(shown))
    series = [
        (routed, "Improved", GREEN, "o", "Routed: improved", True),
        (routed, "Harmed", RED, "o", "Routed: harmed", True),
        (bypassed, "Improved", GREEN, "s", "Bypassed: improved", False),
        (bypassed, "Harmed", RED, "s", "Bypassed: harmed", False),
    ]
    for counters, category, color, marker, label, filled in series:
        values = []
        for name in shown:
            total = sum(counters[name].values())
            values.append(100 * counters[name][category] / total if total else np.nan)
        right.scatter(values, y, s=13, marker=marker,
                      facecolor=color if filled else "white", edgecolor=color,
                      linewidth=0.8, label=label, zorder=3)
    right.set_xlim(0, 55)
    right.set_xlabel("Changed queries (%)")
    right.grid(axis="x", alpha=0.18, linewidth=0.5)
    right.set_axisbelow(True)
    right.set_title("Refinement outcomes")
    legend_handles = [
        Patch(facecolor=GREEN, label="Improved"),
        Patch(facecolor=GRAY, label="Unchanged"),
        Patch(facecolor=RED, label="Harmed"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.5,
               markerfacecolor=INK, markeredgecolor=INK, label="Routed"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=3.5,
               markerfacecolor="white", markeredgecolor=INK, label="Bypassed replay"),
    ]
    fig.legend(handles=legend_handles, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.60, -0.01), fontsize=3.8,
               columnspacing=0.45, handletextpad=0.25)
    fig.subplots_adjust(left=0.22, right=0.995, top=0.84, bottom=0.37, wspace=0.20)
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
    category_colors = [GREEN, CYAN, GOLD, RED, DARK_GRAY]
    table = counts.pivot(index="dataset", columns="category", values="count").fillna(0)
    shares = table[categories].div(table[categories].sum(axis=1), axis=0) * 100

    ks = [1, 5, 10, 20]
    coverage_counts = {name: {k: 0 for k in ks} for name in ["Overall", *DATASETS]}
    totals = Counter()
    for path in (PREDICTIONS / "dema").glob("*/*.json"):
        dataset, case_id = path.parent.name, path.stem
        predictions = json.loads(path.read_text())["predictions"]
        for source, targets in _ground_truth(dataset, case_id).items():
            by_target = {row["target_column"]: row for row in predictions.get(source, [])}
            for target in targets:
                totals[dataset] += 1
                totals["Overall"] += 1
                rank = by_target.get(target, {}).get("retrieval_rank", 10**9)
                for k in ks:
                    if rank is not None and rank <= k:
                        coverage_counts[dataset][k] += 1
                        coverage_counts["Overall"][k] += 1

    shown = ["Overall", *DATASETS]
    coverage_matrix = np.array([
        [100 * coverage_counts[name][k] / totals[name] for k in ks]
        for name in shown
    ])
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.48))
    cmap = LinearSegmentedColormap.from_list(
        "coverage_coral_indigo", ["#CE6955", "#F4F2ED", "#5C71BC"]
    )
    left.imshow(coverage_matrix, aspect="auto", vmin=35, vmax=100, cmap=cmap)
    left.set_xticks(np.arange(len(ks)), ks)
    left.set_yticks(np.arange(len(shown)), shown)
    for row in range(len(shown)):
        for col in range(len(ks)):
            value = coverage_matrix[row, col]
            use_light_text = value <= 55 or value >= 84
            left.text(col, row, f"{value:.0f}", ha="center", va="center",
                      fontsize=4.2, color="white" if use_light_text else INK)
    left.set_xlabel("Candidate limit $k$")
    left.set_title("Candidate coverage (%)")

    y = np.arange(len(shown))
    left_edge = np.zeros(len(shown))
    for category, label, color in zip(categories, category_labels, category_colors):
        values = shares.reindex(shown)[category].to_numpy()
        right.barh(y, values, left=left_edge, color=color, height=0.64, label=label)
        left_edge += values
    right.invert_yaxis()
    right.set_yticks(y, [])
    right.set_xlim(0, 100)
    right.set_xlabel("GT pairs (%)")
    right.set_title("Stage-level errors")
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.66, -0.01), fontsize=3.7,
               columnspacing=0.4, handlelength=0.8, handletextpad=0.25)
    fig.subplots_adjust(left=0.22, right=0.995, top=0.84, bottom=0.37, wspace=0.12)
    save(fig, "diagnostics")


if __name__ == "__main__":
    headline_results()
    efficiency()
    sensitivity()
    mechanism_analysis()
    diagnostics()
