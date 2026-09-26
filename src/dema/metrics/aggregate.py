"""Dataset-level and overall aggregation.

Weighting rules (every aggregated row states its rule in ``weighting``):

* ``per_dataset.csv`` — ``case_macro``: unweighted mean over the evaluated cases
  of the dataset (each case counts once, regardless of its number of columns).
* ``overall.csv`` has two rows per method:
    - ``case_macro``: unweighted mean over *all* evaluated cases of all datasets
      (large datasets such as ChEMBL/OpenData/TPC-DI dominate);
    - ``dataset_macro``: unweighted mean of the per-dataset means (each dataset
      counts once).
  Micro-averaging (pooling columns across cases) is not used anywhere.

``complete`` is False when a method has unevaluated (missing/failed) cases; the
numbers are then computed over the evaluated subset only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES


def _runtime_stats(group: pd.DataFrame) -> dict[str, float]:
    total = pd.to_numeric(group["total_seconds"], errors="coerce").dropna()
    rerank = pd.to_numeric(group["reranking_seconds"], errors="coerce").dropna()
    return {
        "mean_runtime": float(total.mean()) if len(total) else np.nan,
        "median_runtime": float(total.median()) if len(total) else np.nan,
        "p95_runtime": float(np.percentile(total, 95)) if len(total) else np.nan,
        "mean_reranking_runtime": float(rerank.mean()) if len(rerank) else np.nan,
        "total_model_requests": float(pd.to_numeric(group["model_requests"], errors="coerce").sum()),
        "total_retries": float(pd.to_numeric(group["retries"], errors="coerce").sum()),
    }


def aggregate_per_dataset(per_case: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exp = {(r.method, r.dataset): int(r.n_cases_expected) for r in expected.itertuples()}
    for (method, dataset), n_expected in sorted(exp.items()):
        group = per_case[(per_case["method"] == method) & (per_case["dataset"] == dataset)] if len(per_case) else per_case
        row = {"method": method, "dataset": dataset, "weighting": "case_macro",
               "n_cases": int(len(group)), "n_cases_expected": n_expected,
               "complete": bool(len(group) == n_expected)}
        for m in METRIC_NAMES:
            row[m] = float(group[m].mean()) if len(group) else np.nan
        if len(group):
            row.update(_runtime_stats(group))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_overall(per_case: pd.DataFrame, per_dataset: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(expected["method"].unique()):
        n_expected = int(expected[expected["method"] == method]["n_cases_expected"].sum())
        cases = per_case[per_case["method"] == method] if len(per_case) else per_case
        dsets = per_dataset[(per_dataset["method"] == method) & (per_dataset["n_cases"] > 0)]
        base = {"method": method, "n_cases": int(len(cases)), "n_cases_expected": n_expected,
                "n_datasets": int(len(dsets)), "complete": bool(len(cases) == n_expected)}
        case_row = {**base, "weighting": "case_macro"}
        ds_row = {**base, "weighting": "dataset_macro"}
        for m in METRIC_NAMES:
            case_row[m] = float(cases[m].mean()) if len(cases) else np.nan
            ds_row[m] = float(dsets[m].mean()) if len(dsets) else np.nan
        if len(cases):
            case_row.update(_runtime_stats(cases))
        rows += [case_row, ds_row]
    return pd.DataFrame(rows)
