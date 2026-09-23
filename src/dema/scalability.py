"""Scalability runner: runtime as the target-schema width grows.

    python -m dema.scalability [--config configs/scalability.yaml] [--methods ...]

For every (method, dataset, case, target_size, repeat) unit the target table is
column-subsampled deterministically:

* every target column referenced by the case's ground truth is kept;
* the remaining slots are filled with other target columns drawn with a seed
  derived from (seed, case_id, target_size, repeat);
* the original column order is preserved; rows and values are untouched.

A unit is skipped and recorded when the target has fewer columns than
``target_size`` (``insufficient_columns``) or when the ground-truth columns alone
exceed it (``ground_truth_exceeds_size``). Each repetition is stored separately::

    outputs/scalability/raw/<method>/<dataset>/<case_id>/t<size>_r<rep>.json
    outputs/scalability/predictions/<method>/<dataset>/<case_id>/t<size>_r<rep>.json

and ``outputs/scalability/summary.csv`` aggregates them.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .data.loader import load_case
from .data.manifest import read_manifest, select_cases
from .evaluation.metrics import all_metrics
from .models.base import CaseContext
from .models.ranking import validate_ranking
from .models.registry import build_matcher
from .runner import now, prediction_document, run_manifest
from .utils.config import load_config, resolve_path
from .utils.device import set_global_seed
from .utils.io import atomic_write_json, read_json
from .utils.logging import get_console_logger

log = get_console_logger("dema.scalability")


def unit_seed(seed: int, case_id: str, size: int, rep: int) -> int:
    digest = hashlib.sha256(f"{seed}|{case_id}|{size}|{rep}".encode()).hexdigest()
    return int(digest[:12], 16)


def subsample_target_columns(
    target_columns: list[str], required: set[str], size: int, seed: int
) -> list[str] | None:
    """Deterministic subset of ``size`` columns containing ``required`` (order kept)."""
    if size > len(target_columns):
        return None
    required = [c for c in target_columns if c in required]
    if len(required) > size:
        return None
    others = [c for c in target_columns if c not in set(required)]
    rng = random.Random(seed)
    chosen = set(required) | set(rng.sample(others, size - len(required)))
    return [c for c in target_columns if c in chosen]


def summarize(raw_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_root.rglob("*.json")):
        rec = read_json(path)
        if rec.get("status") != "success":
            continue
        rows.append({k: rec[k] for k in ("method", "dataset", "case_id", "target_size", "repeat")}
                    | {"total_seconds": rec["runtime"]["total_seconds"],
                       "reranking_seconds": rec["runtime"]["reranking_seconds"],
                       "MRR": rec["metrics"]["MRR"], "Recall@GT": rec["metrics"]["Recall@GT"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "dataset", "target_size"]).agg(
        n_units=("total_seconds", "size"),
        mean_runtime=("total_seconds", "mean"),
        median_runtime=("total_seconds", "median"),
        p95_runtime=("total_seconds", lambda s: float(np.percentile(s, 95))),
        mean_reranking_runtime=("reranking_seconds", "mean"),
        MRR=("MRR", "mean"),
        **{"Recall@GT": ("Recall@GT", "mean")},
    ).reset_index()
    return agg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/scalability.yaml")
    p.add_argument("--config-dir", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)

    config = load_config(config_dir=args.config_dir)
    from .utils.config import load_yaml

    scfg = load_yaml(resolve_path(args.config))
    out_root = config.outputs_dir / "scalability"
    raw_root = out_root / "raw"
    if args.summarize_only:
        summarize(raw_root).to_csv(out_root / "summary.csv", index=False)
        return 0

    seed = int(scfg.get("seed", 42))
    resume = bool(scfg.get("resume", True)) and not args.overwrite
    abort = scfg.get("on_failure", "continue") == "abort"
    methods = args.methods or list(scfg["methods"])
    records = select_cases(read_manifest(config.manifest_path), scfg["datasets"])
    limit = scfg.get("max_cases_per_dataset")
    if limit:
        kept, seen = [], {}
        for r in records:
            if seen.get(r.dataset, 0) < int(limit):
                kept.append(r)
                seen[r.dataset] = seen.get(r.dataset, 0) + 1
        records = kept

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-scalability"
    manifest = run_manifest(config, run_id, methods, list(scfg["datasets"]), sys.argv)
    manifest["scalability_config"] = scfg
    atomic_write_json(config.outputs_dir / "manifests" / f"{run_id}.json", manifest)

    failed = 0
    for method in methods:
        matcher = None
        for record in records:
            case = None
            for size in scfg["target_sizes"]:
                for rep in range(int(scfg["repeats"])):
                    raw_path = raw_root / method / record.dataset / record.case_id / f"t{size}_r{rep}.json"
                    if resume and raw_path.is_file() and read_json(raw_path).get("status") in (
                        "success", "insufficient_columns", "ground_truth_exceeds_size"
                    ):
                        continue
                    if case is None:
                        case = load_case(record)
                    tcols = [str(c) for c in case.target_df.columns]
                    required = {t for _, t in case.ground_truth if t in set(tcols)}
                    useed = unit_seed(seed, record.case_id, int(size), rep)
                    subset = subsample_target_columns(tcols, required, int(size), useed)
                    base = {"method": method, "dataset": record.dataset, "case_id": record.case_id,
                            "target_size": int(size), "repeat": rep, "seed": useed, "run_id": run_id,
                            "n_target_columns_available": len(tcols)}
                    if subset is None:
                        reason = ("insufficient_columns" if int(size) > len(tcols)
                                  else "ground_truth_exceeds_size")
                        atomic_write_json(raw_path, {**base, "status": reason})
                        continue
                    if matcher is None:
                        matcher = build_matcher(method, config)
                        matcher.load()
                    target_df = case.target_df[subset]
                    unit_id = f"{record.case_id}__t{size}_r{rep}"
                    try:
                        set_global_seed(seed)
                        matcher.set_case_context(CaseContext(record.dataset, unit_id))
                        matches = matcher.match(case.source_df, target_df)
                        validate_ranking(matches, [str(c) for c in case.source_df.columns], subset)
                        doc = prediction_document(record, method, matches, run_id, subset)
                        doc["case_id"], doc["target_columns"] = unit_id, subset
                        atomic_write_json(out_root / "predictions" / method / record.dataset /
                                          record.case_id / f"t{size}_r{rep}.json", doc)
                        metrics = all_metrics(
                            {s: [(e["target_column"], e["score"]) for e in es] for s, es in doc["predictions"].items()},
                            [(s, t) for s, t in case.ground_truth],
                        )
                        atomic_write_json(raw_path, {**base, "status": "success", "finished_at": now(),
                                                     "runtime": matcher.last_runtime.to_dict(), "metrics": metrics})
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        atomic_write_json(raw_path, {**base, "status": "failed",
                                                     "error": f"{type(exc).__name__}: {exc}",
                                                     "traceback": traceback.format_exc()})
                        log.error("%s %s t=%s r=%s failed: %s", method, record.case_id, size, rep, exc)
                        if abort:
                            return 1
                    finally:
                        matcher.set_case_context(None)
                    log.info("%s %s/%s t=%s r=%s done", method, record.dataset, record.case_id, size, rep)
    summary = summarize(raw_root)
    summary.to_csv(out_root / "summary.csv", index=False)
    log.info("scalability summary written (%d rows); %d failed unit(s)", len(summary), failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
