"""End-to-end smoke test (``scripts/smoke_test.sh``).

Runs the methods listed in ``experiment.smoke_test`` on one GDC and one
Valentine case, writing to ``outputs/smoke`` so formal outputs are untouched,
and verifies every pipeline stage. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import sys

from .data.loader import load_case
from .data.manifest import read_manifest
from .evaluation.evaluator import evaluate
from .models.registry import build_matcher
from .preflight import check_decision, check_qwen
from .representation.profiler import profile_table
from .runner import OutputPaths, run, unit_state
from .utils.config import Config, load_config
from .utils.io import read_json
from .utils.logging import get_console_logger

log = get_console_logger("dema.smoke")


def smoke_config(config: Config) -> Config:
    cfg = copy.deepcopy(config)
    cfg.paths["outputs"] = str(config.outputs_dir / "smoke")
    cfg.experiment["on_failure"] = "continue"
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    args = p.parse_args(argv)
    base = load_config(args.config, args.config_dir)
    spec = base.experiment["smoke_test"]
    methods = args.methods or list(spec["methods"])
    config = smoke_config(base)
    shutil.rmtree(config.outputs_dir, ignore_errors=True)

    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        log.info("%s %s", "PASS" if ok else "FAIL", what)
        if not ok:
            failures.append(what)

    records = read_manifest(config.manifest_path)
    selected = []
    for item in spec["cases"]:
        rows = [r for r in records if r.dataset == item["dataset"]]
        check(len(rows) > int(item["index"]), f"benchmark has case {item['dataset']}[{item['index']}]")
        if len(rows) > int(item["index"]):
            selected.append(rows[int(item["index"])])
    if failures:
        return 1

    for record in selected:
        case = load_case(record)
        check(case.source_df.shape[1] == record.n_source_columns
              and case.target_df.shape[1] == record.n_target_columns, f"load {record.case_id}")
        profiles = profile_table(case.source_df, config.section("representation"))
        again = profile_table(case.source_df, config.section("representation"))
        check(profiles == again and len(profiles) == record.n_source_columns,
              f"deterministic representation {record.case_id}")

    if "magneto_qwen" in methods:
        check(not check_qwen(config), "Qwen endpoint call")
    if "dema" in methods:
        check(not check_decision(config), "decision-model endpoint call")
    if failures:
        return 1

    paths = OutputPaths(config.outputs_dir)
    for method in methods:
        matcher = build_matcher(method, config)
        by_dataset: dict[str, list[str]] = {}
        for r in selected:
            by_dataset.setdefault(r.dataset, []).append(r.case_id)
        for dataset, ids in by_dataset.items():
            counts = run(config, method, [dataset], ids, resume=False, matcher=matcher,
                         run_id=f"smoke-{method}", argv=["smoke"])
            check(counts["failed"] == 0, f"{method} ran on {dataset}")
        for r in selected:
            check(unit_state(paths, method, r) == "complete", f"{method} complete prediction {r.case_id}")
            rt = read_json(paths.runtime(method, r.dataset, r.case_id))
            check(rt["total_seconds"] >= 0, f"{method} runtime written {r.case_id}")
            if method in ("magneto_qwen", "dema"):
                check(rt["model_requests"] >= r.n_source_columns and rt["retrieval_seconds"] > 0,
                      f"{method} retrieval + reranking recorded {r.case_id}")
                doc = read_json(paths.prediction(method, r.dataset, r.case_id))
                k = min(int(config.section("retriever")["top_k"]), r.n_target_columns)
                reranked = all(
                    all("reranker_score" in e for e in entries[:k]) for entries in doc["predictions"].values()
                )
                check(reranked, f"{method} top-k carries reranker scores {r.case_id}")

    if {"magneto_qwen", "dema"} <= set(methods):
        # Same candidate sets for both rerankers (shared retriever + cache).
        for r in selected:
            a = read_json(paths.prediction("magneto_qwen", r.dataset, r.case_id))["predictions"]
            b = read_json(paths.prediction("dema", r.dataset, r.case_id))["predictions"]
            k = min(int(config.section("retriever")["top_k"]), r.n_target_columns)
            same = all({e["target_column"] for e in a[s][:k]} == {e["target_column"] for e in b[s][:k]} for s in a)
            check(same, f"identical candidate sets magneto_qwen/dema {r.case_id}")

    tables = evaluate(config, methods, sorted({r.dataset for r in selected}))
    per_case = tables["per_case"]
    ok = len(per_case) == len(methods) * len(selected) and per_case["MRR"].apply(
        lambda v: not math.isnan(v)).all()
    check(bool(ok), "evaluation produced finite metrics for every unit")

    if failures:
        log.error("smoke test FAILED: %s", failures)
        return 1
    log.info("smoke test passed (outputs in %s)", config.outputs_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
