"""Generic experiment runner.

    python -m dema.experiments.runner --method dema --dataset GDC
    python -m dema.experiments.runner --method coma --dataset OpenData --case-id <id> --overwrite

The atomic execution unit is ``method x dataset x case``. For each unit the
runner writes, in this order (each file atomically)::

    saves/predictions/<method>/<dataset>/<case_id>.json   complete ranking
    saves/runtime/<method>/<dataset>/<case_id>.json       runtime record
    saves/status/<method>/<dataset>/<case_id>.json        success/failed marker

A unit is *complete* only if its status is ``success``, the prediction file hash
matches the status marker and the prediction contains a full ranking for every
source column. With ``--resume`` complete units are skipped and everything else
(missing, partial, failed) is re-run. Failed units are never marked successful
and never fall back to retrieval scores.

The matcher (and its models) is built once per process. Each invocation writes a
run manifest to ``saves/manifests/<run_id>.json`` and case logs to ``logs/cases``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from ..data.loader import load_case
from ..data.manifest import CaseRecord, read_manifest, select_cases, spread_per_dataset
from ..data.types import Match
from ..model.base import BaseMatcher, CaseContext
from ..model.ranking import group_by_source, validate_ranking
from ..model.registry import ALL_METHODS, DEMA_METHODS, build_matcher
from ..model.representation.serialization import representation_signature
from ..utils.config import Config, load_config
from ..utils.device import set_global_seed
from ..utils.git import git_commit, git_dirty
from ..utils.io import atomic_write_json, read_json, sha256_file
from ..utils.logging import CaseLogger, get_console_logger

log = get_console_logger("dema.experiments.runner")
RUN_SCHEMA_VERSION = 2


# ------------------------------------------------------------------- paths
class ExperimentPaths:
    """Typed locations for durable saves and diagnostic logs.

    Keeping these roots separate prevents logs and derived reports from being
    mistaken for resumable experiment state.
    """

    def __init__(self, saves_root: Path, logs_root: Path):
        self.saves_root = Path(saves_root)
        self.logs_root = Path(logs_root)

    @classmethod
    def from_config(cls, config: Config) -> "ExperimentPaths":
        return cls(config.saves_dir, config.logs_dir)

    def _p(self, kind: str, method: str, dataset: str, case_id: str, suffix: str) -> Path:
        return self.saves_root / kind / method / dataset / f"{case_id}{suffix}"

    def prediction(self, m, d, c) -> Path:
        return self._p("predictions", m, d, c, ".json")

    def runtime(self, m, d, c) -> Path:
        return self._p("runtime", m, d, c, ".json")

    def status(self, m, d, c) -> Path:
        return self._p("status", m, d, c, ".json")

    def log(self, m, d, c) -> Path:
        return self.logs_root / "cases" / m / d / f"{c}.log"

    def debug(self, run_id, m, d, c) -> Path:
        return self.saves_root / "debug" / run_id / m / d / f"{c}.jsonl"

    def run_manifest(self, run_id) -> Path:
        return self.saves_root / "manifests" / f"{run_id}.json"


# --------------------------------------------------------------- utilities
def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def prediction_document(
    record: CaseRecord, method: str, matches: list[Match], run_id: str, target_columns: list[str],
    experiment_fingerprint: str | None = None,
) -> dict[str, Any]:
    grouped = group_by_source(matches)
    predictions = {}
    for src, items in grouped.items():
        entries = []
        for m in items:
            entry: dict[str, Any] = {
                "target_column": m.target_column,
                # Keep score/rank for readers of the v2 prediction schema.
                "score": m.score,
                "rank": m.rank,
                "final_score": m.score,
                "final_rank": m.rank,
            }
            if m.reranker_score is not None:
                entry["reranker_score"] = m.reranker_score
            if m.retrieval_score is not None:
                entry["retrieval_score"] = m.retrieval_score
            for field in (
                "retrieval_rank", "jev_score", "coma_plus_score", "fusion_score",
                "fusion_rank", "jina_applied", "jina_score", "jina_rank",
                "gate_activated", "gate_disagreement", "gate_margin", "gate_threshold",
            ):
                value = getattr(m, field)
                if value is not None:
                    entry[field] = value
            entries.append(entry)
        predictions[src] = entries
    return {
        "dataset": record.dataset,
        "case_id": record.case_id,
        "method": method,
        "run_id": run_id,
        "created_at": now(),
        "experiment_fingerprint": experiment_fingerprint,
        "n_source_columns": len(predictions),
        "n_target_columns": len(target_columns),
        "predictions": predictions,
    }


def prediction_is_complete(path: Path, record: CaseRecord) -> bool:
    """Structural completeness check of a saved prediction file."""
    try:
        doc = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    preds = doc.get("predictions")
    if not isinstance(preds, dict) or len(preds) != record.n_source_columns:
        return False
    for entries in preds.values():
        if len(entries) != record.n_target_columns:
            return False
        if [e.get("rank") for e in entries] != list(range(1, record.n_target_columns + 1)):
            return False
        if len({e.get("target_column") for e in entries}) != record.n_target_columns:
            return False
    return True


def unit_state(
    paths: ExperimentPaths, method: str, record: CaseRecord, expected_fingerprint: str | None = None
) -> str:
    """``complete`` | ``failed`` | ``incomplete`` | ``stale`` | ``missing``."""
    status_path = paths.status(method, record.dataset, record.case_id)
    pred_path = paths.prediction(method, record.dataset, record.case_id)
    if not status_path.is_file():
        return "incomplete" if pred_path.exists() else "missing"
    try:
        status = read_json(status_path)
    except (OSError, json.JSONDecodeError):
        return "incomplete"
    if expected_fingerprint and status.get("experiment_fingerprint") != expected_fingerprint:
        return "stale"
    if status.get("status") != "success":
        return "failed"
    if not pred_path.is_file() or sha256_file(pred_path) != status.get("prediction_sha256"):
        return "incomplete"
    if not paths.runtime(method, record.dataset, record.case_id).is_file():
        return "incomplete"
    return "complete" if prediction_is_complete(pred_path, record) else "incomplete"


def method_fingerprint(config: Config, method: str) -> str:
    """Hash semantic settings so resume never reuses results from old configs."""
    if method in DEMA_METHODS:
        settings = {name: config.models[name] for name in (
            "representation", "retriever", "reranking", "decision"
        )}
        settings["variant"] = method
        if method in (
            "dema", "dema_always", "dema_gate_m2", "dema_gate_m5",
            "dema_no_rerank", "dema_no_struct", "dema_decision",
            "dema_fusion", "dema_jev_weight", "dema_jina_rerank", "dema_jina_no_coma",
            "dema_shared", "dema_single"
        ):
            settings["magneto_candidates"] = config.models["magneto"]
        if method in (
            "dema", "dema_always", "dema_gate_m2", "dema_gate_m5",
            "dema_no_rerank", "dema_fusion", "dema_jev_weight", "dema_jina_rerank"
        ):
            settings["fusion"] = config.models["fusion"]
            settings["coma_plus"] = config.models["baselines"]["coma_plus"]
        if method == "dema_jev_weight":
            settings["dynamic_weight"] = config.models["dynamic_weight"]
        if method in ("dema", "dema_always", "dema_gate_m2", "dema_gate_m5", "dema_jina_rerank"):
            settings["jina_rerank"] = config.models["jina_rerank"]
        if method in ("dema", "dema_gate_m2", "dema_gate_m5"):
            settings["selective_refinement"] = config.models["selective_refinement"]
        if method in ("dema_no_struct", "dema_jina_no_coma"):
            settings["jina_rerank"] = config.models["jina_rerank"]
    elif method == "magneto_qwen":
        settings = {name: config.models[name] for name in ("magneto", "qwen")}
    else:
        settings = config.models.get("baselines", {}).get(method, {})
    payload = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "method": method,
        "settings": settings,
        "seed": config.experiment.get("seed", 42),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def selected_records(
    config: Config,
    datasets: list[str],
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[CaseRecord]:
    """Apply the runner's dataset/case/dev-limit selection in one place."""
    records = select_cases(read_manifest(config.manifest_path), datasets, case_ids)
    if case_ids is None:
        records = spread_per_dataset(records, config.experiment.get("max_cases_per_dataset"))
    if limit is not None:
        per_dataset: dict[str, int] = {}
        kept = []
        for record in records:
            if per_dataset.get(record.dataset, 0) < limit:
                kept.append(record)
                per_dataset[record.dataset] = per_dataset.get(record.dataset, 0) + 1
        records = kept
    return records


def has_pending_work(
    config: Config,
    method: str,
    datasets: list[str],
    case_ids: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> bool:
    """Whether running a method will perform inference (used before server startup)."""
    records = selected_records(config, datasets, case_ids, limit)
    if not resume:
        return bool(records)
    paths = ExperimentPaths.from_config(config)
    fingerprint = method_fingerprint(config, method)
    return any(unit_state(paths, method, record, fingerprint) != "complete" for record in records)


def record_external_failure(
    config: Config,
    method: str,
    record: CaseRecord,
    run_id: str,
    error: str,
    started_at: str,
    exit_code: int,
) -> None:
    """Record a case worker that died outside Python's exception handling.

    A SIGKILL or parent-enforced timeout cannot be handled by ``run_unit``.
    The supervising process calls this after observing the worker exit so the
    case is failed, rather than silently missing, and resume can retry it.
    """
    paths = ExperimentPaths.from_config(config)
    status_path = paths.status(method, record.dataset, record.case_id)
    if status_path.is_file():
        try:
            current = read_json(status_path)
        except (OSError, json.JSONDecodeError):
            current = {}
        # Preserve the more specific traceback written by a worker that exited
        # normally with a handled case failure.
        if current.get("run_id") == run_id and current.get("status") == "failed":
            return
    atomic_write_json(status_path, {
        "status": "failed",
        "dataset": record.dataset,
        "case_id": record.case_id,
        "method": method,
        "run_id": run_id,
        "experiment_fingerprint": method_fingerprint(config, method),
        "started_at": started_at,
        "finished_at": now(),
        "error": error,
        "exit_code": exit_code,
        "retries": None,
        "failures": None,
    })
    with CaseLogger(paths.log(method, record.dataset, record.case_id)) as clog:
        clog.error("external worker failure: %s", error)
        clog.info("end status=failed")

    manifest_path = paths.run_manifest(run_id)
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            return
        manifest.update(result={"skipped": 0, "success": 0, "failed": 1}, finished_at=now())
        atomic_write_json(manifest_path, manifest)


class DebugRecorder:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------- manifest
def run_manifest(config: Config, run_id: str, methods: list[str], datasets: list[str], argv: list[str]) -> dict:
    models = config.models
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:  # pragma: no cover
        torch_version = None
    return {
        "run_id": run_id,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp": now(),
        "command": argv,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch_version,
        "seed": config.experiment.get("seed"),
        "datasets": datasets,
        "methods": methods,
        "embedding_model": models["retriever"]["model"],
        "qwen_model": models["qwen"]["model"],
        "decision_model": models["decision"]["model"],
        "decision_backend": models["decision"].get("backend"),
        "top_k": models["retriever"]["top_k"],
        "representation_config": representation_signature(models["representation"]),
        "config": config.as_dict(),
    }


# --------------------------------------------------------------------- core
def run_unit(
    matcher: BaseMatcher,
    method: str,
    record: CaseRecord,
    paths: ExperimentPaths,
    run_id: str,
    seed: int,
    experiment_fingerprint: str,
    debug_path: Path | None = None,
) -> bool:
    """Run one method x case. Returns True on success."""
    d, c = record.dataset, record.case_id
    pred_path, rt_path, status_path = paths.prediction(method, d, c), paths.runtime(method, d, c), paths.status(method, d, c)
    with CaseLogger(paths.log(method, d, c)) as clog:
        started = now()
        clog.info("start run_id=%s dataset=%s case_id=%s method=%s", run_id, d, c, method)
        clog.info("model=%s", json.dumps(matcher.describe(), default=str))
        # Remove a previous success marker first: a crash mid-unit must never
        # leave an old "success" next to new partial outputs.
        if status_path.exists():
            status_path.unlink()
        try:
            case = load_case(record)
            clog.info("n_source_columns=%d n_target_columns=%d n_ground_truth=%d",
                      case.source_df.shape[1], case.target_df.shape[1], len(case.ground_truth))
            set_global_seed(seed)
            matcher.set_case_context(CaseContext(d, c))
            matcher.debug_sink = DebugRecorder(debug_path) if debug_path else None
            wall = time.perf_counter()
            matches = matcher.match(case.source_df, case.target_df)
            wall = time.perf_counter() - wall
            source_cols = [str(x) for x in case.source_df.columns]
            target_cols = [str(x) for x in case.target_df.columns]
            validate_ranking(matches, source_cols, target_cols)
            runtime = matcher.last_runtime.to_dict()
            runtime["wall_seconds"] = wall
            doc = prediction_document(record, method, matches, run_id, target_cols, experiment_fingerprint)
            atomic_write_json(pred_path, doc)
            atomic_write_json(rt_path, {"dataset": d, "case_id": c, "method": method, "run_id": run_id, **runtime})
            atomic_write_json(status_path, {
                "status": "success", "dataset": d, "case_id": c, "method": method, "run_id": run_id,
                "experiment_fingerprint": experiment_fingerprint,
                "started_at": started, "finished_at": now(), "prediction_sha256": sha256_file(pred_path),
            })
            clog.info("retries=%d failures=%d model_requests=%d total_seconds=%.3f",
                      runtime["retries"], runtime["failures"], runtime["model_requests"], runtime["total_seconds"])
            clog.info("end status=success")
            return True
        except Exception as exc:  # noqa: BLE001 — every failure is recorded
            tb = traceback.format_exc()
            clog.error("exception: %s\n%s", exc, tb)
            stats = getattr(matcher, "last_runtime", None)
            atomic_write_json(status_path, {
                "status": "failed", "dataset": d, "case_id": c, "method": method, "run_id": run_id,
                "experiment_fingerprint": experiment_fingerprint,
                "started_at": started, "finished_at": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "retries": getattr(stats, "retries", None), "failures": getattr(stats, "failures", None),
            })
            clog.info("end status=failed")
            return False
        finally:
            matcher.set_case_context(None)
            matcher.debug_sink = None


def run(
    config: Config,
    method: str,
    datasets: list[str],
    case_ids: list[str] | None = None,
    resume: bool = True,
    save_debug: bool = False,
    run_id: str | None = None,
    limit: int | None = None,
    use_cache: bool = True,
    argv: list[str] | None = None,
    matcher: BaseMatcher | None = None,
) -> dict[str, int]:
    run_id = run_id or f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{method}-{uuid.uuid4().hex[:6]}"
    paths = ExperimentPaths.from_config(config)
    fingerprint = method_fingerprint(config, method)
    records = selected_records(config, datasets, case_ids, limit)
    manifest = run_manifest(config, run_id, [method], sorted({r.dataset for r in records}), argv or sys.argv)
    manifest["n_cases"] = len(records)
    manifest["experiment_fingerprint"] = fingerprint
    atomic_write_json(paths.run_manifest(run_id), manifest)

    todo = []
    counts = {"skipped": 0, "success": 0, "failed": 0}
    for r in records:
        if resume and unit_state(paths, method, r, fingerprint) == "complete":
            counts["skipped"] += 1
        else:
            todo.append(r)
    log.info("[%s] %d case(s) selected, %d already complete, %d to run (run_id=%s)",
             method, len(records), counts["skipped"], len(todo), run_id)
    if not todo:
        manifest.update(result=counts, finished_at=now())
        atomic_write_json(paths.run_manifest(run_id), manifest)
        return counts

    if matcher is None:
        matcher = build_matcher(method, config, use_cache=use_cache)
    load_start = time.perf_counter()
    matcher.load()  # once per process; excluded from per-case runtime
    manifest["model_load_seconds"] = time.perf_counter() - load_start
    manifest["matcher"] = matcher.describe()

    seed = int(config.experiment.get("seed", 42))
    debug_budget = int(config.experiment.get("debug_max_cases", 3)) if save_debug else 0
    abort = config.experiment.get("on_failure", "continue") == "abort"
    for i, record in enumerate(todo, start=1):
        debug_path = paths.debug(run_id, method, record.dataset, record.case_id) if debug_budget > 0 else None
        debug_budget -= 1 if debug_path else 0
        ok = run_unit(matcher, method, record, paths, run_id, seed, fingerprint, debug_path)
        counts["success" if ok else "failed"] += 1
        log.info("[%s] %d/%d %s/%s %s", method, i, len(todo), record.dataset, record.case_id,
                 "ok" if ok else "FAILED (see " + str(paths.log(method, record.dataset, record.case_id)) + ")")
        if not ok and abort:
            log.error("on_failure=abort: stopping after failed case")
            break
    manifest.update(result=counts, finished_at=now())
    atomic_write_json(paths.run_manifest(run_id), manifest)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=ALL_METHODS)
    p.add_argument("--dataset", nargs="+", required=True, help="dataset name(s), or 'all'")
    p.add_argument("--case-id", nargs="+", default=None)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", dest="resume", action="store_true", default=None,
                       help="skip complete cases (default from experiment.yaml)")
    group.add_argument("--overwrite", dest="resume", action="store_false", help="re-run every selected case")
    p.add_argument("--config", default=None, help="experiment YAML (default configs/experiment.yaml)")
    p.add_argument("--config-dir", default=None)
    p.add_argument("--save-debug", action="store_true",
                   help="store model requests/responses for up to experiment.debug_max_cases cases")
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="at most N cases per dataset")
    p.add_argument("--no-cache", action="store_true", help="disable the candidate cache")
    p.add_argument("--fail-on-error", action="store_true", help="exit non-zero if any case failed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.config_dir)
    datasets = list(config.experiment["datasets"]) if args.dataset == ["all"] else args.dataset
    resume = bool(config.experiment.get("resume", True)) if args.resume is None else args.resume
    counts = run(
        config, args.method, datasets, args.case_id, resume=resume, save_debug=args.save_debug,
        run_id=args.run_id, limit=args.limit, use_cache=not args.no_cache,
        argv=["python", "-m", "dema.experiments.runner", *(argv if argv is not None else sys.argv[1:])],
    )
    log.info("[%s] done: %s", args.method, counts)
    if counts["failed"] and (args.fail_on_error or config.experiment.get("on_failure") == "abort"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
