"""Simple top-level command line interface for DeMa experiments."""

from __future__ import annotations

import argparse
import os
import resource
import signal
import subprocess
import sys
import time
import uuid

from .experiments.preflight import check_benchmark, check_outputs
from .experiments.runner import (
    ExperimentPaths,
    has_pending_work,
    method_fingerprint,
    now,
    record_external_failure,
    selected_records,
    unit_state,
)
from .experiments.services import ManagedService, ServiceError, required_service
from .model.registry import ALL_METHODS, DEMA_METHODS
from .utils.config import ConfigError, load_config
from .utils.logging import get_console_logger

log = get_console_logger("dema.cli")


def _gpu_plan(args: argparse.Namespace, method: str) -> tuple[str, str | None]:
    """Return ``(service_gpu, worker_gpu)`` for one method.

    ``--gpus SERVICE [WORKER]`` is the unified interface.  With one id the
    service and in-process model share that GPU; with two ids they are split.
    The older service-specific flags remain compatible and, for methods with a
    service, now also constrain the worker to the same GPU instead of silently
    falling back to physical GPU 0.
    """
    if args.gpus:
        service_gpu = args.gpus[0]
        worker_gpu = args.gpus[-1]
        return service_gpu, worker_gpu
    if method == "magneto_qwen":
        return str(args.qwen_gpu), str(args.qwen_gpu)
    if method in DEMA_METHODS:
        return str(args.decision_gpu), str(args.decision_gpu)
    return str(args.qwen_gpu), None


def _worker_env(gpu: str | None) -> dict[str, str] | None:
    if gpu is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run(
    command: list[str],
    timeout: float | None = None,
    memory_gb: float | None = None,
    env: dict[str, str] | None = None,
) -> int:
    log.info("running: %s", " ".join(command))
    limit_bytes = int(memory_gb * 1024**3) if memory_gb and memory_gb > 0 else None

    def set_limits() -> None:
        if limit_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    process = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        preexec_fn=set_limits if limit_bytes is not None else None,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.error("command timed out after %.0f seconds; terminating process group", timeout)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124
    if returncode:
        if returncode < 0:
            try:
                reason = signal.Signals(-returncode).name
            except ValueError:
                reason = f"signal {-returncode}"
            log.error("command terminated by %s", reason)
        else:
            log.error("command exited with code %d", returncode)
    return returncode


def _runner_command(
    args: argparse.Namespace,
    method: str,
    datasets: list[str],
    case_ids: list[str] | None = None,
    run_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dema.experiments.runner",
        "--method",
        method,
        "--dataset",
        *datasets,
        "--config",
        args.config,
        "--fail-on-error",
    ]
    selected_case_ids = args.case_id if case_ids is None else case_ids
    if selected_case_ids:
        command += ["--case-id", *selected_case_ids]
    if args.limit is not None and case_ids is None:
        command += ["--limit", str(args.limit)]
    if run_id:
        command += ["--run-id", run_id]
    if args.config_dir:
        command += ["--config-dir", args.config_dir]
    command += ["--overwrite" if args.overwrite else "--resume"]
    if args.save_debug:
        command.append("--save-debug")
    if args.no_cache:
        command.append("--no-cache")
    return command


def _external_failure_reason(returncode: int, timeout: float, memory_gb: float) -> str:
    if returncode == 124:
        return f"case worker exceeded the {timeout:g}s timeout"
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        suffix = f" under a {memory_gb:g} GiB address-space limit" if memory_gb > 0 else ""
        return f"case worker terminated by {name}{suffix}"
    return f"case worker exited with code {returncode} before recording a failure"


def _run_distribution_isolated(args: argparse.Namespace, config, datasets: list[str]) -> int:
    """Run risky Distribution MILPs one case per supervised process."""
    records = selected_records(config, datasets, args.case_id, args.limit)
    paths = ExperimentPaths.from_config(config)
    fingerprint = method_fingerprint(config, "distribution")
    resume = not args.overwrite
    failed = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        if resume and unit_state(paths, "distribution", record, fingerprint) == "complete":
            skipped += 1
            log.info(
                "[distribution] isolated case %d/%d %s/%s skipped (complete)",
                index, len(records), record.dataset, record.case_id,
            )
            continue
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-distribution-{uuid.uuid4().hex[:6]}"
        started_at = now()
        command = _runner_command(
            args, "distribution", [record.dataset], case_ids=[record.case_id], run_id=run_id
        )
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = ""
        returncode = _run(
            command,
            timeout=args.distribution_case_timeout,
            memory_gb=args.distribution_case_memory_gb,
            env=worker_env,
        )
        if returncode:
            failed += 1
            record_external_failure(
                config,
                "distribution",
                record,
                run_id,
                _external_failure_reason(
                    returncode, args.distribution_case_timeout, args.distribution_case_memory_gb
                ),
                started_at,
                returncode,
            )
            log.error(
                "[distribution] isolated case %d/%d %s/%s failed; continuing",
                index, len(records), record.dataset, record.case_id,
            )
            if config.experiment.get("on_failure") == "abort":
                break
    log.info(
        "[distribution] isolated run finished: total=%d skipped=%d failed=%d",
        len(records), skipped, failed,
    )
    return 1 if failed else 0


def run_experiment(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config, args.config_dir)
    except ConfigError as exc:
        log.error("configuration invalid: %s", exc)
        return 2
    methods = args.methods or list(config.experiment["methods"])
    datasets = args.datasets or list(config.experiment["datasets"])
    if args.gpus and len(args.gpus) > 2:
        log.error("--gpus accepts one GPU (shared) or two GPUs (service worker)")
        return 2

    if not config.manifest_path.is_file():
        log.info("manifest is missing; preparing benchmark data once")
        command = [sys.executable, "-m", "dema.data.prepare", "--datasets", *config.experiment["datasets"]]
        if args.config_dir:
            command += ["--config-dir", args.config_dir]
        if _run(command):
            return 1
    elif args.verify_data:
        command = [sys.executable, "-m", "dema.data.prepare", "--datasets", *datasets, "--verify-only"]
        if args.config_dir:
            command += ["--config-dir", args.config_dir]
        if _run(command):
            return 1

    errors = check_benchmark(config) + check_outputs(config)
    if errors:
        for error in errors:
            log.error("preflight: %s", error)
        return 1

    failed: list[str] = []
    abort = config.experiment.get("on_failure") == "abort"
    index = 0
    while index < len(methods):
        method = methods[index]
        service = required_service(method)

        # Consecutive methods using one model endpoint share a single server
        # lifecycle.  In particular, all DeMa variants load Open-Jev only once.
        group = [method]
        if service:
            while index + len(group) < len(methods):
                candidate = methods[index + len(group)]
                if required_service(candidate) != service:
                    break
                group.append(candidate)

        pending = {
            item: has_pending_work(
                config, item, datasets, args.case_id, args.limit, resume=not args.overwrite
            )
            for item in group if item != "distribution"
        }
        service_gpu, _ = _gpu_plan(args, method)

        def run_group() -> bool:
            for item in group:
                _, worker_gpu = _gpu_plan(args, item)
                if item == "distribution":
                    rc = _run_distribution_isolated(args, config, datasets)
                else:
                    rc = _run(
                        _runner_command(args, item, datasets), env=_worker_env(worker_gpu)
                    )
                if rc:
                    failed.append(item)
                    if abort:
                        return False
            return True

        try:
            if service and any(pending.values()):
                workers = sorted({str(_gpu_plan(args, item)[1]) for item in group})
                log.info(
                    "[%s] shared %s service: service GPU=%s worker GPU(s)=%s",
                    ",".join(group), service, service_gpu, ",".join(workers),
                )
                with ManagedService(service, config, service_gpu, args.server_timeout):
                    keep_going = run_group()
            else:
                keep_going = run_group()
        except ServiceError as exc:
            log.error("%s", exc)
            failed.extend(item for item in group if item not in failed)
            keep_going = not abort

        index += len(group)
        if not keep_going:
            break

    if not args.no_evaluate:
        command = [
            sys.executable,
            "-m",
            "dema.metrics.evaluator",
            "--config",
            args.config,
            "--methods",
            *methods,
            "--datasets",
            *datasets,
        ]
        if args.case_id:
            command += ["--case-id", *args.case_id]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        if args.config_dir:
            command += ["--config-dir", args.config_dir]
        if _run(command):
            failed.append("evaluation")

    if failed:
        log.error("finished with failures: %s", ", ".join(failed))
        return 1
    log.info("experiment finished successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dema",
        description="Run DeMa experiments from one command; required model services are managed automatically.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run methods and write metrics")
    run.add_argument("--config", default="configs/experiment.yaml")
    run.add_argument("--config-dir", default=None)
    run.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=None)
    run.add_argument("--datasets", nargs="+", default=None)
    run.add_argument("--case-id", nargs="+", default=None)
    run.add_argument("--limit", type=int, default=None, help="at most N cases per dataset")
    run.add_argument("--overwrite", action="store_true", help="rerun complete cases")
    run.add_argument("--save-debug", action="store_true")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--verify-data", action="store_true", help="recheck raw/processed data integrity")
    run.add_argument("--no-evaluate", action="store_true")
    run.add_argument("--qwen-gpu", default="0")
    run.add_argument("--decision-gpu", default="0")
    run.add_argument(
        "--gpus",
        nargs="+",
        metavar="GPU",
        default=None,
        help=(
            "unified GPU assignment: one id shares a GPU between the model service and "
            "experiment worker; two ids assign SERVICE_GPU WORKER_GPU"
        ),
    )
    run.add_argument("--server-timeout", type=float, default=900)
    run.add_argument(
        "--distribution-case-timeout",
        type=float,
        default=3600,
        help="seconds allowed for each isolated Distribution case (default: 3600)",
    )
    run.add_argument(
        "--distribution-case-memory-gb",
        type=float,
        default=64,
        help="address-space limit for each isolated Distribution process; 0 disables it (default: 64)",
    )
    run.set_defaults(func=run_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
