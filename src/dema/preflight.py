"""Pre-flight checks before running experiments.

    python -m dema.preflight [--methods ...]

Checks: configuration valid, processed benchmark + manifest present, output
directory writable, Qwen endpoint answers a real scoring request (if
``magneto_qwen`` is selected) and the decision endpoint answers a real
System One request (if ``dema`` is selected). Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from .data.manifest import read_manifest
from .data.types import ColumnProfile
from .evaluation.runtime import RuntimeStats
from .utils.config import ConfigError, load_config
from .utils.logging import get_console_logger

log = get_console_logger("dema.preflight")

_SRC = ColumnProfile("birth_date", "datetime", ("1990-01-02", "1985-07-30"))
_CANDS = [("c0", ColumnProfile("date_of_birth", "datetime", ("1970-03-04",))),
          ("c1", ColumnProfile("city", "string", ("Paris", "Oslo")))]


def check_benchmark(config) -> list[str]:
    errors = []
    try:
        records = read_manifest(config.manifest_path)
    except FileNotFoundError as exc:
        return [str(exc)]
    if not records:
        errors.append("manifest is empty")
    wanted = set(config.experiment["datasets"])
    have = {r.dataset for r in records}
    if wanted - have:
        errors.append(f"datasets missing from manifest: {sorted(wanted - have)}")
    for r in records:
        for key in ("source_path", "target_path", "ground_truth_path", "metadata_path"):
            if not r.abs(key).is_file():
                errors.append(f"missing processed file {r.abs(key)}")
                break
    return errors


def check_outputs(config) -> list[str]:
    try:
        config.outputs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=config.outputs_dir):
            pass
    except OSError as exc:
        return [f"output directory not writable: {exc}"]
    return []


def check_qwen(config) -> list[str]:
    from .models.generative_reranker import QwenReranker

    cfg = config.section("qwen")
    cfg["max_retries"] = 0
    try:
        scores = QwenReranker(cfg).score(_SRC, _CANDS, RuntimeStats())
    except Exception as exc:  # noqa: BLE001
        return [f"Qwen endpoint {cfg['base_url']} failed: {exc}"]
    log.info("Qwen endpoint OK: %s", scores)
    return []


def check_decision(config) -> list[str]:
    from .models.decision_reranker import DecisionReranker

    cfg = config.section("decision")
    cfg["max_retries"] = 0
    try:
        probs = DecisionReranker(cfg).score(_SRC, _CANDS, RuntimeStats())
    except Exception as exc:  # noqa: BLE001
        return [f"decision endpoint {cfg['base_url']} failed: {exc}"]
    log.info("decision endpoint OK: %s", probs)
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--config-dir", default=None)
    args = p.parse_args(argv)
    try:
        config = load_config(args.config, args.config_dir)
    except ConfigError as exc:
        log.error("configuration invalid: %s", exc)
        return 1
    methods = args.methods or list(config.experiment["methods"])
    errors = check_benchmark(config) + check_outputs(config)
    if "magneto_qwen" in methods:
        errors += check_qwen(config)
    if "dema" in methods:
        errors += check_decision(config)
    for e in errors:
        log.error("preflight: %s", e)
    if errors:
        return 1
    log.info("preflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
