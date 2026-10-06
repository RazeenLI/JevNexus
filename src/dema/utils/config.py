"""Configuration loading.

All experimental constants live in ``configs/*.yaml``. This module only loads
them and resolves repository-relative paths.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must contain a mapping: {path}")
    return data


def resolve_path(value: str | os.PathLike, root: Path = REPO_ROOT) -> Path:
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else (root / path)


@dataclass
class Config:
    """Bundle of the four configuration files."""

    paths: dict[str, Any]
    models: dict[str, Any]
    experiment: dict[str, Any]
    scalability: dict[str, Any] = field(default_factory=dict)
    config_dir: Path = CONFIG_DIR

    # ---------------------------------------------------------------- paths
    def path(self, key: str) -> Path:
        if key not in self.paths:
            raise ConfigError(f"paths.yaml has no key '{key}'")
        return resolve_path(self.paths[key])

    @property
    def raw_dir(self) -> Path:
        return self.path("raw_data")

    @property
    def processed_dir(self) -> Path:
        return self.path("processed_data")

    @property
    def manifest_path(self) -> Path:
        return self.path("manifest")

    @property
    def models_dir(self) -> Path:
        return self.path("models")

    @property
    def saves_dir(self) -> Path:
        return self.path("saves")

    @property
    def logs_dir(self) -> Path:
        return self.path("logs")

    @property
    def metrics_dir(self) -> Path:
        return self.path("metrics")

    @property
    def cache_dir(self) -> Path:
        return self.path("cache")

    def section(self, name: str) -> dict[str, Any]:
        """Deep copy of a top-level section of models.yaml."""
        if name not in self.models:
            raise ConfigError(f"models.yaml has no section '{name}'")
        return copy.deepcopy(self.models[name])

    def baseline(self, name: str) -> dict[str, Any]:
        baselines = self.models.get("baselines", {})
        if name not in baselines:
            raise ConfigError(f"models.yaml baselines has no entry '{name}'")
        return copy.deepcopy(baselines[name] or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "paths": copy.deepcopy(self.paths),
            "models": copy.deepcopy(self.models),
            "experiment": copy.deepcopy(self.experiment),
            "scalability": copy.deepcopy(self.scalability),
        }


def load_config(
    experiment_path: str | os.PathLike | None = None,
    config_dir: str | os.PathLike | None = None,
) -> Config:
    """Load paths/models/experiment/scalability configs.

    ``experiment_path`` may point to an alternative experiment YAML; the other
    files are taken from ``config_dir`` (default: ``configs/``).
    """

    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    exp_path = Path(experiment_path) if experiment_path is not None else cdir / "experiment.yaml"
    scal_path = cdir / "scalability.yaml"
    config = Config(
        paths=load_yaml(cdir / "paths.yaml"),
        models=load_yaml(cdir / "models.yaml"),
        experiment=load_yaml(exp_path),
        scalability=load_yaml(scal_path) if scal_path.is_file() else {},
        config_dir=cdir,
    )
    validate_config(config)
    return config


REQUIRED_MODEL_SECTIONS = ("representation", "retriever", "qwen", "decision", "fusion", "baselines")


def validate_config(config: Config) -> None:
    for key in ("raw_data", "processed_data", "manifest", "models", "saves", "logs", "metrics", "cache"):
        if key not in config.paths:
            raise ConfigError(f"paths.yaml is missing '{key}'")
    for section in REQUIRED_MODEL_SECTIONS:
        if section not in config.models:
            raise ConfigError(f"models.yaml is missing section '{section}'")
    rep = config.models["representation"]
    if rep.get("sampling") != "frequency":
        raise ConfigError("representation.sampling must be 'frequency'")
    if int(rep.get("max_values", 0)) <= 0:
        raise ConfigError("representation.max_values must be positive")
    if int(config.models["retriever"].get("top_k", 0)) <= 0:
        raise ConfigError("retriever.top_k must be positive")
    if float(config.models["qwen"].get("temperature", 0)) != 0.0:
        raise ConfigError("qwen.temperature must be 0")
    if config.models["decision"].get("candidate_context") not in ("shared", "single"):
        raise ConfigError("decision.candidate_context must be 'shared' or 'single'")
    fusion = config.models["fusion"]
    fusion_total = float(fusion.get("jev_weight", -1)) + float(fusion.get("coma_plus_weight", -1))
    if float(fusion.get("jev_weight", -1)) < 0 or float(fusion.get("coma_plus_weight", -1)) < 0:
        raise ConfigError("fusion weights must be non-negative")
    if abs(fusion_total - 1.0) > 1e-9:
        raise ConfigError("fusion weights must sum to 1")
    dynamic = config.models.get("dynamic_weight", {})
    dynamic_min = float(dynamic.get("min_jev_weight", 0.1))
    dynamic_max = float(dynamic.get("max_jev_weight", 0.7))
    if not 0 <= dynamic_min <= dynamic_max <= 1:
        raise ConfigError("dynamic_weight bounds must satisfy 0 <= min <= max <= 1")
    if int(dynamic.get("evidence_top_n", 5)) <= 0:
        raise ConfigError("dynamic_weight.evidence_top_n must be positive")
    jina = config.models.get("jina_rerank", {})
    if jina and int(jina.get("top_n", 3)) <= 1:
        raise ConfigError("jina_rerank.top_n must be greater than one")
    gate = config.models.get("selective_refinement", {})
    if float(gate.get("margin_threshold", -1)) < 0:
        raise ConfigError("selective_refinement.margin_threshold must be non-negative")
    decision_serving = config.models.get("serving", {}).get("decision", {})
    if int(decision_serving.get("batch_size", 0)) <= 0:
        raise ConfigError("serving.decision.batch_size must be positive")
    for key in ("datasets", "methods"):
        if not isinstance(config.experiment.get(key), list) or not config.experiment[key]:
            raise ConfigError(f"experiment config must list '{key}'")
    if config.experiment.get("on_failure", "continue") not in ("continue", "abort"):
        raise ConfigError("experiment.on_failure must be 'continue' or 'abort'")
