"""magneto_qwen: the original Magneto code with its LLM swapped to Qwen3.5-9B.

Runs ``Magneto(...).get_matches`` from the vendored upstream package in
``vendor/magneto`` (see PATCHES.md); JevNexus only converts its output to the
shared complete-ranking format. Magneto returns its top-k (20) targets per
source column; the remaining targets are appended after them in target-schema
order. Magneto's own behaviour is kept, including its fallback to the retrieval
order after 5 unparsable LLM responses — such fallbacks are counted in
``runtime.failures`` so they stay visible.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

from ..utils.config import REPO_ROOT
from .base import ScoreMatrixBaseline

VENDOR_DIR = REPO_ROOT / "vendor" / "magneto"


def _import_magneto():
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    from magneto import _local_llm
    from magneto.magneto import Magneto

    return Magneto, _local_llm


class MagnetoOriginalMatcher(ScoreMatrixBaseline):
    name = "magneto_qwen"

    def __init__(self, cfg: dict[str, Any], qwen_cfg: dict[str, Any]):
        super().__init__(cfg)
        self.qwen_cfg = qwen_cfg
        self._magneto_cls = None
        self._llm = None

    def load(self) -> None:
        if self._magneto_cls is not None:
            return
        self._magneto_cls, self._llm = _import_magneto()
        key = os.environ.get(self.qwen_cfg.get("api_key_env") or "", "") or None
        self._llm.configure(
            base_url=self.qwen_cfg["base_url"],
            model=self.qwen_cfg.get("served_model_name") or self.qwen_cfg["model"],
            timeout=float(self.qwen_cfg.get("timeout", 300)),
            extra_body=self.cfg.get("extra_body") or {},
            api_key=key,
        )

    def describe(self) -> dict[str, Any]:
        return {"method": self.name, "implementation": "upstream Magneto (vendor/magneto)",
                "llm": self.qwen_cfg.get("served_model_name"), "base_url": self.qwen_cfg["base_url"],
                "config": self.cfg}

    def params_for(self, dataset: str | None) -> dict[str, Any]:
        params = dict(self.cfg.get("params") or {})
        params.update((self.cfg.get("params_by_dataset") or {}).get(dataset or "", {}) or {})
        params["llm_model_kwargs"] = dict(self.cfg.get("llm_model_kwargs") or {})
        return params

    def compute_scores(self, source_df, target_df):
        self.load()
        dataset = self.case_context.dataset if self.case_context else None
        self._llm.reset_stats()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matches = self._magneto_cls(**self.params_for(dataset)).get_matches(source_df, target_df)
        stats = self.last_runtime
        stats.model_requests = self._llm.STATS["requests"]
        stats.input_tokens = self._llm.STATS["input_tokens"]
        stats.output_tokens = self._llm.STATS["output_tokens"]
        # Magneto warns on every unparsable response and on each fallback.
        stats.failures = sum("Failed to parse response" in str(w.message) for w in caught)
        stats.retries = sum("Error parsing JSON response" in str(w.message) for w in caught)
        scores: dict[tuple[str, str], float] = {}
        for key, score in matches.items():
            src, tgt = str(key[0][1]), str(key[1][1])
            scores[(src, tgt)] = max(float(score), scores.get((src, tgt), float("-inf")))
        return scores
