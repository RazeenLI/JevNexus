"""COMA (schema-based) via valentine's pure-Python COMA implementation.

The benchmark wrapper's ``max_n=top_k`` and COMA's default ``delta=0.15`` are
preserved. Valentine 1.x is a maintained Python implementation rather than the
older Java bridge; this controlled difference is recorded in the baseline audit.
"""

from __future__ import annotations

from .base import ValentineBaseline


class COMAMatcher(ValentineBaseline):
    name = "coma"

    def build_matcher(self):
        from valentine.algorithms import Coma

        c = self.cfg
        return Coma(
            max_n=int(c.get("max_n", 20)),
            use_instances=bool(c.get("use_instances", False)),
            use_schema=bool(c.get("use_schema", True)),
            delta=float(c.get("delta", 0.15)),
            threshold=float(c.get("threshold", 0.0)),
            instance_weight=float(c.get("instance_weight", 1.0)),
        )
