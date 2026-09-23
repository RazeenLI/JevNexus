"""COMA (schema-based) via valentine's pure-Python COMA implementation.

Simplification: Magneto's comparison used the original Java COMA 3.0 through
older valentine releases; valentine 1.x ships a Python port of the same
matcher library, which is used here (no Java runtime needed). ``delta=1.0`` and
``threshold=0`` keep every scored pair so a complete ranking exists.
"""

from __future__ import annotations

from .base import ValentineBaseline


class COMAMatcher(ValentineBaseline):
    name = "coma"

    def build_matcher(self):
        from valentine.algorithms import Coma

        c = self.cfg
        return Coma(
            max_n=int(c.get("max_n", 0)),
            use_instances=bool(c.get("use_instances", False)),
            use_schema=bool(c.get("use_schema", True)),
            delta=float(c.get("delta", 1.0)),
            threshold=float(c.get("threshold", 0.0)),
            instance_weight=float(c.get("instance_weight", 1.0)),
        )
