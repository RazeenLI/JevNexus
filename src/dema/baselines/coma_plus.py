"""COMA++: COMA with the instance-based (TF-IDF) matcher enabled.

Corresponds to Magneto's ``ComaInst`` baseline. The reference randomly sampled
500 rows per table; here valentine keeps the first ``instance_sample_size``
non-empty rows, which is deterministic.
"""

from __future__ import annotations

from .coma import COMAMatcher


class COMAPlusMatcher(COMAMatcher):
    name = "coma_plus"

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        cfg.setdefault("use_instances", True)
        super().__init__(cfg)
