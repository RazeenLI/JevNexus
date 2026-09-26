"""COMA++: COMA with the instance-based (TF-IDF) matcher enabled.

Corresponds to the benchmark's ``ComaInst`` baseline. It samples up to 500 rows
from each table before invoking COMA. A fixed seed makes repeated experiments
reproducible; this is the only sampling difference from the unseeded wrapper.
"""

from __future__ import annotations

from .coma import COMAMatcher


class COMAPlusMatcher(COMAMatcher):
    name = "coma_plus"

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        cfg.setdefault("use_instances", True)
        super().__init__(cfg)

    def compute_scores(self, source_df, target_df):
        size = int(self.cfg.get("instance_sample_size", 500))
        seed = int(self.cfg.get("sample_seed", 42))
        if len(source_df) > size:
            source_df = source_df.sample(n=size, random_state=seed)
        if len(target_df) > size:
            target_df = target_df.sample(n=size, random_state=seed)
        # Frames are already sampled exactly like the reference wrapper.
        previous = self.cfg.get(self.instance_sample_size_key)
        self.cfg[self.instance_sample_size_key] = None
        try:
            return super().compute_scores(source_df, target_df)
        finally:
            self.cfg[self.instance_sample_size_key] = previous
