"""Distribution-based matching (Zhang et al., SIGMOD 2011) via valentine."""

from __future__ import annotations

from .base import ValentineBaseline


class DistributionMatcher(ValentineBaseline):
    name = "distribution"

    def build_matcher(self):
        from valentine.algorithms import DistributionBased

        c = self.cfg
        return DistributionBased(
            threshold1=float(c.get("threshold1", 0.15)),
            threshold2=float(c.get("threshold2", 0.15)),
            quantiles=int(c.get("quantiles", 256)),
            process_num=int(c.get("process_num", 1)),
            use_bloom_filters=bool(c.get("use_bloom_filters", False)),
        )
