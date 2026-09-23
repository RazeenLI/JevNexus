"""Similarity Flooding (Melnik et al., ICDE 2002) via valentine."""

from __future__ import annotations

from .base import ValentineBaseline


class SimilarityFloodingMatcher(ValentineBaseline):
    name = "similarity_flooding"

    def build_matcher(self):
        from valentine.algorithms import Formula, Policy, SimilarityFlooding, StringMatcher

        c = self.cfg
        return SimilarityFlooding(
            coeff_policy=Policy[str(c.get("coeff_policy", "inverse_average")).upper()],
            formula=Formula[str(c.get("formula", "formula_c")).upper()],
            string_matcher=StringMatcher[str(c.get("string_matcher", "prefix_suffix")).upper()],
        )
