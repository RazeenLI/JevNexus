"""Optional trivial baselines: Jaccard (valentine) and column-name Levenshtein."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from .base import ScoreMatrixBaseline, ValentineBaseline


class JaccardMatcher(ValentineBaseline):
    name = "jaccard"

    def build_matcher(self):
        from valentine.algorithms import JaccardDistanceMatcher

        return JaccardDistanceMatcher(threshold_dist=float(self.cfg.get("threshold_dist", 0.8)))


class LevenshteinMatcher(ScoreMatrixBaseline):
    """Normalized Levenshtein similarity of lower-cased column names."""

    name = "levenshtein"

    def compute_scores(self, source_df, target_df):
        scores = {}
        for s in map(str, source_df.columns):
            for t in map(str, target_df.columns):
                scores[(s, t)] = Levenshtein.normalized_similarity(s.lower(), t.lower())
        return scores
