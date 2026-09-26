"""Resume semantics: complete cases are skipped, partial/failed ones re-run."""

import json

import pandas as pd

from dema.data.types import Match
from dema.model.base import BaseMatcher
from dema.experiments.runner import ExperimentPaths, run, unit_state


class CountingMatcher(BaseMatcher):
    name = "levenshtein"

    def __init__(self, fail_on=()):
        super().__init__()
        self.calls = []
        self.fail_on = set(fail_on)

    def match(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
        cid = self.case_context.case_id
        self.calls.append(cid)
        if cid in self.fail_on:
            raise RuntimeError("simulated failure")
        out = []
        for s in source_df.columns:
            out += [Match(s, t, 1.0 / (i + 1), i + 1) for i, t in enumerate(target_df.columns)]
        return out


def test_resume_reruns_only_incomplete(mini_benchmark):
    config, records = mini_benchmark
    paths = ExperimentPaths.from_config(config)
    gdc = next(r for r in records if r.dataset == "GDC")
    od = next(r for r in records if r.dataset == "OpenData")

    m = CountingMatcher(fail_on={od.case_id})
    counts = run(config, "levenshtein", ["GDC", "OpenData"], matcher=m, argv=["t"])
    assert counts == {"skipped": 0, "success": 1, "failed": 1}
    assert unit_state(paths, "levenshtein", gdc) == "complete"
    assert unit_state(paths, "levenshtein", od) == "failed"

    # completed case skipped, failed case re-run
    m2 = CountingMatcher()
    run(config, "levenshtein", ["GDC", "OpenData"], matcher=m2, argv=["t"])
    assert m2.calls == [od.case_id]

    # partial case: truncate a prediction file -> incomplete -> re-run
    pred = paths.prediction("levenshtein", "GDC", gdc.case_id)
    doc = json.loads(pred.read_text())
    first = next(iter(doc["predictions"]))
    doc["predictions"][first] = doc["predictions"][first][:1]
    pred.write_text(json.dumps(doc))
    assert unit_state(paths, "levenshtein", gdc) == "incomplete"
    m3 = CountingMatcher()
    run(config, "levenshtein", ["GDC", "OpenData"], matcher=m3, argv=["t"])
    assert m3.calls == [gdc.case_id]

    # prediction without status marker (crash before marker) -> re-run
    paths.status("levenshtein", "GDC", gdc.case_id).unlink()
    m4 = CountingMatcher()
    run(config, "levenshtein", ["GDC"], matcher=m4, argv=["t"])
    assert m4.calls == [gdc.case_id]

    # --overwrite re-runs everything
    m5 = CountingMatcher()
    run(config, "levenshtein", ["GDC", "OpenData"], resume=False, matcher=m5, argv=["t"])
    assert sorted(m5.calls) == sorted([gdc.case_id, od.case_id])

    # A semantic config change invalidates old success markers.
    config.experiment["seed"] += 1
    m6 = CountingMatcher()
    run(config, "levenshtein", ["GDC"], matcher=m6, argv=["t"])
    assert m6.calls == [gdc.case_id]
