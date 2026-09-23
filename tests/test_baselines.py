"""Every baseline returns a complete ranking through the shared interface."""

import os

import pandas as pd
import pytest

from dema.models.ranking import validate_ranking
from dema.models.registry import build_matcher

SRC = pd.DataFrame({"name": ["alice", "bob", "carol"], "age": [31, 45, 27], "city": ["Paris", "Oslo", "Rome"]})
TGT = pd.DataFrame({"full_name": ["dave", "erin", "alice"], "years": [30, 44, 28],
                    "town": ["Paris", "Lyon", "Oslo"], "zip": ["75001", "69001", "0150"]})

FAST = ["coma", "coma_plus", "distribution", "similarity_flooding", "jaccard", "levenshtein"]


@pytest.mark.parametrize("method", FAST)
def test_fast_baselines_complete_ranking(method, config):
    m = build_matcher(method, config)
    m.load()
    matches = m.match(SRC, TGT)
    validate_ranking(matches, list(SRC.columns), list(TGT.columns))
    assert m.last_runtime.total_seconds >= 0


@pytest.mark.skipif(not os.environ.get("DEMA_SLOW_TESTS"), reason="set DEMA_SLOW_TESTS=1 (downloads models)")
@pytest.mark.parametrize("method", ["isresmat", "unicorn"])
def test_neural_baselines_complete_ranking(method, config):
    if method == "isresmat":
        config.models["baselines"]["isresmat"]["n_trn_cols"] = 4
    m = build_matcher(method, config)
    m.load()
    matches = m.match(SRC, TGT)
    validate_ranking(matches, list(SRC.columns), list(TGT.columns))
