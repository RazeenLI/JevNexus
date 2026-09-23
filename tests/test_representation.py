import numpy as np
import pandas as pd

from dema.representation.profiler import infer_type, profile_table
from dema.representation.sampler import non_null_strings, sample_values
from dema.representation.serializer import serialize_profile

REP = {"max_values": 3, "sampling": "frequency", "include_dtype": True, "max_value_chars": 64,
       "type_min_fraction": 1.0, "datetime_min_fraction": 0.9, "mixed_min_numeric_fraction": 0.2}


# ------------------------------------------------------------------ sampling
def test_sampling_is_deterministic_and_frequency_ordered():
    values = ["b", "a", "c", "a", "b", "a", "d"]
    assert sample_values(values, 3) == ["a", "b", "c"]  # a:3, b:2, then c/d tie -> lexicographic
    assert sample_values(values, 3) == sample_values(list(reversed(values)), 3)


def test_sampling_null_safe():
    values = [None, np.nan, "", "  ", "x", pd.NA, "x", "y"]
    assert sample_values(values, 10) == ["x", "y"]
    assert sample_values([None, np.nan], 5) == []


def test_sampling_respects_max_values():
    assert len(sample_values([str(i) for i in range(100)], 10)) == 10


def test_sampling_stable_tie_ordering():
    assert sample_values(["z", "y", "x"], 2) == ["x", "y"]
    assert sample_values(["x", "z", "y"], 2) == ["x", "y"]


def test_float_normalization():
    assert non_null_strings([1.0, 2.5, np.nan]) == ["1", "2.5"]


# ------------------------------------------------------------ type inference
def test_type_inference_cases():
    assert infer_type(["1", "2", "-3"]) == "integer"
    assert infer_type(["1.5", "2", "3e2"]) == "float"
    assert infer_type(["true", "False", "yes"]) == "boolean"
    assert infer_type(["2020-01-03", "2019-05-17", "2021/08/21"]) == "datetime"
    assert infer_type(["red", "green", "blue"]) == "string"
    assert infer_type(["1", "2", "abc", "def"]) == "mixed"
    assert infer_type([]) == "string"  # empty column


def test_years_are_not_datetimes():
    assert infer_type(["1999", "2001"]) == "integer"


def test_profile_table_and_serializer():
    df = pd.DataFrame({"release_date": ["2020-01-03", "2019-05-17", "2020-01-03", None],
                       "n": [1, 2, 2, np.nan], "empty": [None] * 4})
    profiles = profile_table(df, REP)
    assert [p.name for p in profiles] == ["release_date", "n", "empty"]
    assert profiles[0].dtype == "datetime" and profiles[0].values == ("2020-01-03", "2019-05-17")
    assert profiles[1].dtype == "integer" and profiles[1].values == ("2", "1")
    assert profiles[2].values == ()
    assert profile_table(df, REP) == profiles  # deterministic across runs
    text = serialize_profile(profiles[0], REP)
    assert text == "Column: release_date\nType: datetime\nValues: 2020-01-03 | 2019-05-17"
