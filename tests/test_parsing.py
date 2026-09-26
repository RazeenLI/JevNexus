import pytest

from dema.data.types import ColumnProfile
from dema.model.contracts import RerankerOutputError, load_json_strict, validate_probability_map
from dema.model.decision import DecisionReranker, parse_decision_response

IDS = ["c0", "c1", "c2"]


@pytest.mark.parametrize("text, message", [
    ('{"c0": 0.9, "c1": 0.1}', "missing"),
    ('{"c0": 0.9, "c1": 0.1, "c2": 0.3, "c3": 0.2}', "unexpected"),
    ('{"c0": 0.9, "c1": 0.1, "c2": 0.3, "c1": 0.2}', "duplicate"),
    ('{"c0": 1.2, "c1": 0.1, "c2": 0.3}', "outside"),
    ('{"c0": -0.1, "c1": 0.1, "c2": 0.3}', "outside"),
    ('{"c0": "high", "c1": 0.1, "c2": 0.3}', "not a number"),
    ('{"c0": true, "c1": 0.1, "c2": 0.3}', "not a number"),
    ('{"c0": NaN, "c1": 0.1, "c2": 0.3}', "non-finite"),
    ('{"c0": 0.9, "c1": 0.1, "c2": 0.3', "JSON"),
    ("I think c0 matches.", "JSON"),
])
def test_probability_contract_rejects_invalid_responses(text, message):
    with pytest.raises(RerankerOutputError, match=message):
        validate_probability_map(load_json_strict(text), IDS)


# ------------------------------------------------------------ decision output
def resp(**probs):
    return {"answers": {k: {"type": "noul", "noul": v} for k, v in probs.items()}}


def test_decision_exactly_k_probabilities():
    out = parse_decision_response(resp(c0=0.2, c1=0.9, c2=0.0), IDS)
    assert out == {"c0": 0.2, "c1": 0.9, "c2": 0.0} and list(out) == IDS


def test_decision_all_ids_present():
    with pytest.raises(RerankerOutputError, match="missing"):
        parse_decision_response(resp(c0=0.2, c1=0.9), IDS)
    with pytest.raises(RerankerOutputError, match="unexpected"):
        parse_decision_response(resp(c0=0.2, c1=0.9, c2=0.1, c9=0.1), IDS)


def test_decision_probability_range():
    with pytest.raises(RerankerOutputError, match="outside"):
        parse_decision_response(resp(c0=0.2, c1=1.01, c2=0.1), IDS)
    with pytest.raises(RerankerOutputError):
        parse_decision_response({"nope": 1}, IDS)
    with pytest.raises(RerankerOutputError, match="type"):
        parse_decision_response({"answers": {c: {"type": "choice", "noul": 0.5} for c in IDS}}, IDS)


def test_decision_request_has_no_retrieval_information():
    cfg = {"model": "m", "base_url": "http://x", "prompt": {
        "question": "Does candidate {cid} match?", "question_single": "{candidate}", "criteria": None},
        "candidate_context": "single"}
    rr = DecisionReranker(cfg, backend=object())
    state, questions = rr.build_request(
        ColumnProfile("a", "string", ("1",)), [("c0", ColumnProfile("b", "integer", ("2",)))]
    )
    blob = repr(state) + repr(questions)
    assert "retrieval" not in blob and "rank" not in blob and "score" not in blob
    assert questions["c0"]["instructions"] == "name: b\ntype: integer\nvalues: 2"
    assert "candidates" not in state
