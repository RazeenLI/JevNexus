import pytest

from dema.data.types import Candidate
from dema.model.ranking import RankingError, build_final_ranking, ranking_from_scores, validate_ranking


def full(order_scores):
    return [Candidate("s", t, sc, r) for r, (t, sc) in enumerate(order_scores, start=1)]


def test_topk_by_reranker_tail_by_retrieval():
    ordering = full([("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)])
    matches = build_final_ranking(ordering, {"a": 0.1, "b": 0.7, "c": 0.7}, top_k=3)
    assert [m.target_column for m in matches] == ["b", "c", "a", "d", "e"]  # b/c tie -> retrieval rank
    assert [m.rank for m in matches] == [1, 2, 3, 4, 5]
    assert matches[0].reranker_score == 0.7 and matches[0].retrieval_score == 0.8
    assert matches[3].reranker_score is None and matches[3].retrieval_score == 0.6
    validate_ranking(matches, ["s"], ["a", "b", "c", "d", "e"])


def test_missing_rerank_score_is_error():
    with pytest.raises(RankingError):
        build_final_ranking(full([("a", 0.9), ("b", 0.8)]), {"a": 0.5}, top_k=2)


def test_every_target_exactly_once_from_partial_scores():
    matches = ranking_from_scores(["s1", "s2"], ["x", "y", "z"], {("s1", "y"): 0.9, ("s2", "z"): 0.1})
    validate_ranking(matches, ["s1", "s2"], ["x", "y", "z"])
    s1 = [m.target_column for m in matches if m.source_column == "s1"]
    assert s1 == ["y", "x", "z"]  # scored first, unscored in schema order


def test_validate_detects_duplicates_and_gaps():
    ordering = build_final_ranking(full([("a", 0.9), ("b", 0.8)]), {"a": 0.1, "b": 0.2}, top_k=2)
    with pytest.raises(RankingError):
        validate_ranking(ordering, ["s"], ["a", "b", "c"])
    with pytest.raises(RankingError):
        validate_ranking(ordering + ordering[:1], ["s"], ["a", "b"])
