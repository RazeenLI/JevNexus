import numpy as np
import pytest

from dema.data.types import ColumnProfile
from dema.models.retriever import CandidateCache, CandidateRetriever, cosine_matrix, order_targets

from conftest import trigram_encoder

REP = {"version": "t", "max_values": 3, "sampling": "frequency", "include_dtype": True}
RET = {"model": "fake", "top_k": 2}


def prof(name):
    return ColumnProfile(name, "string", ("v",))


def test_cosine_matrix():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, 3.0]])
    sim = cosine_matrix(a, b)
    assert sim == pytest.approx(np.array([[1.0, 0.0], [2 ** -0.5, 2 ** -0.5]]))


def test_order_targets_ties_by_position():
    order = order_targets(np.array([[0.5, 0.9, 0.5, 0.1]]))
    assert order.tolist() == [[1, 0, 2, 3]]


def fake_retriever(scores, cache=None):
    r = CandidateRetriever(RET, REP, cache=cache, encoder=lambda texts: None)
    r.score_matrix = lambda s, t: np.asarray(scores, dtype=float)
    return r


def test_topk_ordering_and_uniqueness():
    src = [prof("a"), prof("b")]
    tgt = [prof("x"), prof("y"), prof("z")]
    r = fake_retriever([[0.1, 0.9, 0.5], [0.3, 0.3, 0.2]])
    out = r.retrieve(src, tgt, top_k=2)
    assert [c.target_column for c in out["a"]] == ["y", "z"]
    assert [c.retrieval_rank for c in out["a"]] == [1, 2]
    assert [c.target_column for c in out["b"]] == ["x", "y"]  # tie -> schema order
    for cands in out.values():
        assert len({c.target_column for c in cands}) == len(cands)


def test_k_larger_than_target_count():
    r = fake_retriever([[0.2, 0.1]])
    out = r.retrieve([prof("a")], [prof("x"), prof("y")], top_k=20)
    assert [c.target_column for c in out["a"]] == ["x", "y"]


def test_real_scoring_path_with_encoder():
    r = CandidateRetriever(RET, REP, encoder=trigram_encoder)
    out = r.retrieve([prof("customer_name")], [prof("zip"), prof("customer_names"), prof("city")], top_k=1)
    assert out["customer_name"][0].target_column == "customer_names"


def test_cache_reuse_and_key(tmp_path):
    cache = CandidateCache(tmp_path)
    src, tgt = [prof("a")], [prof("x"), prof("y")]
    r1 = fake_retriever([[0.2, 0.8]], cache)
    first = r1.rank_all(src, tgt, 2, "D", "c1")
    assert not r1.last_cache_hit
    r2 = fake_retriever([[0.9, 0.1]], cache)  # would give another order if recomputed
    second = r2.rank_all(src, tgt, 2, "D", "c1")
    assert r2.last_cache_hit and second == first
    third = r2.rank_all(src, tgt, 1, "D", "c1")  # top_k is part of the key
    assert not r2.last_cache_hit and third["a"][0].target_column == "x"
