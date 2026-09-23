import math

import pytest

from dema.evaluation.metrics import (
    all_metrics, hits_at_k, mean_average_precision, mrr, ndcg_at_k, recall_at_gt, recall_at_k,
)

# Hand-constructed example (see comments for the arithmetic).
SCORED = {
    "s1": [("a", 0.9), ("b", 0.8), ("c", 0.1)],
    "s2": [("b", 0.95), ("a", 0.5), ("c", 0.4)],
    "s3": [("c", 0.7), ("b", 0.6), ("a", 0.3)],
}
RANK = {s: [t for t, _ in e] for s, e in SCORED.items()}
GT = [("s1", "b"), ("s2", "b"), ("s2", "c"), ("s3", "a")]  # s2 has two valid targets


def test_mrr_uses_best_correct_rank():
    # s1: rank 2 -> 1/2; s2: best of {b@1, c@3} -> 1; s3: rank 3 -> 1/3
    assert mrr(RANK, GT) == pytest.approx((0.5 + 1 + 1 / 3) / 3)


def test_mrr_missing_source_counts_as_zero():
    assert mrr(RANK, GT + [("s4", "x")]) == pytest.approx((0.5 + 1 + 1 / 3) / 4)


def test_hits_at_k():
    assert hits_at_k(RANK, GT, 1) == pytest.approx(1 / 3)
    assert hits_at_k(RANK, GT, 2) == pytest.approx(2 / 3)
    assert hits_at_k(RANK, GT, 5) == pytest.approx(1.0)


def test_recall_at_k_is_pair_level():
    assert recall_at_k(RANK, GT, 1) == pytest.approx(1 / 4)  # (s2,b)
    assert recall_at_k(RANK, GT, 2) == pytest.approx(2 / 4)  # + (s1,b)
    assert recall_at_k(RANK, GT, 3) == pytest.approx(1.0)


def test_recall_at_gt_global_topn():
    # global order: (s2,b).95 (s1,a).9 (s1,b).8 (s3,c).7 -> top-4 hits (s2,b),(s1,b)
    assert recall_at_gt(SCORED, GT) == pytest.approx(0.5)


def test_recall_at_gt_single_pair_and_ties():
    scored = {"s": [("x", 0.5), ("y", 0.5)], "t": [("y", 0.5), ("x", 0.5)]}
    # all scores tie -> per-source rank decides: (s,x) and (t,y) come first
    assert recall_at_gt(scored, [("s", "x"), ("t", "y")]) == 1.0
    assert recall_at_gt(scored, [("s", "y"), ("t", "x")]) == 0.0


def test_ndcg_and_map():
    ndcg = (1 / math.log2(3) + (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)) + 0.5) / 3
    assert ndcg_at_k(RANK, GT, 5) == pytest.approx(ndcg)
    assert mean_average_precision(RANK, GT) == pytest.approx((0.5 + (1 + 2 / 3) / 2 + 1 / 3) / 3)


def test_all_metrics_keys_and_empty_gt():
    out = all_metrics(SCORED, GT)
    assert {"MRR", "Recall@GT", "Hits@1", "Hits@5", "Hits@10", "Recall@1", "Recall@5",
            "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "MAP"} <= set(out)
    assert math.isnan(all_metrics(SCORED, [])["MRR"])
