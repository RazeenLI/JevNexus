"""DeMa integration and the upstream-Magneto adapter."""

import pytest

from dema.data.loader import load_case
from dema.experiments.runner import ExperimentPaths, run, unit_state
from dema.model.base import CaseContext
from dema.model.fusion import FixedScoreFusionReranker
from dema.model.experimental_rerank import (
    JevDynamicWeightReranker,
    JevJinaTopReranker,
    JinaListwiseModel,
    JinaTopReranker,
)
from dema.model.ranking import validate_ranking
from dema.data.types import ColumnProfile
from dema.metrics.runtime import RuntimeStats
from dema.model.registry import build_matcher
from dema.utils.io import read_json

from conftest import trigram_encoder


class _FixedDecision:
    context = "single"
    include_dtype = False

    def score(self, source, candidates, stats, debug=None):
        return {"c0": 0.5, "c1": 0.2}

    def describe(self):
        return {"type": "fixed-test-decision"}


class _WeightBackend:
    def predict(self, state, questions):
        assert list(questions) == ["weight"]
        assert "Jev score" in questions["weight"]["instructions"]
        return {"answers": {"weight": {"type": "noul", "noul": 0.75}}}


class _DynamicDecision(_FixedDecision):
    max_retries = 0
    backend = _WeightBackend()


class _ReverseJinaModel:
    def rerank(self, query, documents, top_n=None):
        assert "name: source" in query
        assert all("Jev score" in document for document in documents)
        assert top_n == len(documents)
        return [{"index": index, "relevance_score": float(index)}
                for index in reversed(range(len(documents)))]


def test_fixed_fusion_uses_raw_scores_and_zero_for_unselected_coma_pair():
    reranker = FixedScoreFusionReranker(_FixedDecision(), 0.4, 0.6)
    reranker.coma_scores = {("source", "target_a"): 0.75}
    source = ColumnProfile("source", "", ())
    candidates = [
        ("c0", ColumnProfile("target_a", "", ())),
        ("c1", ColumnProfile("target_b", "", ())),
    ]

    scores = reranker.score(source, candidates, RuntimeStats())

    assert scores == pytest.approx({"c0": 0.65, "c1": 0.08})


def test_jev_dynamic_weight_maps_probability_to_bounded_weight():
    base = FixedScoreFusionReranker(_DynamicDecision(), 0.4, 0.6)
    base.coma_scores = {("source", "target_a"): 0.75, ("source", "target_b"): 0.5}
    reranker = JevDynamicWeightReranker(
        base, {"min_jev_weight": 0.1, "max_jev_weight": 0.7, "evidence_top_n": 2}
    )
    source = ColumnProfile("source", "", ())
    candidates = [("c0", ColumnProfile("target_a", "", ())),
                  ("c1", ColumnProfile("target_b", "", ()))]

    stats = RuntimeStats()
    scores = reranker.score(source, candidates, stats)

    # P=0.75 -> Jev weight=0.55, COMA+ weight=0.45.
    assert scores == pytest.approx({"c0": 0.6125, "c1": 0.335})
    assert stats.model_requests == 1  # the fake first-pass decision does not count itself


def test_jina_reranker_only_permutes_top_scores():
    base = FixedScoreFusionReranker(_FixedDecision(), 0.4, 0.6)
    base.coma_scores = {("source", "target_a"): 0.75, ("source", "target_b"): 0.5}
    adapter = JinaListwiseModel({}, model=_ReverseJinaModel())
    reranker = JinaTopReranker(
        base,
        {"model": "fake", "top_n": 2, "include_scores": True},
        listwise_model=adapter,
    )
    source = ColumnProfile("source", "", ())
    candidates = [("c0", ColumnProfile("target_a", "", ())),
                  ("c1", ColumnProfile("target_b", "", ()))]

    scores = reranker.score(source, candidates, RuntimeStats())

    # Fixed scores are c0=.65, c1=.38; reversed Jina order swaps only the values.
    assert scores == pytest.approx({"c0": 0.38, "c1": 0.65})


def test_jina_no_coma_ablation_uses_only_jev_scores():
    class NoComaJinaModel:
        def rerank(self, query, documents, top_n=None):
            assert all("Jev score" in document for document in documents)
            assert all("COMA+" not in document and "fusion" not in document for document in documents)
            return [{"index": 1, "relevance_score": 1.0},
                    {"index": 0, "relevance_score": 0.0}]

    adapter = JinaListwiseModel({}, model=NoComaJinaModel())
    reranker = JevJinaTopReranker(
        _FixedDecision(),
        {"model": "fake", "top_n": 2, "include_scores": True},
        listwise_model=adapter,
    )
    source = ColumnProfile("source", "", ())
    candidates = [("c0", ColumnProfile("target_a", "", ())),
                  ("c1", ColumnProfile("target_b", "", ()))]

    scores = reranker.score(source, candidates, RuntimeStats())

    assert scores == pytest.approx({"c0": 0.2, "c1": 0.5})


class _FakeMagnetoStats:
    STATS = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    @classmethod
    def reset_stats(cls):
        for key in cls.STATS:
            cls.STATS[key] = 0


class _FakeUpstreamMagneto:
    last_params = None

    def __init__(self, **params):
        type(self).last_params = params

    def get_matches(self, source_df, target_df):
        return {
            (("source", str(source)), ("target", str(target))): float(source == target)
            for source in source_df.columns for target in target_df.columns
        }


def matcher(method, config):
    m = build_matcher(method, config)
    if method == "dema_legacy":
        m.retrieval._encoder = trigram_encoder  # no model download in tests
    elif method == "magneto_qwen":
        m._magneto_cls = _FakeUpstreamMagneto
        m._llm = _FakeMagnetoStats
    return m


def test_dema_uses_single_candidate_questions(mini_benchmark, fake_servers):
    config, records = mini_benchmark
    _, decision = fake_servers
    config.models["retriever"]["top_k"] = 3
    case = load_case(records[0])
    m = matcher("dema_legacy", config)
    ranking = m.match(case.source_df, case.target_df)
    validate_ranking(ranking, list(case.source_df.columns), list(case.target_df.columns))
    assert m.last_runtime.model_requests == case.source_df.shape[1]
    for request in decision.requests:
        assert len(request["questions"]) == 3
        assert "Candidate target columns" not in request["state"]
        for question in request["questions"].values():
            assert question["type"] == "noul"
            assert "name:" in question["instructions"]
            assert "retrieval" not in question["instructions"]


def test_registered_dema_variants_have_separate_context_and_retrieval(config):
    main = build_matcher("dema", config)
    no_rerank = build_matcher("dema_no_rerank", config)
    no_struct = build_matcher("dema_no_struct", config)
    decision = build_matcher("dema_decision", config)
    fusion = build_matcher("dema_fusion", config)
    dynamic = build_matcher("dema_jev_weight", config)
    jina = build_matcher("dema_jina_rerank", config)
    jina_no_coma = build_matcher("dema_jina_no_coma", config)
    shared = build_matcher("dema_shared", config)
    single = build_matcher("dema_single", config)
    own = build_matcher("dema_own_retrieval", config)

    assert main.name == "dema"
    assert no_rerank.name == "dema_no_rerank"
    assert no_struct.name == "dema_no_struct"
    assert decision.name == "dema_decision"
    assert type(main.reranker).__name__ == "JinaTopReranker"
    assert type(no_rerank.reranker).__name__ == "FixedScoreFusionReranker"
    assert type(no_struct.reranker).__name__ == "JevJinaTopReranker"
    assert type(decision.reranker).__name__ == "DecisionReranker"
    assert fusion.reranker.context == "single"
    assert fusion.reranker.jev_weight == 0.4
    assert fusion.reranker.coma_plus_weight == 0.6
    assert dynamic.reranker.min_jev_weight == 0.1
    assert dynamic.reranker.max_jev_weight == 0.7
    assert jina.reranker.top_n == 3
    assert jina.reranker.include_scores is True
    assert jina_no_coma.reranker.top_n == 3
    assert type(jina_no_coma.reranker).__name__ == "JevJinaTopReranker"
    assert shared.reranker.context == "shared"
    assert single.reranker.context == "single"
    assert own.reranker.context == "shared"
    assert shared.reranking_cfg["candidate_order"] == "retrieval"
    assert single.reranking_cfg["candidate_order"] == "retrieval"
    assert shared.reranker.include_dtype is False
    assert single.reranker.include_dtype is False
    assert type(shared.retrieval).__name__ == "MagnetoCandidateRetriever"
    assert type(fusion.retrieval).__name__ == "MagnetoCandidateRetriever"
    assert type(single.retrieval).__name__ == "MagnetoCandidateRetriever"
    assert type(own.retrieval).__name__ == "CandidateRetriever"


def test_magneto_adapter_keeps_upstream_pipeline_and_dataset_params(config):
    import pandas as pd

    m = matcher("magneto_qwen", config)
    m.set_case_context(CaseContext("GDC", "toy"))
    source = pd.DataFrame({"id": [1], "name": ["a"]})
    target = pd.DataFrame({"id": [1], "label": ["a"], "other": [2]})
    ranking = m.match(source, target)
    validate_ranking(ranking, list(source.columns), list(target.columns))
    params = _FakeUpstreamMagneto.last_params
    assert params["embedding_model"] == "mpnet"
    assert params["use_gpt_reranker"] is True
    assert params["use_bp_reranker"] is False
    assert params["encoding_mode"] == "header_values_default"


def test_runner_marks_failed_case_and_keeps_it_visible(mini_benchmark, fake_servers):
    config, records = mini_benchmark
    _, decision = fake_servers
    decision.mode = "out_of_range"
    counts = run(
        config, "dema_legacy", ["GDC"], matcher=matcher("dema_legacy", config), argv=["test"]
    )
    assert counts == {"skipped": 0, "success": 0, "failed": 1}
    paths = ExperimentPaths.from_config(config)
    rec = next(r for r in records if r.dataset == "GDC")
    assert unit_state(paths, "dema_legacy", rec) == "failed"
    assert not paths.prediction("dema_legacy", "GDC", rec.case_id).exists()
    status = read_json(paths.status("dema_legacy", "GDC", rec.case_id))
    assert "outside [0,1]" in status["error"]
    assert "exception" in paths.log("dema_legacy", "GDC", rec.case_id).read_text()
    decision.mode = "ok"
    counts = run(
        config, "dema_legacy", ["GDC"], matcher=matcher("dema_legacy", config), argv=["test"]
    )
    assert counts["success"] == 1 and unit_state(paths, "dema_legacy", rec) == "complete"


def test_full_run_writes_separated_artifacts_and_evaluates(mini_benchmark, fake_servers):
    from dema.metrics.evaluator import evaluate

    config, records = mini_benchmark
    for method in ("magneto_qwen", "dema_legacy", "coma"):
        m = matcher(method, config) if method != "coma" else None
        assert run(config, method, ["GDC", "OpenData"], matcher=m, argv=["t"])["failed"] == 0
    paths = ExperimentPaths.from_config(config)
    for rec in records:
        doc = read_json(paths.prediction("dema_legacy", rec.dataset, rec.case_id))
        assert len(doc["predictions"]) == rec.n_source_columns
        assert all(len(values) == rec.n_target_columns for values in doc["predictions"].values())
        assert paths.log("dema_legacy", rec.dataset, rec.case_id).is_relative_to(config.logs_dir)
        assert paths.prediction("dema_legacy", rec.dataset, rec.case_id).is_relative_to(config.saves_dir)
    manifests = list((config.saves_dir / "manifests").glob("*.json"))
    assert manifests and "git_commit" in read_json(manifests[0])
    tables = evaluate(config, ["coma", "magneto_qwen", "dema_legacy"], ["GDC", "OpenData"])
    assert len(tables["per_case"]) == 6
    assert set(tables["overall"]["weighting"]) == {"case_macro", "dataset_macro"}
    assert tables["overall"]["complete"].all()
