"""magneto_qwen and dema against fake model servers."""

import pytest

from dema.data.loader import load_case
from dema.models.generative_reranker import RerankerFailure
from dema.models.registry import build_matcher
from dema.models.ranking import validate_ranking
from dema.runner import OutputPaths, run, unit_state
from dema.utils.io import read_json

from conftest import trigram_encoder


def matcher(method, config):
    m = build_matcher(method, config)
    m.retriever._encoder = trigram_encoder  # no model download in tests
    return m


def test_both_rerankers_get_identical_inputs(mini_benchmark, fake_servers):
    config, records = mini_benchmark
    qwen, decision = fake_servers
    config.models["retriever"]["top_k"] = 3
    case = load_case(records[0])
    mq, md = matcher("magneto_qwen", config), matcher("dema", config)
    assert mq.profiles(case.source_df, case.target_df) == md.profiles(case.source_df, case.target_df)
    rq = mq.match(case.source_df, case.target_df)
    rd = md.match(case.source_df, case.target_df)
    for ranking in (rq, rd):
        validate_ranking(ranking, list(case.source_df.columns), list(case.target_df.columns))
    top = lambda r: {(m.source_column, m.target_column, m.retrieval_score) for m in r if m.reranker_score is not None}
    assert top(rq) == top(rd)  # same candidate sets and retrieval scores
    assert md.last_runtime.retrieval_cache_hit  # dema reused magneto's cached candidates
    assert md.last_runtime.model_requests == case.source_df.shape[1]
    # decision requests: one noul question per candidate, no retrieval information
    for req in decision.requests:
        assert all(q["type"] == "noul" for q in req["questions"].values())
        assert len(req["questions"]) == 3
        assert "retrieval" not in req["state"] and "score" not in req["state"]
    for req in qwen.requests:
        assert req["temperature"] == 0 and req["chat_template_kwargs"] == {"enable_thinking": False}


def test_invalid_output_retries_then_fails_without_fallback(mini_benchmark, fake_servers):
    config, records = mini_benchmark
    qwen, _ = fake_servers
    qwen.mode = "invalid_json"
    case = load_case(records[0])
    m = matcher("magneto_qwen", config)
    with pytest.raises(RerankerFailure):
        m.match(case.source_df, case.target_df)
    assert len(qwen.requests) == 2  # 1 attempt + max_retries=1


def test_runner_marks_failed_case_and_keeps_it_visible(mini_benchmark, fake_servers):
    config, records = mini_benchmark
    _, decision = fake_servers
    decision.mode = "out_of_range"
    counts = run(config, "dema", ["GDC"], matcher=matcher("dema", config), argv=["test"])
    assert counts == {"skipped": 0, "success": 0, "failed": 1}
    paths = OutputPaths(config.outputs_dir)
    rec = next(r for r in records if r.dataset == "GDC")
    assert unit_state(paths, "dema", rec) == "failed"
    assert not paths.prediction("dema", "GDC", rec.case_id).exists()
    status = read_json(paths.status("dema", "GDC", rec.case_id))
    assert "outside [0,1]" in status["error"]
    assert "exception" in paths.log("dema", "GDC", rec.case_id).read_text()
    decision.mode = "ok"
    counts = run(config, "dema", ["GDC"], matcher=matcher("dema", config), argv=["test"])
    assert counts["success"] == 1 and unit_state(paths, "dema", rec) == "complete"


def test_full_run_writes_all_outputs_and_evaluates(mini_benchmark, fake_servers):
    from dema.evaluation.evaluator import evaluate

    config, records = mini_benchmark
    for method in ("magneto_qwen", "dema", "coma"):
        m = matcher(method, config) if method != "coma" else None
        assert run(config, method, ["GDC", "OpenData"], matcher=m, argv=["t"])["failed"] == 0
    paths = OutputPaths(config.outputs_dir)
    for rec in records:
        doc = read_json(paths.prediction("dema", rec.dataset, rec.case_id))
        assert len(doc["predictions"]) == rec.n_source_columns
        assert all(len(v) == rec.n_target_columns for v in doc["predictions"].values())
        rt = read_json(paths.runtime("dema", rec.dataset, rec.case_id))
        assert rt["total_seconds"] >= rt["reranking_seconds"] >= 0
    manifests = list((config.outputs_dir / "manifests").glob("*.json"))
    assert manifests and "git_commit" in read_json(manifests[0])
    tables = evaluate(config, ["coma", "magneto_qwen", "dema"], ["GDC", "OpenData"])
    assert len(tables["per_case"]) == 6
    assert set(tables["overall"]["weighting"]) == {"case_macro", "dataset_macro"}
    assert tables["overall"]["complete"].all()
