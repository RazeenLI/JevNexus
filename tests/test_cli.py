from __future__ import annotations

from dema.cli import _gpu_plan, _worker_env, build_parser
from dema.experiments.runner import has_pending_work, run
from dema.experiments.services import required_service
from dema.model.base import BaseMatcher
from dema.model.ranking import ranking_from_scores


class NameMatcher(BaseMatcher):
    name = "coma"

    def match(self, source_df, target_df):
        scores = {
            (str(source), str(target)): float(str(source).lower() == str(target).lower())
            for source in source_df.columns
            for target in target_df.columns
        }
        return ranking_from_scores(list(source_df.columns), list(target_df.columns), scores)


def test_method_service_mapping():
    assert required_service("magneto_qwen") == "qwen"
    assert required_service("dema_fusion") == "decision"
    assert required_service("dema_no_rerank") == "decision"
    assert required_service("dema_no_struct") == "decision"
    assert required_service("dema_decision") == "decision"
    assert required_service("dema_jev_weight") == "decision"
    assert required_service("dema_jina_rerank") == "decision"
    assert required_service("dema_jina_no_coma") == "decision"
    assert required_service("dema") == "decision"
    assert required_service("dema_legacy") == "decision"
    assert required_service("dema_shared") == "decision"
    assert required_service("dema_single") == "decision"
    assert required_service("dema_own_retrieval") == "decision"
    assert required_service("coma") is None


def test_unified_single_gpu_is_shared_by_service_and_worker():
    args = build_parser().parse_args(["run", "--gpus", "1"])
    assert _gpu_plan(args, "magneto_qwen") == ("1", "1")
    assert _worker_env("1")["CUDA_VISIBLE_DEVICES"] == "1"


def test_unified_two_gpus_split_service_and_worker():
    args = build_parser().parse_args(["run", "--gpus", "1", "0"])
    assert _gpu_plan(args, "magneto_qwen") == ("1", "0")


def test_legacy_qwen_gpu_also_constrains_magneto_worker():
    args = build_parser().parse_args(["run", "--qwen-gpu", "1"])
    assert _gpu_plan(args, "magneto_qwen") == ("1", "1")


def test_complete_resume_does_not_need_service(mini_benchmark):
    config, records = mini_benchmark
    config.experiment["max_cases_per_dataset"] = None
    datasets = sorted({record.dataset for record in records})
    assert has_pending_work(config, "coma", datasets)
    counts = run(config, "coma", datasets, matcher=NameMatcher())
    assert counts["success"] == len(records)
    assert not has_pending_work(config, "coma", datasets)
    assert has_pending_work(config, "coma", datasets, resume=False)
