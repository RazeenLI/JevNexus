"""Fill the candidate cache before the rerankers run.

    python -m dema.experiments.precompute [--scalability configs/scalability.yaml]

DeMa may run concurrently with other experiments. Computing every DeMa
candidate ordering once, up front, keeps embedding work out of the measured
reranking run. Both DeMa's original retriever and the controlled variants'
Magneto candidate generator have separate caches. The untouched Magneto-Qwen
baseline still owns its upstream pipeline and does not use either cache. Uses
exactly the cache keys of DeMa:
formal cases (dataset, case_id) and, optionally, the scalability units
(dataset, ``<case_id>__t<size>_r<rep>``) of the reranking methods. Idempotent:
existing entries are cache hits.
"""

from __future__ import annotations

import argparse
import sys

from ..data.loader import load_case
from ..data.manifest import read_manifest, select_cases
from ..model.registry import make_magneto_retriever, make_retriever
from ..model.representation.profile import profile_table
from .scalability import scalability_records, unit_plan
from ..utils.config import load_config, load_yaml, resolve_path
from ..utils.logging import get_console_logger

log = get_console_logger("dema.experiments.precompute")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--scalability", default=None, help="also precompute units of this scalability config")
    args = p.parse_args(argv)
    config = load_config(args.config, args.config_dir)
    rep_cfg = config.section("representation")
    top_k = int(config.section("retriever")["top_k"])
    own_retriever = make_retriever(config)
    magneto_retriever = make_magneto_retriever(config)

    def fill_own(dataset, case_id, source_df, target_df):
        own_retriever.rank_all(profile_table(source_df, rep_cfg), profile_table(target_df, rep_cfg),
                               top_k, dataset=dataset, case_id=case_id)
        return own_retriever.last_cache_hit

    def fill_magneto(dataset, case_id, source_df, target_df):
        magneto_retriever.rank_dataframes(
            source_df, target_df, top_k, dataset=dataset, case_id=case_id
        )
        return magneto_retriever.last_cache_hit

    methods = set(config.experiment["methods"])
    fills = []
    if methods & {
        "dema", "dema_no_rerank", "dema_no_struct", "dema_decision",
        "dema_fusion", "dema_jev_weight", "dema_jina_rerank", "dema_jina_no_coma",
        "dema_shared", "dema_single",
    }:
        fills.append(fill_magneto)
    if methods & {"dema_own_retrieval", "dema_legacy"}:
        fills.append(fill_own)

    hits = computed = 0
    if fills:
        for record in select_cases(read_manifest(config.manifest_path), config.experiment["datasets"]):
            case = load_case(record)
            for fill in fills:
                hit = fill(record.dataset, record.case_id, case.source_df, case.target_df)
                hits, computed = hits + hit, computed + (not hit)
    if args.scalability:
        scfg = load_yaml(resolve_path(args.scalability))
        scalability_methods = set(scfg["methods"])
        scalability_fills = []
        if scalability_methods & {
            "dema", "dema_no_rerank", "dema_no_struct", "dema_decision",
            "dema_fusion", "dema_jev_weight", "dema_jina_rerank", "dema_jina_no_coma",
            "dema_shared", "dema_single",
        }:
            scalability_fills.append(fill_magneto)
        if scalability_methods & {"dema_own_retrieval", "dema_legacy"}:
            scalability_fills.append(fill_own)
        if scalability_fills:
            seed = int(scfg.get("seed", 42))
            for record in scalability_records(config, scfg):
                case = load_case(record)
                for size in scfg["target_sizes"]:
                    for rep in range(int(scfg["repeats"])):
                        unit_id, _, subset, _ = unit_plan(case, int(size), rep, seed)
                        if subset is None:
                            continue
                        for fill in scalability_fills:
                            hit = fill(record.dataset, unit_id, case.source_df, case.target_df[subset])
                            hits, computed = hits + hit, computed + (not hit)
    log.info("candidate cache ready: %d computed, %d already cached", computed, hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
