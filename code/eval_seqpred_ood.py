#!/usr/bin/env python3
"""Unseen-hardware generalization for the SeqPredictor baseline.

Reuses eval_film_ood.evaluate_by_distance verbatim with SeqPredictor as the
reference method (model=None -> |pred_delta| ranking signal).  Same strata
(close/medium/far), same K fractions, same RNG seeds, same seed-42 HW holdout.

Produces a standalone JSON and merges per-stratum SeqPred metrics into the
canonical eval_hw_generalization.json:
  <stratum>.metrics.<task>.seqpred
  <stratum>.summary.{cache_mean,nc_mean}.seqpred
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # release code/ dir

from dataset import MicroArchDataset
from eval_film_common import (
    build_group_hw_map, hw_holdout_split, set_seed,
    add_dataset_path_arg, resolve_data_path,
)
from eval_film_ood import (
    compute_hw_l0_matrix,
    precompute_bbv_lookup,
    evaluate_by_distance,
    OOD_TASKS,
    K_FRACTIONS,
)
from seqpred.predict import load_seqpred, collect_predictions_seqpred, DEFAULT_CHECKPOINT


def _remap_and_inject(canon, seq_results, k_fractions):
    """Copy seqpred's 'model' metrics into canon under the 'seqpred' method key."""
    for stratum in ("close", "medium", "far"):
        c = canon.get(stratum, {})
        s = seq_results.get(stratum, {})
        if "metrics" not in s or "metrics" not in c:
            continue
        for task in OOD_TASKS:
            c_task = c["metrics"].get(task)
            s_task = s["metrics"].get(task)
            if isinstance(c_task, dict) and isinstance(s_task, dict) and "model" in s_task:
                c_task["seqpred"] = s_task["model"]
        for group in ("cache_mean", "nc_mean"):
            c_sum = c.get("summary", {}).get(group)
            s_sum = s.get("summary", {}).get(group)
            if isinstance(c_sum, dict) and isinstance(s_sum, dict) and "model" in s_sum:
                c_sum["seqpred"] = s_sum["model"]


def main():
    parser = argparse.ArgumentParser()
    add_dataset_path_arg(parser)
    parser.add_argument("--seqpred-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default="eval_seqpred_ood.json")
    parser.add_argument("--inject-into", default="eval_hw_generalization.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--score-source", choices=("current", "endpoint", "routed"),
                        default="endpoint",
                        help="SeqPred has no C3A scorer; endpoint = |pred_delta|.")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset ...")
    dataset = MicroArchDataset(resolve_data_path(args.data_path))
    train_idx, val_idx, train_hw, val_hw = hw_holdout_split(dataset)
    val_idx_list = val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx)
    print(f"  total: {len(dataset)}  val: {len(val_idx_list)}  test HW: {len(val_hw)}")

    print("Precomputing BBV lookup ...")
    bbv_lookup = precompute_bbv_lookup(dataset, val_idx_list)
    print("Computing HW L0 distance matrix ...")
    l0_mat = compute_hw_l0_matrix(dataset)

    print("Loading SeqPredictor ...")
    bundle = load_seqpred(args.seqpred_checkpoint, device)
    print("Running SeqPredictor inference on held-out HW samples ...")
    all_data, coverage = collect_predictions_seqpred(
        bundle, dataset, val_idx_list, device, progress_every=500
    )
    print(f"  {len(all_data)} predictions. Coverage: {coverage}")

    groups_by_method = {"model": build_group_hw_map(all_data)}
    method_models = {"model": None}
    strata = {"close": (0, 10), "medium": (11, 25), "far": (26, 47)}

    # Use a separate SimPoint cache so we never collide with the FiLM run's cache
    # (SeqPred's per-pair gids can differ slightly due to dropped intervals).
    cache_path = "ood_simpoint_cache.pkl"
    seq_cache_path = "ood_simpoint_cache_seqpred.pkl"
    saved_film_cache = None
    if os.path.exists(cache_path):
        saved_film_cache = cache_path + ".film_bak"
        os.rename(cache_path, saved_film_cache)
    if os.path.exists(seq_cache_path):
        os.rename(seq_cache_path, cache_path)

    try:
        results = evaluate_by_distance(
            groups_by_method, method_models, device, l0_mat, strata, bbv_lookup,
            score_source=args.score_source,
        )
    finally:
        if os.path.exists(cache_path):
            os.rename(cache_path, seq_cache_path)
        if saved_film_cache and os.path.exists(saved_film_cache):
            os.rename(saved_film_cache, cache_path)

    results.setdefault("_meta", {})["seqpred_checkpoint"] = args.seqpred_checkpoint
    results["_meta"]["score_source"] = args.score_source
    results["seqpred_coverage"] = coverage

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved standalone TaoPredictor unseen-hardware generalization results to {args.output}")

    for name in ("close", "medium", "far"):
        r = results.get(name, {})
        if "summary" in r:
            cm = r["summary"]["cache_mean"]["model"]
            print(f"  [{name}] SeqPred cache-mean recall@5% = {cm.get('recall@5%')}  "
                  f"ndcg@5% = {cm.get('ndcg@5%')}")

    if args.inject_into and os.path.exists(args.inject_into):
        with open(args.inject_into) as f:
            canon = json.load(f)
        _remap_and_inject(canon, results, K_FRACTIONS)
        with open(args.inject_into, "w") as f:
            json.dump(canon, f, indent=2)
        print(f"Injected seqpred series into {args.inject_into}")
    else:
        print(f"  (skipped injection; {args.inject_into} not found)")


if __name__ == "__main__":
    main()
