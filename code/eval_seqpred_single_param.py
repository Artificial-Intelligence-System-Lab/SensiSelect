#!/usr/bin/env python3
"""Single-parameter sensitivity retrieval for the SeqPredictor
baseline.

Reuses the FiLM eval machinery verbatim (eval_film_single_param +
eval_film_common) but swaps the prediction source to SeqPredictor.  SeqPred
predicts in its own normalized label space; the GT (label) is taken from the
same FiLM dataset samples, so ground-truth deltas are identical and the
rank-based metrics are directly comparable.

Output (default eval_seqpred_single_param.json) has the same structure as the
FiLM `methods.model` blob: {overall, per_parameter, n_evaluated, coverage}.
It is also merged as `methods.seqpred` into the canonical
eval_single_param_sweep.json so the figure picks it up automatically.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # release code/ dir

from eval_film_common import TASK_KEYS, CACHE_TASKS, SEED, set_seed, build_group_hw_map, add_sweep_data_args
from eval_film_single_param import (
    load_sweep_dataset,
    evaluate_sweep_pairs,
    aggregate_results,
    parse_k_fractions,
    DEFAULT_K_FRACTIONS,
)
from seqpred.predict import load_seqpred, collect_predictions_seqpred, DEFAULT_CHECKPOINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqpred-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default="eval_seqpred_single_param.json")
    parser.add_argument("--inject-into", default="eval_single_param_sweep.json",
                        help="canonical FiLM JSON to inject methods.seqpred into")
    parser.add_argument("--k-fractions", default=None)
    parser.add_argument("--max-sweeps", type=int, default=0,
                        help="if >0, only evaluate the first N sweeps (smoke)")
    add_sweep_data_args(parser)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_fractions = parse_k_fractions(args.k_fractions)

    print("Loading dataset with single-param sweep configs ...")
    ds, base_id_map, sweeps, sweep_id_list = load_sweep_dataset(
        args.data_path,
        args.sweep_meta_path,
        args.sweep_config_path,
    )
    if args.max_sweeps > 0:
        sweeps = sweeps[: args.max_sweeps]
    all_sweep_hw_ids = set(base_id_map.values()) | set(sweep_id_list)
    sweep_indices = [i for i, s in enumerate(ds.samples) if s.get("hw_id", 0) in all_sweep_hw_ids]
    print(f"Sweep samples: {len(sweep_indices)} / {len(ds.samples)}")

    print("Loading SeqPredictor ...")
    bundle = load_seqpred(args.seqpred_checkpoint, device)

    print("Running SeqPredictor inference on sweep samples ...")
    all_data, coverage = collect_predictions_seqpred(
        bundle, ds, sweep_indices, device, progress_every=200
    )
    print(f"Collected {len(all_data)} SeqPred predictions. Coverage: {coverage}")

    # model=None -> generic |pred_delta| ranking signal (same as baselines path).
    overall_raw, param_raw, n_evaluated, _pairs = evaluate_sweep_pairs(
        all_data, base_id_map, sweeps, sweep_id_list, None, device,
        k_fractions=k_fractions,
    )
    overall = aggregate_results(overall_raw, k_fractions=k_fractions)
    param_results = {}
    for param, raw in param_raw.items():
        param_results[param] = aggregate_results(raw, k_fractions=k_fractions)
        param_results[param]["n_pairs"] = raw["n_pairs"]

    blob = {
        "label": "SeqPredictor",
        "checkpoint": args.seqpred_checkpoint,
        "overall": overall,
        "per_parameter": param_results,
        "n_evaluated": n_evaluated,
        "seqpred_coverage": coverage,
    }

    with open(args.output, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"Saved standalone TaoPredictor sensitivity retrieval results to {args.output}")

    # Headline numbers (cache-mean recall@5%).
    cm = overall["cache_mean"]
    print(f"\n  TaoPredictor sensitivity retrieval cache-mean recall@5% = {cm.get('recall@5%')}  "
          f"cache-mean spearman = {cm.get('spearman')}  n_evaluated = {n_evaluated}")

    # Inject into the canonical FiLM JSON as methods.seqpred (if present).
    if args.inject_into and os.path.exists(args.inject_into):
        with open(args.inject_into) as f:
            canon = json.load(f)
        canon.setdefault("methods", {})
        canon["methods"]["seqpred"] = blob
        with open(args.inject_into, "w") as f:
            json.dump(canon, f, indent=2)
        print(f"Injected methods.seqpred into {args.inject_into}")
        # sanity: compare headline vs FiLM 'model'
        try:
            film_cm = canon["methods"]["model"]["overall"]["cache_mean"]
            print(f"  COMPARE cache-mean recall@5%: SeqPred={cm.get('recall@5%')} "
                  f"vs Ours(model)={film_cm.get('recall@5%')}")
        except Exception:
            pass
    else:
        print(f"  (skipped injection; {args.inject_into} not found)")


if __name__ == "__main__":
    main()
