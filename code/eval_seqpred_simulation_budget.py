#!/usr/bin/env python3
"""Simulation budget evaluation for the SeqPredictor baseline.

Reuses eval_film_simulation_budget's evaluate_pairwise / evaluate_multi_hw
verbatim, but with SeqPredictor predictions as the reference method (model=None
so the generic |pred_delta| ranking signal is used).  Same K_values, tail_frac,
top_m, structured HW groups, and RNG seeds as the FiLM run, on the same seed-42
HW holdout.

Output is a standalone JSON and merges results into
eval_simulation_budget.json:
  pairwise.{overall,hard,medium,easy}.<task>.<K>.seqpred
  multi_hw.<task>.<K>.{top1_seqpred,pairwise_acc_seqpred,topm_acc_seqpred,regret_seqpred}
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # release code/ dir

from dataset import MicroArchDataset
from eval_film_common import (
    build_group_hw_map, hw_holdout_split, set_seed, TASK_KEYS,
    add_dataset_path_arg, resolve_data_path,
)
from eval_film_simulation_budget import (
    precompute_bbv_lookup,
    evaluate_pairwise,
    evaluate_multi_hw,
    K_VALUES,
    N_PAIRS,
    N_HW_GROUPS,
    TAIL_FRAC,
    TOP_M,
)
from seqpred.predict import load_seqpred, collect_predictions_seqpred, DEFAULT_CHECKPOINT


def _remap_pairwise(pairwise):
    """Rename method key 'model' -> 'seqpred' inside the pairwise agg blob."""
    for stratum in ("overall", "hard", "medium", "easy"):
        sd = pairwise.get(stratum, {})
        for task in TASK_KEYS:
            td = sd.get(task, {})
            for k in list(td.keys()):
                kd = td[k]
                if "model" in kd:
                    kd["seqpred"] = kd.pop("model")
                # drop simpoint/random duplicates (already in FiLM JSON)
                kd.pop("simpoint", None)
                kd.pop("random", None)
    return pairwise


def _remap_multi(multi):
    for task in TASK_KEYS:
        td = multi.get(task, {})
        for k in list(td.keys()):
            kd = td[k]
            for prefix in ("top1", "pairwise_acc", "topm_acc", "regret"):
                mk = f"{prefix}_model"
                if mk in kd:
                    kd[f"{prefix}_seqpred"] = kd.pop(mk)
                kd.pop(f"{prefix}_simpoint", None)
                kd.pop(f"{prefix}_random", None)
    return multi


def _kget(d, k):
    """Look up a K entry tolerating int/str key mismatch (JSON stringifies keys)."""
    if k in d:
        return d[k]
    if str(k) in d:
        return d[str(k)]
    try:
        ik = int(k)
        if ik in d:
            return d[ik]
    except (ValueError, TypeError):
        pass
    return None


def _inject_pairwise(canon_pw, seq_pw):
    for stratum in ("overall", "hard", "medium", "easy"):
        c_s = canon_pw.get(stratum, {})
        s_s = seq_pw.get(stratum, {})
        for task in TASK_KEYS:
            c_t = c_s.get(task, {})
            s_t = s_s.get(task, {})
            for k in c_t:
                s_k = _kget(s_t, k)
                if s_k is not None and "seqpred" in s_k:
                    c_t[k]["seqpred"] = s_k["seqpred"]


def _inject_multi(canon_m, seq_m):
    for task in TASK_KEYS:
        c_t = canon_m.get(task, {})
        s_t = seq_m.get(task, {})
        for k in c_t:
            s_k = _kget(s_t, k)
            if s_k is not None:
                for prefix in ("top1", "pairwise_acc", "topm_acc", "regret"):
                    sk = f"{prefix}_seqpred"
                    if sk in s_k:
                        c_t[k][sk] = s_k[sk]


def main():
    parser = argparse.ArgumentParser()
    add_dataset_path_arg(parser)
    parser.add_argument("--seqpred-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default="eval_seqpred_simulation_budget.json")
    parser.add_argument("--inject-into", default="eval_simulation_budget.json")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n_hw_groups", type=int, default=N_HW_GROUPS)
    parser.add_argument("--tail-frac", type=float, default=TAIL_FRAC)
    parser.add_argument("--top-m", type=int, default=TOP_M)
    parser.add_argument("--max-intervals-per-pair", type=int, default=None)
    parser.add_argument("--multi-selector", default="max_ranknorm")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset ...")
    dataset = MicroArchDataset(resolve_data_path(args.data_path))
    print("Precomputing BBV lookup ...")
    bbv_lookup = precompute_bbv_lookup(dataset)

    train_idx, val_idx, train_hw, val_hw = hw_holdout_split(dataset)
    val_idx_list = val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx)
    print(f"  HW split: {len(train_hw)} train + {len(val_hw)} test")

    print("Loading SeqPredictor ...")
    bundle = load_seqpred(args.seqpred_checkpoint, device)
    print("Running SeqPredictor inference on held-out HW samples ...")
    all_data, coverage = collect_predictions_seqpred(
        bundle, dataset, val_idx_list, device, progress_every=500
    )
    print(f"  {len(all_data)} predictions. Coverage: {coverage}")

    groups_by_method = {"model": build_group_hw_map(all_data)}
    method_models = {"model": None}  # None -> |pred_delta| ranking signal
    held_out_hw = {item["hw_id"] for item in all_data}

    print("Scenario 1: pairwise ...")
    pairwise = evaluate_pairwise(
        groups_by_method, method_models, device, bbv_lookup, held_out_hw,
        n_pairs=args.n_pairs, tail_frac=args.tail_frac,
        max_intervals_per_pair=args.max_intervals_per_pair,
    )
    print("Scenario 2: multi-hw ...")
    multi = evaluate_multi_hw(
        groups_by_method, method_models, device, bbv_lookup, held_out_hw,
        n_groups=args.n_hw_groups, structured=True, tail_frac=args.tail_frac,
        top_m=args.top_m, max_intervals_per_pair=args.max_intervals_per_pair,
        multi_selector=args.multi_selector,
    )

    pairwise = _remap_pairwise(pairwise)
    multi = _remap_multi(multi)

    out = {
        "seqpred_checkpoint": args.seqpred_checkpoint,
        "K_values": K_VALUES,
        "tail_frac": float(args.tail_frac),
        "top_m": int(args.top_m),
        "multi_selector": args.multi_selector,
        "seqpred_coverage": coverage,
        "pairwise": pairwise,
        "multi_hw": multi,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved standalone TaoPredictor simulation budget results to {args.output}")

    # headline
    try:
        pw5 = pairwise["overall"]["l2_mpki"]["5"]["seqpred"]
    except Exception:
        pw5 = pairwise["overall"].get("l2_mpki", {}).get(5, {}).get("seqpred")
    print(f"\n  TaoPredictor simulation budget pairwise overall l2_mpki @K=5 acc = {pw5}")

    if args.inject_into and os.path.exists(args.inject_into):
        with open(args.inject_into) as f:
            canon = json.load(f)
        _inject_pairwise(canon["pairwise"], pairwise)
        _inject_multi(canon["multi_hw"], multi)
        canon.setdefault("seqpred_coverage", coverage)
        with open(args.inject_into, "w") as f:
            json.dump(canon, f, indent=2)
        print(f"Injected seqpred series into {args.inject_into}")
    else:
        print(f"  (skipped injection; {args.inject_into} not found)")


if __name__ == "__main__":
    main()
