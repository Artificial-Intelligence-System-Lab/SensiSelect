#!/usr/bin/env python3
"""Cache-focused unseen-hardware generalization breakdown.

This evaluation keeps the pair-level retrieval object from single-parameter sensitivity retrieval, but
stratifies held-out hardware pairs by hardware distance to test unseen-HW
robustness. The main outputs are cache-focused retrieval metrics:

  - recall@K
  - ndcg@K

reported by close / medium / far hardware-pair strata for:

  - SensiSelect
  - Abs. Regressor
  - SimPoint
  - Random
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from eval_film_common import (
    CACHE_TASKS,
    SEED,
    TASK_KEYS,
    TASK_INDEX,
    add_model_override_args,
    add_dataset_path_arg,
    add_regressor_override_mode_arg,
    build_group_hw_map,
    collect_model_overrides,
    collect_pair_bundle,
    collect_predictions_from_indices,
    compute_pair_scores,
    load_film_model,
    ndcg_at_k,
    set_ndcg_at_k,
    set_seed,
    topk_recall,
    hw_holdout_split,
    validate_abs_regressor_checkpoint,
    DEFAULT_ABS_REGRESSOR_CHECKPOINT,
    resolve_regressor_model_overrides,
    resolve_data_path,
)

OOD_TASKS = CACHE_TASKS + ('cpi', 'rob_stall', 'branch_mpki')
NC_OOD_TASKS = tuple(t for t in OOD_TASKS if t not in CACHE_TASKS)
from dataset import MicroArchDataset

K_FRACTIONS = (0.01, 0.05, 0.10)
N_RANDOM_SEEDS = 10
N_SIMPOINT_SEEDS = 5
MAX_PAIRS_PER_STRATUM = 500


def compute_hw_l0_matrix(dataset):
    """Compute L0 distance between unique hardware configurations."""
    hw_vecs = {}
    for sample in dataset.samples:
        hwid = int(sample.get("hw_id", -1))
        if hwid not in hw_vecs:
            hw_vecs[hwid] = sample["hw"].numpy()

    hw_ids = sorted(hw_vecs.keys())
    hw_mat = np.array([hw_vecs[h] for h in hw_ids])

    l0_mat = {}
    for i in range(len(hw_ids)):
        for j in range(i + 1, len(hw_ids)):
            diff = np.abs(hw_mat[i] - hw_mat[j])
            l0 = int((diff > 1e-6).sum())
            l0_mat[(hw_ids[i], hw_ids[j])] = l0
            l0_mat[(hw_ids[j], hw_ids[i])] = l0
    return l0_mat


def precompute_bbv_lookup(dataset, sample_indices):
    """gid -> projected BBV vector for SimPoint-style selection."""
    interval_bbvs = {}
    for idx in sample_indices:
        sample = dataset.samples[idx]
        gid = sample.get("group_id", idx)
        if gid in interval_bbvs:
            continue
        if hasattr(dataset, "bbv_raw") and dataset.bbv_raw and idx < len(dataset.bbv_raw):
            bbv_x_raw, bbv_w_raw = dataset.bbv_raw[idx]
            if bbv_x_raw is not None and len(bbv_x_raw) > 0:
                bbv_x = np.array(bbv_x_raw, dtype=np.float32)
                bbv_w = np.array(bbv_w_raw, dtype=np.float32)
                if bbv_w.sum() > 0:
                    bbv_w = bbv_w / bbv_w.sum()
                interval_bbvs[gid] = (bbv_x * bbv_w[:, None]).sum(axis=0)
                continue
        interval_bbvs[gid] = np.array(sample["bbe"], dtype=np.float32)[:128]
    return interval_bbvs


def build_simpoint_for_gids(bbv_lookup, gids, k, seed):
    """BBV k-means -> K unique interval selections."""
    if k >= len(gids):
        return set(gids)
    ordered_gids = [gid for gid in gids if gid in bbv_lookup]
    if not ordered_gids:
        return set(gids[:k])
    bbv_matrix = np.array([bbv_lookup[gid] for gid in ordered_gids], dtype=np.float32)
    k = max(1, min(int(k), len(bbv_matrix)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    km.fit(bbv_matrix)

    selected = []
    used = set()
    for center in km.cluster_centers_:
        dists = np.linalg.norm(bbv_matrix - center, axis=1)
        for idx in np.argsort(dists):
            idx = int(idx)
            if idx not in used:
                selected.append(idx)
                used.add(idx)
                break

    if len(selected) < k:
        center_dists = np.linalg.norm(
            bbv_matrix[:, None, :] - km.cluster_centers_[None, :, :],
            axis=2,
        )
        nearest_center_dist = center_dists.min(axis=1)
        for idx in np.argsort(nearest_center_dist):
            idx = int(idx)
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
            if len(selected) >= k:
                break
    return {ordered_gids[i] for i in sorted(selected)}


def _metric_bucket(k_fractions):
    bucket = {"mean_delta": []}
    for k_frac in k_fractions:
        suffix = f"@{int(k_frac * 100)}%"
        bucket[f"recall{suffix}"] = []
        bucket[f"ndcg{suffix}"] = []
    return bucket


def _aggregate_metric_bucket(bucket, k_fractions):
    out = {"mean_delta": round(float(np.mean(bucket["mean_delta"])), 4) if bucket["mean_delta"] else 0.0}
    for k_frac in k_fractions:
        suffix = f"@{int(k_frac * 100)}%"
        for key in (f"recall{suffix}", f"ndcg{suffix}"):
            vals = bucket[key]
            out[key] = round(float(np.mean(vals)), 4) if vals else 0.0
    return out


def _ensure_exact_gids(method_name, expected_gids, observed_gids, context):
    """Fail loudly if a method's pair bundle is not aligned to the reference gids."""
    expected = list(expected_gids)
    observed = list(observed_gids)
    if observed == expected:
        return
    expected_set = set(expected)
    observed_set = set(observed)
    missing = [g for g in expected if g not in observed_set]
    extra = [g for g in observed if g not in expected_set]
    raise ValueError(
        f"{method_name} bundle misaligned for {context}: "
        f"expected {len(expected)} gids, got {len(observed)} "
        f"(missing={len(missing)}, extra={len(extra)})"
    )


def _precompute_simpoint_for_pair(args):
    """Compute SimPoint+random selections for one HW pair (CPU-parallel)."""
    gids, k_fractions, n_simpoint_seeds, n_random_seeds, bbv_lookup, seed_base = args
    results = {}
    rng = np.random.default_rng(seed_base + 200)
    for k_frac in k_fractions:
        k = max(1, min(int(len(gids) * k_frac), len(gids)))
        results[k_frac] = {
            "simpoint": [build_simpoint_for_gids(bbv_lookup, gids, k, SEED + i)
                         for i in range(n_simpoint_seeds)],
            "random": [set(rng.choice(gids, size=k, replace=False).tolist())
                       for _ in range(n_random_seeds)],
            "k": k,
        }
    return results


def evaluate_by_distance(
    groups_by_method,
    models_by_method,
    device,
    l0_mat,
    strata,
    bbv_lookup,
    k_fractions=K_FRACTIONS,
    n_random_seeds=N_RANDOM_SEEDS,
    n_simpoint_seeds=N_SIMPOINT_SEEDS,
    score_source="current",
):
    ref_groups = groups_by_method["model"]
    learned_methods = list(models_by_method.keys())

    sample_group = next(iter(ref_groups.values()), None)
    if sample_group is None:
        return {}
    all_hw_ids = sorted({hw_id for hw_map in ref_groups.values() for hw_id in hw_map})
    rng = np.random.default_rng(SEED)
    hw_pairs_by_stratum = {name: [] for name in strata}

    for i in range(len(all_hw_ids)):
        for j in range(i + 1, len(all_hw_ids)):
            h1, h2 = all_hw_ids[i], all_hw_ids[j]
            l0 = l0_mat.get((h1, h2), 47)
            for name, (lo, hi) in strata.items():
                if lo <= l0 <= hi:
                    hw_pairs_by_stratum[name].append((h1, h2, l0))
                    break

    # ---- Precompute/load SimPoint + random selections (cached) ----
    import pickle as _pickle
    _cache_path = "ood_simpoint_cache.pkl"
    # Sample pairs for each stratum (do this once, deterministic)
    sampled_pairs_by_stratum = {}
    for stratum_name, pairs in hw_pairs_by_stratum.items():
        if not pairs:
            sampled_pairs_by_stratum[stratum_name] = []
            continue
        sp = pairs
        if len(sp) > MAX_PAIRS_PER_STRATUM:
            idx = rng.choice(len(sp), MAX_PAIRS_PER_STRATUM, replace=False)
            sp = [sp[i] for i in idx]
        sampled_pairs_by_stratum[stratum_name] = sp

    if os.path.exists(_cache_path):
        print(f"  Loading SimPoint cache from {_cache_path} ...", flush=True)
        with open(_cache_path, "rb") as _f:
            _cached = _pickle.load(_f)
        simpoint_lookup = _cached["lookup"]
        print(f"    Loaded {len(simpoint_lookup)} cached HW pairs.", flush=True)
    else:
        print("  Precomputing SimPoint+random selections (parallel, one-time) ...", flush=True)
        all_simpoint_tasks = []
        all_pair_keys = []
        for stratum_name, sp in sampled_pairs_by_stratum.items():
            for h1, h2, _ in sp:
                gids, _, _, _, _ = collect_pair_bundle(ref_groups, h1, h2)
                gids = list(int(g) for g in (gids.tolist() if hasattr(gids, 'tolist') else gids))
                all_simpoint_tasks.append((gids, k_fractions, n_simpoint_seeds, n_random_seeds,
                                            bbv_lookup, SEED + h1 * 1000 + h2))
                all_pair_keys.append((stratum_name, h1, h2))

        n_jobs = min(os.cpu_count() or 1, len(all_simpoint_tasks), 8)
        print(f"    {len(all_simpoint_tasks)} HW pairs, {n_jobs} parallel workers", flush=True)
        _results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_precompute_simpoint_for_pair)(task) for task in all_simpoint_tasks
        )
        simpoint_lookup = {key: result for key, result in zip(all_pair_keys, _results)}
        with open(_cache_path, "wb") as _f:
            _pickle.dump({"lookup": simpoint_lookup}, _f)
        print(f"    Saved cache to {_cache_path}", flush=True)
    # --------------------------------------------------------------------------------

    results = {}
    for stratum_name, pairs in sampled_pairs_by_stratum.items():
        if not pairs:
            results[stratum_name] = {"n_pairs": 0, "note": "no pairs in this stratum"}
            continue

        n_pairs_sampled = len(pairs)

        method_keys = [*learned_methods, "simpoint", "random"]
        accum = {
            task: {method: _metric_bucket(k_fractions) for method in method_keys}
            for task in OOD_TASKS
        }
        n_pairs_valid = 0
        n_pairs_by_task = {task: 0 for task in OOD_TASKS}

        for h1, h2, _l0 in pairs:
            gids, items_a, items_b, gt_mat, pred_mat = collect_pair_bundle(ref_groups, h1, h2)
            if gt_mat is None or len(gids) < 10:
                continue
            n_pairs_valid += 1

            pair_scores_by_method = {}
            pred_abs_by_method = {}
            for method_name in learned_methods:
                mgids, mitems_a, mitems_b, _mgt, mpred_mat = collect_pair_bundle(
                    groups_by_method[method_name], h1, h2
                )
                _ensure_exact_gids(method_name, gids, mgids, f"ood ({h1}, {h2})")
                if mpred_mat is None:
                    raise ValueError(
                        f"{method_name} bundle missing prediction matrix for ood ({h1}, {h2})"
                    )
                pair_scores_by_method[method_name] = compute_pair_scores(
                    models_by_method[method_name], mitems_a, mitems_b, device
                )
                pred_abs_by_method[method_name] = np.abs(mpred_mat)

            # Look up precomputed SimPoint + random selections
            precomp = simpoint_lookup[(stratum_name, h1, h2)]

            for task in OOD_TASKS:
                t = TASK_INDEX[task]
                gt_abs = np.abs(gt_mat[:, t])

                if gt_abs.size < 10:
                    continue

                n_pairs_by_task[task] += 1
                for method_name in method_keys:
                    accum[task][method_name]["mean_delta"].append(float(gt_abs.mean()))

                gid_to_idx = {gid: idx for idx, gid in enumerate(gids)}
                for k_frac in k_fractions:
                    k = max(1, min(int(len(gids) * k_frac), len(gids)))
                    suffix = f"@{int(k_frac * 100)}%"

                    for method_name in learned_methods:
                        pred_abs = pred_abs_by_method[method_name]
                        if score_source == "endpoint":
                            rank_signal = pred_abs[:, t] if pred_abs is not None else np.zeros_like(gt_abs)
                        elif score_source == "routed":
                            # close (L0<=10): endpoint |delta|
                            # medium/far (L0>10): C3A pair scores (offline replay via cached feat/H)
                            # NOTE: affects cache tasks only; non-cache tasks fall through to |delta|
                            # since compute_pair_scores only produces C3A cache scores.
                            if _l0 <= 10:
                                rank_signal = pred_abs[:, t] if pred_abs is not None else np.zeros_like(gt_abs)
                            else:
                                rank_signal = pair_scores_by_method[method_name].get(
                                    task,
                                    pred_abs[:, t] if pred_abs is not None else np.zeros_like(gt_abs),
                                )
                        else:
                            rank_signal = pair_scores_by_method[method_name].get(
                                task,
                                pred_abs[:, t] if pred_abs is not None else np.zeros_like(gt_abs),
                            )
                        accum[task][method_name][f"recall{suffix}"].append(
                            topk_recall(gt_abs, rank_signal, k)
                        )
                        accum[task][method_name][f"ndcg{suffix}"].append(
                            ndcg_at_k(gt_abs, rank_signal, k)
                        )

                    # Use precomputed SimPoint + random selections
                    sel = precomp[k_frac]
                    sim_vals = []
                    sim_ndcg_vals = []
                    for selected_gids in sel["simpoint"]:
                        selected_idx = [gid_to_idx[g] for g in selected_gids if g in gid_to_idx]
                        sim_vals.append(topk_recall(gt_abs, np.isin(np.arange(len(gids)), selected_idx).astype(float), k))
                        sim_ndcg_vals.append(set_ndcg_at_k(gt_abs, selected_idx, k))
                    accum[task]["simpoint"][f"recall{suffix}"].append(float(np.mean(sim_vals)))
                    accum[task]["simpoint"][f"ndcg{suffix}"].append(float(np.mean(sim_ndcg_vals)))

                    rnd_vals = []
                    rnd_ndcg_vals = []
                    for selected_gids in sel["random"]:
                        selected_idx = [gid_to_idx[g] for g in selected_gids if g in gid_to_idx]
                        rnd_vals.append(topk_recall(gt_abs, np.isin(np.arange(len(gids)), selected_idx).astype(float), k))
                        rnd_ndcg_vals.append(set_ndcg_at_k(gt_abs, selected_idx, k))
                    accum[task]["random"][f"recall{suffix}"].append(float(np.mean(rnd_vals)))
                    accum[task]["random"][f"ndcg{suffix}"].append(float(np.mean(rnd_ndcg_vals)))

        if n_pairs_valid == 0:
            results[stratum_name] = {
                "n_pairs": 0,
                "n_pairs_sampled": n_pairs_sampled,
                "n_pairs_valid": 0,
                "n_pairs_by_task": n_pairs_by_task,
                "l0_range": f"{strata[stratum_name][0]}-{strata[stratum_name][1]}",
                "mean_l0": round(float(np.mean([l0 for _, _, l0 in pairs])), 1),
                "note": "no valid evaluated pairs in this stratum",
            }
            continue

        stratum_result = {
            "n_pairs": n_pairs_valid,
            "n_pairs_sampled": n_pairs_sampled,
            "n_pairs_valid": n_pairs_valid,
            "n_pairs_by_task": n_pairs_by_task,
            "l0_range": f"{strata[stratum_name][0]}-{strata[stratum_name][1]}",
            "mean_l0": round(float(np.mean([l0 for _, _, l0 in pairs])), 1),
            "metrics": {},
            "summary": {},
        }

        for task in OOD_TASKS:
            stratum_result["metrics"][task] = {
                method: _aggregate_metric_bucket(accum[task][method], k_fractions)
                for method in method_keys
            }

        cache_summary = {}
        nc_summary = {}
        for method in method_keys:
            method_summary = {}
            mean_delta_vals = [stratum_result["metrics"][task][method]["mean_delta"] for task in CACHE_TASKS]
            method_summary["mean_delta"] = round(float(np.mean(mean_delta_vals)), 4) if mean_delta_vals else 0.0
            for k_frac in k_fractions:
                suffix = f"@{int(k_frac * 100)}%"
                for key in (f"recall{suffix}", f"ndcg{suffix}"):
                    vals = [stratum_result["metrics"][task][method][key] for task in CACHE_TASKS]
                    method_summary[key] = round(float(np.mean(vals)), 4) if vals else 0.0
            cache_summary[method] = method_summary
        for method in method_keys:
            method_summary = {}
            for k_frac in k_fractions:
                suffix = f"@{int(k_frac * 100)}%"
                for key in (f"recall{suffix}", f"ndcg{suffix}"):
                    vals = [stratum_result["metrics"][task][method][key] for task in NC_OOD_TASKS]
                    method_summary[key] = round(float(np.mean(vals)), 4) if vals else 0.0
            nc_summary[method] = method_summary
        stratum_result["summary"]["cache_mean"] = cache_summary
        stratum_result["summary"]["nc_mean"] = nc_summary
        results[stratum_name] = stratum_result

    results["_meta"] = {
        "k_fractions": list(k_fractions),
        "n_random_seeds": int(n_random_seeds),
        "n_simpoint_seeds": int(n_simpoint_seeds),
        "tasks": list(OOD_TASKS),
        "strata": {name: list(bounds) for name, bounds in strata.items()},
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.getenv(
            "CHECKPOINT",
            str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"),
        ),
    )
    parser.add_argument(
        "--regressor-checkpoint",
        default=DEFAULT_ABS_REGRESSOR_CHECKPOINT,
    )
    parser.add_argument("--output", default="eval_hw_generalization.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--score-source", choices=("current", "endpoint", "routed"), default="current",
                        help="current: C3A pair scores; endpoint: |pred_delta| only; "
                             "routed: endpoint for close(L0<=10), C3A for medium/far(L0>10)")
    add_dataset_path_arg(parser)
    add_model_override_args(parser)
    add_regressor_override_mode_arg(parser)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset ...")
    dataset = MicroArchDataset(resolve_data_path(args.data_path))
    train_idx, val_idx, train_hw, val_hw = hw_holdout_split(dataset)
    val_idx_list = val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx)
    print(f"  total: {len(dataset)}  val: {len(val_idx_list)}")
    print(f"  HW split: {len(train_hw)} train HW + {len(val_hw)} test HW")

    print("Precomputing BBV lookup ...")
    bbv_lookup = precompute_bbv_lookup(dataset, val_idx_list)

    print("Computing HW L0 distance matrix ...")
    l0_mat = compute_hw_l0_matrix(dataset)
    model_overrides = collect_model_overrides(args)
    model = load_film_model(
        args.checkpoint,
        device,
        dataset,
        model_overrides=model_overrides,
    )
    regressor = None
    if args.regressor_checkpoint:
        regressor_overrides = resolve_regressor_model_overrides(
            model_overrides,
            args.regressor_override_mode,
        )
        validate_abs_regressor_checkpoint(args.regressor_checkpoint, device)
        regressor = load_film_model(
            args.regressor_checkpoint,
            device,
            dataset,
            model_overrides=regressor_overrides,
            allow_ambient_env=False,
        )

    method_models = {"model": model}
    if regressor is not None:
        method_models["regressor"] = regressor

    groups_by_method = {}
    for method_name, method_model in method_models.items():
        print(f"Running inference for {method_name} ...")
        all_data = collect_predictions_from_indices(
            method_model, dataset, val_idx_list, device, args.batch_size
        )
        print(f"  {len(all_data)} predictions")
        groups_by_method[method_name] = build_group_hw_map(all_data)

    strata = {
        "close": (0, 10),
        "medium": (11, 25),
        "far": (26, 47),
    }
    results = evaluate_by_distance(groups_by_method, method_models, device, l0_mat, strata, bbv_lookup,
                                    score_source=args.score_source)
    results["_meta"]["checkpoint"] = args.checkpoint
    results["_meta"]["regressor_checkpoint"] = args.regressor_checkpoint
    results["_meta"]["score_source"] = args.score_source

    print("\n" + "=" * 88)
    print("UNSEEN-HARDWARE GENERALIZATION BY HW DISTANCE (CACHE FOCUSED)")
    print("=" * 88)
    for name in ("close", "medium", "far"):
        result = results[name]
        if "note" in result:
            print(f"{name}: {result['note']}")
            continue
        summary = result["summary"]["cache_mean"]
        print(
            f"{name:<8} L0={result['l0_range']:<7} "
            f"sampled={result['n_pairs_sampled']:<4} valid={result['n_pairs_valid']:<4} "
            f"R@5 model/"
            f"{'reg/' if 'regressor' in summary else ''}"
            f"sim/rnd="
            f"{summary['model']['recall@5%']:.4f}/"
            f"{(f'{summary['regressor']['recall@5%']:.4f}/' if 'regressor' in summary else '')}"
            f"{summary['simpoint']['recall@5%']:.4f}/"
            f"{summary['random']['recall@5%']:.4f}  "
            f"N@5 model/"
            f"{'reg/' if 'regressor' in summary else ''}"
            f"sim/rnd="
            f"{summary['model']['ndcg@5%']:.4f}/"
            f"{(f'{summary['regressor']['ndcg@5%']:.4f}/' if 'regressor' in summary else '')}"
            f"{summary['simpoint']['ndcg@5%']:.4f}/"
            f"{summary['random']['ndcg@5%']:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
