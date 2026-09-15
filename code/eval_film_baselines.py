#!/usr/bin/env python3
"""SimPoint + Random baselines on single-parameter sweep data.

Evaluates representative and random interval selection without a learned model.
Optionally reads SensiSelect results for a printed comparison.

Usage:
    python eval_film_baselines.py [--output PATH]
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

from eval_film_common import (
    TASK_KEYS,
    CACHE_TASKS,
    SEED,
    set_seed,
    set_ndcg_at_k,
    add_sweep_data_args,
    resolve_data_path,
)
from dataset import MicroArchDataset

N_RANDOM_SEEDS = 10
DEFAULT_K_FRACTIONS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)


def _normalized_percent(value):
    pct = value * 100.0
    pct_int = int(round(pct))
    if abs(pct - pct_int) > 1e-8:
        raise ValueError(
            f"Unsupported k fraction {value}; only integer-percentage budgets are allowed."
        )
    return pct_int


def parse_k_fractions(raw, default=DEFAULT_K_FRACTIONS):
    if raw is None:
        return [v for v in default]
    values = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Invalid k fraction {value}; expected 0 < k <= 1.")
        pct_int = _normalized_percent(value)
        values.append(pct_int / 100.0)
    if not values:
        raise ValueError("At least one k fraction is required.")
    return sorted(set(values))


def budget_key(prefix, k_frac):
    return f'{prefix}@{_normalized_percent(k_frac)}%'


def labels_to_numpy(labels):
    if hasattr(labels, "detach"):
        return labels.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(labels, dtype=np.float32)


def load_sweep_dataset(data_path, sweep_meta_path, sweep_config_path):
    ds = MicroArchDataset(resolve_data_path(data_path))
    with open(resolve_data_path(sweep_meta_path)) as f:
        meta = json.load(f)
    with open(resolve_data_path(sweep_config_path)) as f:
        sweep_configs = json.load(f)
    base_id_map = {}
    for cfg in sweep_configs:
        if cfg['name'].startswith('Base_'):
            base_id_map[cfg['name'].replace('Base_', '')] = cfg['id']
    sweep_id_list = [cfg['id'] for cfg in sweep_configs if not cfg['name'].startswith('Base_')]
    sweeps = meta['sweeps']
    # SWEEP_ID_WHITELIST: restrict to the held-out single-parameter configs (see
    # eval_film_single_param.load_sweep_dataset); keeps baselines on the same set.
    _wl = os.getenv("SWEEP_ID_WHITELIST", "").strip()
    if _wl:
        wl = {int(x) for x in _wl.replace(",", " ").split() if x.strip()}
        kept = [(sw, sid) for sw, sid in zip(sweeps, sweep_id_list) if sid in wl]
        sweeps = [k[0] for k in kept]
        sweep_id_list = [k[1] for k in kept]
        print(f"SWEEP_ID_WHITELIST: restricted to {len(sweep_id_list)} held-out sweep configs.")
    return ds, base_id_map, sweeps, sweep_id_list


def build_bbv_map(dataset, hw_ids_of_interest):
    interval_bbvs = {}
    for idx, sample in enumerate(dataset.samples):
        if sample.get('hw_id', 0) not in hw_ids_of_interest:
            continue
        gid = sample.get('group_id', idx)
        if gid in interval_bbvs:
            continue
        if dataset.bbv_raw and idx < len(dataset.bbv_raw):
            bbv_x_raw, bbv_w_raw = dataset.bbv_raw[idx]
            if bbv_x_raw is not None and len(bbv_x_raw) > 0:
                bbv_x = np.array(bbv_x_raw, dtype=np.float32)
                bbv_w = np.array(bbv_w_raw, dtype=np.float32)
                if bbv_w.sum() > 0:
                    bbv_w = bbv_w / bbv_w.sum()
                interval_bbvs[gid] = (bbv_x * bbv_w[:, None]).sum(axis=0)
            else:
                interval_bbvs[gid] = np.array(sample['bbe'], dtype=np.float32)[:128]
        else:
            interval_bbvs[gid] = np.array(sample['bbe'], dtype=np.float32)[:128]

    gids = sorted(interval_bbvs.keys())
    bbv_mat = np.array([interval_bbvs[g] for g in gids])

    np.random.seed(SEED)
    proj = np.random.randn(min(128, bbv_mat.shape[1]), 20).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    bbv_20 = bbv_mat[:, :proj.shape[0]] @ proj

    return {gid: bbv_20[i] for i, gid in enumerate(gids)}


def simpoint_select(gid_to_bbv, valid_gids, k):
    ordered_gids = [gid for gid in valid_gids if gid in gid_to_bbv]
    if not ordered_gids:
        return set()
    bbv_matrix = np.array([gid_to_bbv[gid] for gid in ordered_gids], dtype=np.float32)
    k = min(k, len(bbv_matrix))
    if k < 1:
        return set()
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
    km.fit(bbv_matrix)
    selected = []
    used = set()
    for c in km.cluster_centers_:
        dists = np.linalg.norm(bbv_matrix - c, axis=1)
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


def evaluate_baselines(dataset, base_id_map, sweeps, sweep_id_list,
                       k_fractions=DEFAULT_K_FRACTIONS):
    all_sweep_hw_ids = set(base_id_map.values()) | set(sweep_id_list)

    groups = defaultdict(dict)
    for s in dataset.samples:
        if s.get('hw_id', 0) in all_sweep_hw_ids:
            groups[s.get('group_id', 0)][s.get('hw_id', 0)] = labels_to_numpy(s['labels'])

    print("Building BBV matrix for sweep intervals...")
    gid_to_bbv = build_bbv_map(dataset, all_sweep_hw_ids)
    gids = sorted(gid_to_bbv.keys())
    n_intervals = len(gids)
    print(f"  {n_intervals} unique intervals")

    simpoint_results = {
        'spearman': {t: [] for t in TASK_KEYS},
        'recall': {t: {budget_key('recall', k): [] for k in k_fractions} for t in TASK_KEYS},
        'ndcg': {t: {budget_key('ndcg', k): [] for k in k_fractions} for t in TASK_KEYS},
        'mean_delta': {t: [] for t in TASK_KEYS},
    }
    random_results = {
        'spearman': {t: [] for t in TASK_KEYS},
        'recall': {t: {budget_key('recall', k): [] for k in k_fractions} for t in TASK_KEYS},
        'ndcg': {t: {budget_key('ndcg', k): [] for k in k_fractions} for t in TASK_KEYS},
        'mean_delta': {t: [] for t in TASK_KEYS},
    }

    random_rngs = [np.random.default_rng(SEED + i) for i in range(N_RANDOM_SEEDS)]
    n_evaluated = 0

    for sw_idx, sw in enumerate(sweeps):
        if sw_idx >= len(sweep_id_list):
            break
        sweep_hwid = sweep_id_list[sw_idx]
        base_hwid = base_id_map[sw['base']]

        valid_gids = [g for g in gids if base_hwid in groups[g] and sweep_hwid in groups[g]]
        if len(valid_gids) < 10:
            continue

        n_evaluated += 1
        simpoint_selected_by_frac = {
            k_frac: simpoint_select(gid_to_bbv, valid_gids, max(1, int(len(valid_gids) * k_frac)))
            for k_frac in k_fractions
        }
        random_selected_by_frac = {
            k_frac: [
                set(rng.choice(len(valid_gids), max(1, int(len(valid_gids) * k_frac)), replace=False))
                for rng in random_rngs
            ]
            for k_frac in k_fractions
        }

        for t, task in enumerate(TASK_KEYS):
            gt_abs = np.array([abs(groups[g][base_hwid][t] - groups[g][sweep_hwid][t])
                               for g in valid_gids])

            # SimPoint
            for k_frac in k_fractions:
                key = budget_key('recall', k_frac)
                ndcg_key = budget_key('ndcg', k_frac)
                topk = max(1, int(len(valid_gids) * k_frac))
                gt_topk_gids = set(valid_gids[i] for i in np.argsort(gt_abs)[-topk:])
                sp_recall = len(gt_topk_gids & simpoint_selected_by_frac[k_frac]) / len(gt_topk_gids)
                simpoint_results['recall'][task][key].append(sp_recall)
                sp_idx = [i for i, gid in enumerate(valid_gids) if gid in simpoint_selected_by_frac[k_frac]]
                simpoint_results['ndcg'][task][ndcg_key].append(set_ndcg_at_k(gt_abs, sp_idx, topk))

            simpoint_results['mean_delta'][task].append(gt_abs.mean())

            # Random
            for k_frac in k_fractions:
                key = budget_key('recall', k_frac)
                ndcg_key = budget_key('ndcg', k_frac)
                topk = max(1, int(len(valid_gids) * k_frac))
                gt_topk_idx = set(np.argsort(gt_abs)[-topk:])
                seed_recalls = []
                seed_ndcgs = []
                for rand_idx in random_selected_by_frac[k_frac]:
                    seed_recalls.append(len(gt_topk_idx & rand_idx) / len(gt_topk_idx))
                    seed_ndcgs.append(set_ndcg_at_k(gt_abs, rand_idx, topk))
                random_results['recall'][task][key].append(float(np.mean(seed_recalls)))
                random_results['ndcg'][task][ndcg_key].append(float(np.mean(seed_ndcgs)))

            random_results['mean_delta'][task].append(gt_abs.mean())

    return simpoint_results, random_results, n_evaluated


def aggregate(raw, k_fractions=DEFAULT_K_FRACTIONS):
    def _mean_over(tasks, key):
        vals = [agg[t][key] for t in tasks if t in agg]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    agg = {}
    for task in TASK_KEYS:
        agg[task] = {}
        vals = raw['spearman'].get(task, [])
        agg[task]['spearman'] = round(float(np.mean(vals)), 4) if vals else None
        for k_frac in k_fractions:
            recall_key = budget_key('recall', k_frac)
            ndcg_key = budget_key('ndcg', k_frac)
            recall_vals = raw['recall'][task][recall_key]
            ndcg_vals = raw['ndcg'][task][ndcg_key]
            agg[task][recall_key] = round(float(np.mean(recall_vals)), 4) if recall_vals else 0.0
            agg[task][ndcg_key] = round(float(np.mean(ndcg_vals)), 4) if ndcg_vals else 0.0
        agg[task]['mean_delta'] = round(float(np.mean(raw['mean_delta'][task])), 4) if raw['mean_delta'][task] else 0.0
    agg['mean'] = {}
    mean_spearman_vals = [agg[t]['spearman'] for t in TASK_KEYS if agg[t]['spearman'] is not None]
    agg['mean']['spearman'] = round(float(np.mean(mean_spearman_vals)), 4) if mean_spearman_vals else None
    for k_frac in k_fractions:
        recall_key = budget_key('recall', k_frac)
        ndcg_key = budget_key('ndcg', k_frac)
        agg['mean'][recall_key] = round(np.mean([agg[t][recall_key] for t in TASK_KEYS]), 4)
        agg['mean'][ndcg_key] = round(np.mean([agg[t][ndcg_key] for t in TASK_KEYS]), 4)
    agg['mean']['mean_delta'] = round(np.mean([agg[t]['mean_delta'] for t in TASK_KEYS]), 4)
    cache_spearman_vals = [agg[t]['spearman'] for t in CACHE_TASKS if agg[t]['spearman'] is not None]
    agg['cache_mean'] = {
        'spearman': round(float(np.mean(cache_spearman_vals)), 4) if cache_spearman_vals else None
    }
    for k_frac in k_fractions:
        recall_key = budget_key('recall', k_frac)
        ndcg_key = budget_key('ndcg', k_frac)
        agg['cache_mean'][recall_key] = _mean_over(CACHE_TASKS, recall_key)
        agg['cache_mean'][ndcg_key] = _mean_over(CACHE_TASKS, ndcg_key)
    agg['cache_mean']['mean_delta'] = _mean_over(CACHE_TASKS, 'mean_delta')
    return agg


def primary_recall_key(k_fractions):
    return budget_key('recall', 0.05) if 0.05 in k_fractions else budget_key('recall', k_fractions[-1])


def infer_available_recall_keys(overall_blob):
    return sorted(
        [key for key in overall_blob if key.startswith('recall@')],
        key=lambda x: float(x.split('@', 1)[1].rstrip('%')),
    )


def resolve_display_recall_key(k_fractions, ours_blob):
    preferred = primary_recall_key(k_fractions)
    if not ours_blob:
        return preferred

    available = infer_available_recall_keys(ours_blob)
    if preferred in available:
        return preferred

    requested = [budget_key('recall', frac) for frac in k_fractions]
    overlap = [key for key in requested if key in available]
    if overlap:
        return primary_recall_key([float(key.split('@', 1)[1].rstrip('%')) / 100.0 for key in overlap])

    return available[-1] if available else preferred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='eval_single_param_baselines.json')
    parser.add_argument(
        '--ours-results-path',
        default='eval_single_param_sweep.json',
        help='Path to the main-model single-parameter evaluation JSON for comparison.',
    )
    parser.add_argument('--k-fractions', default=None,
                        help='Comma-separated top-K fractions, e.g. 0.01,0.02,0.03,0.05,0.07,0.10')
    add_sweep_data_args(parser)
    args = parser.parse_args()

    set_seed()
    k_fractions = parse_k_fractions(args.k_fractions)
    recall_display_key = primary_recall_key(k_fractions)

    ds, base_id_map, sweeps, sweep_id_list = load_sweep_dataset(
        args.data_path,
        args.sweep_meta_path,
        args.sweep_config_path,
    )
    print(f"Bases: {base_id_map}")
    print(f"Sweeps: {len(sweeps)}")

    sp_raw, rand_raw, n_eval = evaluate_baselines(
        ds, base_id_map, sweeps, sweep_id_list, k_fractions=k_fractions
    )
    print(f"\nEvaluated {n_eval} (base, sweep) pairs")

    sp_agg = aggregate(sp_raw, k_fractions=k_fractions)
    rand_agg = aggregate(rand_raw, k_fractions=k_fractions)

    # Load current-model results for comparison
    try:
        with open(args.ours_results_path) as f:
            ours = json.load(f)['overall']
    except Exception:
        ours = None
    recall_display_key = resolve_display_recall_key(k_fractions, ours)

    print("\n" + "=" * 90)
    print("SINGLE-PARAMETER SWEEP: Method Comparison")
    print("=" * 90)
    recall_display_label = f"--- {recall_display_key.replace('recall@', 'Recall@')} ---"
    print(f"{'':15} {'--- XI-Spearman ---':>36}  |  {recall_display_label:>30}")
    print("-" * 90)
    for task in TASK_KEYS:
        ours_sp = ours[task]['spearman'] if ours else 0
        sp_sp = sp_agg[task]['spearman']
        sp_sp_str = f"{sp_sp:.4f}" if sp_sp is not None else "N/A"
        ours_r5 = ours[task][recall_display_key] if ours else 0
        sp_r5 = sp_agg[task][recall_display_key]
        rand_r5 = rand_agg[task][recall_display_key]
        print(f"{task:<15} {ours_sp:>12.4f} {sp_sp_str:>12} {'N/A':>12}  |  "
              f"{ours_r5:>10.4f} {sp_r5:>10.4f} {rand_r5:>10.4f}")

    m_ours = ours['mean'] if ours else {}
    m_sp = sp_agg['mean']
    m_rand = rand_agg['mean']
    m_sp_str = f"{m_sp['spearman']:.4f}" if m_sp['spearman'] is not None else "N/A"
    print("-" * 90)
    print(f"{'Mean':<15} {m_ours.get('spearman', 0):>12.4f} {m_sp_str:>12} {'N/A':>12}  |  "
          f"{m_ours.get(recall_display_key, 0):>10.4f} {m_sp[recall_display_key]:>10.4f} {m_rand[recall_display_key]:>10.4f}")

    results = {
        'simpoint': sp_agg,
        'random': rand_agg,
        'n_evaluated': n_eval,
        'n_random_seeds': N_RANDOM_SEEDS,
        'k_fractions': list(k_fractions),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
