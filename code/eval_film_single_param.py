#!/usr/bin/env python3
"""Single-Parameter Sensitivity Retrieval for FiLM model.

Usage:
    python eval_film_single_param.py [--checkpoint PATH] [--output PATH]
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import spearmanr

from eval_film_common import (
    TASK_KEYS, CACHE_TASKS, SEED, set_seed,
    load_film_model, collect_predictions_from_indices, build_group_hw_map,
    collect_pair_bundle, compute_pair_scores, ndcg_at_k,
    add_model_override_args, collect_model_overrides,
    add_sweep_data_args,
    add_regressor_override_mode_arg, resolve_regressor_model_overrides,
    validate_abs_regressor_checkpoint, DEFAULT_ABS_REGRESSOR_CHECKPOINT,
    resolve_checkpoint_path, resolve_data_path,
)
from dataset import MicroArchDataset

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
    sweeps = meta['sweeps']
    sweep_id_list = [cfg['id'] for cfg in sweep_configs if not cfg['name'].startswith('Base_')]
    # SWEEP_ID_WHITELIST: restrict to specific sweep config ids so single-parameter
    # retrieval / ablation is evaluated only on the held-out configurations (those not
    # used during training). The paper reports this held-out single-parameter set.
    _wl = os.getenv("SWEEP_ID_WHITELIST", "").strip()
    if _wl:
        wl = {int(x) for x in _wl.replace(",", " ").split() if x.strip()}
        kept = [(sw, sid) for sw, sid in zip(sweeps, sweep_id_list) if sid in wl]
        sweeps = [k[0] for k in kept]
        sweep_id_list = [k[1] for k in kept]
        print(f"SWEEP_ID_WHITELIST: restricted to {len(sweep_id_list)} held-out sweep configs: "
              f"{sorted(sweep_id_list)}")
    return ds, base_id_map, sweeps, sweep_id_list


def evaluate_sweep_pairs(all_data, base_id_map, sweeps, sweep_id_list, model, device,
                         k_fractions=DEFAULT_K_FRACTIONS):
    groups = build_group_hw_map(all_data)

    def _eval_pair(base_hwid, sweep_hwid):
        gids, items_a, items_b, gt_mat, pred_mat = collect_pair_bundle(groups, base_hwid, sweep_hwid)
        if gt_mat is None or len(gids) < 10:
            return None
        pair_scores = compute_pair_scores(model, items_a, items_b, device)
        result = {}
        for t, task in enumerate(TASK_KEYS):
            gt_abs = np.abs(gt_mat[:, t])
            pred_abs = np.abs(pred_mat[:, t])
            rank_signal = pair_scores.get(task, pred_abs)
            tr = {}
            if gt_abs.std() > 1e-8 and rank_signal.std() > 1e-8:
                sp, _ = spearmanr(gt_abs, rank_signal)
                tr['spearman'] = float(sp) if not np.isnan(sp) else None
            else:
                tr['spearman'] = None
            for k_frac in k_fractions:
                k = max(1, int(len(gids) * k_frac))
                gt_topk = set(np.argsort(gt_abs)[-k:])
                pred_topk = set(np.argsort(rank_signal)[-k:])
                tr[budget_key('recall', k_frac)] = len(gt_topk & pred_topk) / len(gt_topk)
                tr[budget_key('ndcg', k_frac)] = ndcg_at_k(gt_abs, rank_signal, k)
            gt_sign = np.sign(gt_mat[:, t])
            pred_sign = np.sign(pred_mat[:, t])
            nonzero = gt_sign != 0
            tr['dir_acc'] = float((gt_sign[nonzero] == pred_sign[nonzero]).mean()) if nonzero.sum() > 0 else None
            tr['mean_delta'] = float(gt_abs.mean())
            result[task] = tr
        return result

    param_results = defaultdict(lambda: {
        'spearman': {t: [] for t in TASK_KEYS},
        'recall': {t: {budget_key('recall', k): [] for k in k_fractions} for t in TASK_KEYS},
        'ndcg': {t: {budget_key('ndcg', k): [] for k in k_fractions} for t in TASK_KEYS},
        'dir_acc': {t: [] for t in TASK_KEYS},
        'n_pairs': 0,
        'mean_delta': {t: [] for t in TASK_KEYS},
    })
    overall_raw = {
        'spearman': {t: [] for t in TASK_KEYS},
        'recall': {t: {budget_key('recall', k): [] for k in k_fractions} for t in TASK_KEYS},
        'ndcg': {t: {budget_key('ndcg', k): [] for k in k_fractions} for t in TASK_KEYS},
        'dir_acc': {t: [] for t in TASK_KEYS},
        'mean_delta': {t: [] for t in TASK_KEYS},
    }

    n_evaluated = 0
    evaluated_pairs = set()
    for sw_idx, sw in enumerate(sweeps):
        if sw_idx >= len(sweep_id_list):
            break
        sweep_hwid = sweep_id_list[sw_idx]
        base_hwid = base_id_map[sw['base']]
        param = sw['swept_param']

        result = _eval_pair(base_hwid, sweep_hwid)
        if result is None:
            continue

        n_evaluated += 1
        evaluated_pairs.add((int(base_hwid), int(sweep_hwid), str(param)))
        param_results[param]['n_pairs'] += 1

        for task in TASK_KEYS:
            tr = result[task]
            if tr['spearman'] is not None:
                overall_raw['spearman'][task].append(tr['spearman'])
                param_results[param]['spearman'][task].append(tr['spearman'])
            for k_frac in k_fractions:
                recall_key = budget_key('recall', k_frac)
                ndcg_key = budget_key('ndcg', k_frac)
                overall_raw['recall'][task][recall_key].append(tr[recall_key])
                param_results[param]['recall'][task][recall_key].append(tr[recall_key])
                overall_raw['ndcg'][task][ndcg_key].append(tr[ndcg_key])
                param_results[param]['ndcg'][task][ndcg_key].append(tr[ndcg_key])
            if tr['dir_acc'] is not None:
                overall_raw['dir_acc'][task].append(tr['dir_acc'])
                param_results[param]['dir_acc'][task].append(tr['dir_acc'])
            overall_raw['mean_delta'][task].append(tr['mean_delta'])
            param_results[param]['mean_delta'][task].append(tr['mean_delta'])

    return overall_raw, dict(param_results), n_evaluated, evaluated_pairs


def aggregate_results(raw, k_fractions=DEFAULT_K_FRACTIONS):
    def _mean_over(tasks, key):
        vals = [agg[t][key] for t in tasks if t in agg]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    agg = {}
    for task in TASK_KEYS:
        agg[task] = {}
        agg[task]['spearman'] = round(float(np.mean(raw['spearman'][task])), 4) if raw['spearman'][task] else 0.0
        for k_frac in k_fractions:
            recall_key = budget_key('recall', k_frac)
            ndcg_key = budget_key('ndcg', k_frac)
            recall_vals = raw['recall'][task][recall_key]
            ndcg_vals = raw['ndcg'][task][ndcg_key]
            agg[task][recall_key] = round(float(np.mean(recall_vals)), 4) if recall_vals else 0.0
            agg[task][ndcg_key] = round(float(np.mean(ndcg_vals)), 4) if ndcg_vals else 0.0
        agg[task]['dir_acc'] = round(float(np.mean(raw['dir_acc'][task])), 4) if raw.get('dir_acc', {}).get(task) else 0.0
        agg[task]['mean_delta'] = round(float(np.mean(raw['mean_delta'][task])), 4) if raw['mean_delta'][task] else 0.0
    agg['mean'] = {}
    agg['mean']['spearman'] = round(np.mean([agg[t]['spearman'] for t in TASK_KEYS]), 4)
    for k_frac in k_fractions:
        recall_key = budget_key('recall', k_frac)
        ndcg_key = budget_key('ndcg', k_frac)
        agg['mean'][recall_key] = round(np.mean([agg[t][recall_key] for t in TASK_KEYS]), 4)
        agg['mean'][ndcg_key] = round(np.mean([agg[t][ndcg_key] for t in TASK_KEYS]), 4)
    dir_acc_vals = [agg[t]['dir_acc'] for t in TASK_KEYS if agg[t]['dir_acc'] > 0]
    agg['mean']['dir_acc'] = round(float(np.mean(dir_acc_vals)), 4) if dir_acc_vals else 0.0
    agg['mean']['mean_delta'] = round(np.mean([agg[t]['mean_delta'] for t in TASK_KEYS]), 4)
    agg['cache_mean'] = {'spearman': _mean_over(CACHE_TASKS, 'spearman')}
    for k_frac in k_fractions:
        recall_key = budget_key('recall', k_frac)
        ndcg_key = budget_key('ndcg', k_frac)
        agg['cache_mean'][recall_key] = _mean_over(CACHE_TASKS, recall_key)
        agg['cache_mean'][ndcg_key] = _mean_over(CACHE_TASKS, ndcg_key)
    agg['cache_mean']['dir_acc'] = _mean_over(CACHE_TASKS, 'dir_acc')
    agg['cache_mean']['mean_delta'] = _mean_over(CACHE_TASKS, 'mean_delta')
    return agg


def summary_budget_keys(k_fractions):
    recall_keys = [budget_key('recall', k) for k in k_fractions]
    ndcg_key = budget_key('ndcg', 0.05) if 0.05 in k_fractions else budget_key('ndcg', k_fractions[-1])
    return recall_keys, ndcg_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default=os.getenv('CHECKPOINT',
                                          str(Path(__file__).resolve().parent.parent / 'checkpoints' / 'best_model.pth')))
    parser.add_argument(
        '--regressor-checkpoint',
        default=DEFAULT_ABS_REGRESSOR_CHECKPOINT,
    )
    parser.add_argument('--output', default='eval_single_param_sweep.json')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--k-fractions', default=None,
                        help='Comma-separated top-K fractions, e.g. 0.01,0.02,0.03,0.05,0.07,0.10')
    add_sweep_data_args(parser)
    add_model_override_args(parser)
    add_regressor_override_mode_arg(parser)
    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k_fractions = parse_k_fractions(args.k_fractions)
    recall_keys, ndcg_summary_key = summary_budget_keys(k_fractions)

    print("Loading dataset with single-param sweep configs...")
    ds, base_id_map, sweeps, sweep_id_list = load_sweep_dataset(
        args.data_path,
        args.sweep_meta_path,
        args.sweep_config_path,
    )
    all_sweep_hw_ids = set(base_id_map.values()) | set(sweep_id_list)
    print(f"Bases: {len(base_id_map)}, Sweeps: {len(sweeps)}, Total hw_ids: {len(all_sweep_hw_ids)}")

    sweep_indices = [i for i, s in enumerate(ds.samples) if s.get('hw_id', 0) in all_sweep_hw_ids]
    print(f"Sweep samples: {len(sweep_indices)} / {len(ds.samples)}")

    model_overrides = collect_model_overrides(args)
    model = load_film_model(
        args.checkpoint, device, ds,
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
            ds,
            model_overrides=regressor_overrides,
            allow_ambient_env=False,
        )

    method_specs = [
        ('model', 'SensiSelect', model, resolve_checkpoint_path(args.checkpoint)),
    ]
    if regressor is not None:
        method_specs.append((
            'regressor',
            'Abs. Regressor',
            regressor,
            resolve_checkpoint_path(args.regressor_checkpoint),
        ))

    method_results = {}
    n_evaluated = None
    reference_pairs = None
    for method_key, label, method_model, ckpt in method_specs:
        print(f"Running inference on sweep samples for {label} ...")
        all_data = collect_predictions_from_indices(
            method_model, ds, sweep_indices, device, args.batch_size
        )
        print(f"Collected {len(all_data)} predictions")

        print(f"\nEvaluating cross-interval ranking on single-param pairs for {label} ...")
        overall_raw, param_raw, method_n_evaluated, method_pairs = evaluate_sweep_pairs(
            all_data, base_id_map, sweeps, sweep_id_list, method_model, device,
            k_fractions=k_fractions,
        )
        if n_evaluated is None:
            n_evaluated = method_n_evaluated
        if reference_pairs is None:
            reference_pairs = set(method_pairs)
        elif set(method_pairs) != reference_pairs:
            missing = sorted(reference_pairs - set(method_pairs))
            extra = sorted(set(method_pairs) - reference_pairs)
            raise ValueError(
                f"{label} evaluated a different sweep-pair set than SensiSelect: "
                f"missing={len(missing)} extra={len(extra)}"
            )
        print(f"Evaluated {method_n_evaluated} total (base, sweep) pairs")

        overall = aggregate_results(overall_raw, k_fractions=k_fractions)
        param_results = {}
        for param, raw in param_raw.items():
            param_results[param] = aggregate_results(raw, k_fractions=k_fractions)
            param_results[param]['n_pairs'] = raw['n_pairs']
        method_results[method_key] = {
            'label': label,
            'checkpoint': ckpt,
            'overall': overall,
            'per_parameter': param_results,
            'n_evaluated': method_n_evaluated,
        }

        print("\n" + "=" * 80)
        print(f"SINGLE-PARAMETER SWEEP: Cross-Interval Ranking ({label})")
        print("=" * 80)
        recall_headers = ''.join(f" {key.replace('recall@', 'R@'):>8}" for key in recall_keys)
        ndcg_header = ndcg_summary_key.replace('ndcg@', 'N@')
        print(f"{'Task':<15} {'Rank-rho':>10}{recall_headers} {ndcg_header:>8} {'DirAcc':>8}")
        print("-" * 80)
        for task in TASK_KEYS:
            t = overall[task]
            recall_values = ''.join(f" {t[key]:>8.4f}" for key in recall_keys)
            print(
                f"{task:<15} {t['spearman']:>10.4f}"
                f"{recall_values} {t[ndcg_summary_key]:>8.4f} {t['dir_acc']:>8.4f}"
            )
        m = overall['mean']
        print("-" * 80)
        mean_recall_values = ''.join(f" {m[key]:>8.4f}" for key in recall_keys)
        print(
            f"{'All mean':<15} {m['spearman']:>10.4f}"
            f"{mean_recall_values} {m[ndcg_summary_key]:>8.4f} {m['dir_acc']:>8.4f}"
        )
        c = overall['cache_mean']
        cache_recall_values = ''.join(f" {c[key]:>8.4f}" for key in recall_keys)
        print(
            f"{'Cache mean':<15} {c['spearman']:>10.4f}"
            f"{cache_recall_values} {c[ndcg_summary_key]:>8.4f} {c['dir_acc']:>8.4f}"
        )

    results = {
        'overall': method_results['model']['overall'],
        'per_parameter': method_results['model']['per_parameter'],
        'n_evaluated': n_evaluated,
        'n_evaluated_by_method': {
            method_key: blob['n_evaluated'] for method_key, blob in method_results.items()
        },
        'checkpoint': args.checkpoint,
        'k_fractions': list(k_fractions),
        'methods': method_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
