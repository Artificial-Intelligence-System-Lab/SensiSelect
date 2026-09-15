#!/usr/bin/env python3
"""Evaluate mechanism-level ablations under the same single-parameter sweep
protocol used by the main sensitivity-retrieval experiment.

Usage example:
    python eval_film_ablation.py \
      --variant a0 "Base Model: Absolute, Delta, and Dead-Zone Directional Supervision" /path/a0.pth \
      --variant a1 "+ Grouped Endpoint Sensitivity Ranking" /path/a1.pth \
      --variant a2 "+ Cache-Aware Ranking Score Head" /path/a2.pth \
      --variant a3 "+ Locality-Proxy Cache Enhancement" /path/a3.pth \
      --variant a4 "Cross-HW-Pair Global Ranking without Locality Proxy" /path/a4.pth \
      --variant a5 "Full SensiSelect" /path/a5.pth
"""

import argparse
import json
import os
from pathlib import Path
import torch

from eval_film_common import (
    set_seed, load_film_model, collect_predictions_from_indices, CACHE_TASKS, TASK_KEYS,
    add_model_override_args, collect_model_overrides, add_sweep_data_args,
    CHECKPOINT_DIR, resolve_checkpoint_path,
)
from eval_film_single_param import (
    DEFAULT_K_FRACTIONS, parse_k_fractions, load_sweep_dataset,
    evaluate_sweep_pairs, aggregate_results, budget_key,
)

DEFAULT_VARIANTS = [
    (
        'a0',
        'Base Model: Absolute, Delta, and Dead-Zone Directional Supervision',
        os.getenv('A0_CKPT', str(CHECKPOINT_DIR / 'abl_a0_base_jepa_best_model.pth')),
    ),
    (
        'a1',
        '+ Grouped Endpoint Sensitivity Ranking',
        os.getenv('A1_CKPT', str(CHECKPOINT_DIR / 'abl_a1_grouped_rank_jepa_best_model.pth')),
    ),
    (
        'a2',
        '+ Cache-Aware Ranking Score Head',
        os.getenv('A2_CKPT', str(CHECKPOINT_DIR / 'abl_a2_c3a_jepa_best_cache_rank.pth')),
    ),
    (
        'a3',
        '+ Locality-Proxy Cache Enhancement',
        os.getenv('A3_CKPT', str(CHECKPOINT_DIR / 'abl_a3_h1a_jepa_best_cache_rank.pth')),
    ),
    (
        'a4',
        'Cross-HW-Pair Global Ranking without Locality Proxy',
        os.getenv('A4_CKPT', str(CHECKPOINT_DIR / 'abl_a4_noh1a_chain_v2_sq_stall_best_model.pth')),
    ),
    (
        'a5',
        'Full SensiSelect',
        os.getenv('A5_CKPT', str(CHECKPOINT_DIR / 'best_model.pth')),
    ),
]
DEFAULT_ORDER = [key for key, _, _ in DEFAULT_VARIANTS]


def parse_variant_specs(raw_specs):
    if not raw_specs:
        return [
            {'key': key, 'name': name, 'checkpoint': ckpt}
            for key, name, ckpt in DEFAULT_VARIANTS if ckpt
        ]
    variants = []
    for spec in raw_specs:
        key, name, checkpoint = spec
        variants.append({'key': key, 'name': name, 'checkpoint': checkpoint})
    return variants


def parse_variant_configs(raw_specs):
    configs = {}
    if not raw_specs:
        return configs
    for key, payload in raw_specs:
        configs[key] = json.loads(payload)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='eval_ablation_results.json')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--k-fractions', default=None,
                        help='Comma-separated top-K fractions; defaults to the main retrieval protocol.')
    add_sweep_data_args(parser)
    parser.add_argument(
        '--variant',
        action='append',
        nargs=3,
        metavar=('KEY', 'LABEL', 'CHECKPOINT'),
        help='Custom ablation variant triplet: KEY LABEL CHECKPOINT',
    )
    parser.add_argument(
        '--variant-config',
        action='append',
        nargs=2,
        metavar=('KEY', 'JSON'),
        help='Per-variant model override JSON, e.g. --variant-config h3d \'{\"h3a_mode\":\"residual_only\",\"h3a_eta\":0.01}\'',
    )
    add_model_override_args(parser)
    args = parser.parse_args()

    variants = parse_variant_specs(args.variant)
    variant_configs = parse_variant_configs(args.variant_config)
    global_overrides = collect_model_overrides(args)
    if not variants:
        raise ValueError(
            'No variants configured. Provide --variant ... or set *_CKPT env vars '
            '(A0_CKPT, A1_CKPT, A2_CKPT, A3_CKPT, A4_CKPT, A5_CKPT).'
        )

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k_fractions = parse_k_fractions(args.k_fractions)
    if 0.05 not in k_fractions:
        raise ValueError("Ablation reporting requires k=5% to match the main retrieval protocol.")

    print('Loading single-parameter sweep dataset ...')
    dataset, base_id_map, sweeps, sweep_id_list = load_sweep_dataset(
        args.data_path,
        args.sweep_meta_path,
        args.sweep_config_path,
    )
    print(f'  total samples: {len(dataset)}')
    print(f'  sweep pairs: {len(sweeps)}')
    all_sweep_hw_ids = set(base_id_map.values()) | set(sweep_id_list)
    sweep_indices = [i for i, s in enumerate(dataset.samples) if s.get('hw_id', 0) in all_sweep_hw_ids]
    print(f'  sweep samples: {len(sweep_indices)}')

    all_results = {}
    variant_order = []
    variant_labels = {}

    for variant in variants:
        ckpt = variant['checkpoint']
        try:
            ckpt = resolve_checkpoint_path(ckpt)
        except FileNotFoundError:
            print(f"[SKIP] {variant['name']}: {ckpt} not found")
            continue

        print(f"\nEvaluating: {variant['name']} ...")
        model_overrides = dict(global_overrides)
        model_overrides.update(variant_configs.get(variant['key'], {}))
        model = load_film_model(ckpt, device, dataset, model_overrides=model_overrides)
        all_data = collect_predictions_from_indices(
            model, dataset, sweep_indices, device, args.batch_size)
        overall_raw, per_parameter_raw, n_evaluated, evaluated_pairs = evaluate_sweep_pairs(
            all_data, base_id_map, sweeps, sweep_id_list, model, device, k_fractions=k_fractions)
        overall = aggregate_results(overall_raw, k_fractions=k_fractions)
        per_parameter = {}
        for param, payload in per_parameter_raw.items():
            raw = {
                'spearman': payload['spearman'],
                'recall': payload['recall'],
                'ndcg': payload['ndcg'],
                'dir_acc': payload['dir_acc'],
                'mean_delta': payload['mean_delta'],
            }
            per_parameter[param] = aggregate_results(raw, k_fractions=k_fractions)
            per_parameter[param]['n_pairs'] = payload['n_pairs']

        noncache_tasks = [t for t in TASK_KEYS if t not in set(CACHE_TASKS)]
        summary = {
            'cache_min_sens': round(min(overall[t]['spearman'] for t in CACHE_TASKS), 4),
            'cache_mean_sens': overall['cache_mean']['spearman'],
            'cache_mean_recall@5%': overall['cache_mean'][budget_key('recall', 0.05)],
            'cache_mean_ndcg@5%': overall['cache_mean'][budget_key('ndcg', 0.05)],
            'noncache_mean_sens': round(float(
                sum(overall[t]['spearman'] for t in noncache_tasks) / len(noncache_tasks)
            ), 4),
            'all_mean_sens': overall['mean']['spearman'],
            'all_mean_recall@5%': overall['mean'][budget_key('recall', 0.05)],
            'all_mean_ndcg@5%': overall['mean'][budget_key('ndcg', 0.05)],
        }

        all_results[variant['key']] = {
            'name': variant['name'],
            'checkpoint': ckpt,
            'protocol': 'single_param_sweep',
            'k_fractions': list(k_fractions),
            'n_evaluated': n_evaluated,
            'n_evaluated_pairs': len(evaluated_pairs),
            'overall': overall,
            'per_parameter': per_parameter,
            'summary': summary,
        }
        variant_order.append(variant['key'])
        variant_labels[variant['key']] = variant['name']

        print(f"  evaluated sweep pairs: {n_evaluated}")
        print(f"  Cache min sens: {summary['cache_min_sens']:.4f}, "
              f"Cache mean sens: {summary['cache_mean_sens']:.4f}, "
              f"Cache R@5%: {summary['cache_mean_recall@5%']:.4f}, "
              f"Cache NDCG@5%: {summary['cache_mean_ndcg@5%']:.4f}")

    print('\n' + '=' * 100)
    print('ABLATION RESULTS (FINAL-MODEL MECHANISMS)')
    print('=' * 100)
    print(f"{'Variant':<40} {'CacheMin':>9} {'CacheMean':>10} {'CacheR@5':>10} {'CacheN@5':>10}")
    print('-' * 100)
    for key in [k for k in DEFAULT_ORDER if k in all_results] + [k for k in variant_order if k not in DEFAULT_ORDER]:
        result = all_results[key]
        print(
            f"{result['name']:<40} "
            f"{result['summary']['cache_min_sens']:>8.4f} "
            f"{result['summary']['cache_mean_sens']:>10.4f} "
            f"{result['summary']['cache_mean_recall@5%']:>10.4f} "
            f"{result['summary']['cache_mean_ndcg@5%']:>10.4f}"
        )

    payload = {
        'protocol': 'single_param_sweep',
        'k_fractions': list(k_fractions),
        'variant_order': variant_order,
        'variant_labels': variant_labels,
        'results': all_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
