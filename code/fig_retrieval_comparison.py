#!/usr/bin/env python3
"""Figures: sensitivity retrieval on single-parameter hardware sweeps."""

import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

from fig_common import FIG_HALF, COLORS, TASK_DISPLAY, CACHE_TASKS, savefig, style_figure

NC_PLOT_TASKS = ('cpi', 'rob_stall', 'branch_mpki')
ALL_PLOT_TASKS = CACHE_TASKS + NC_PLOT_TASKS
RETRIEVAL_FIGSIZE = (FIG_HALF[0], 2.75 * 2.0 / 3.0)
RANDOM_ALPHA = 0.55
CURVE_LINE_WIDTH = 1.05
CURVE_MARKER_SIZE = 3.3


def load_results():
    with open('eval_single_param_sweep.json') as f:
        ours = json.load(f)
    with open('eval_single_param_baselines.json') as f:
        baselines = json.load(f)
    return ours, baselines


def _method_entries(ours, baselines):
    methods = []
    model_blob = ours.get('methods', {}).get('model', ours)
    methods.append(('Ours', model_blob['overall']))
    reg_blob = ours.get('methods', {}).get('regressor')
    if reg_blob is not None and isinstance(reg_blob.get('overall'), dict):
        methods.append(('Abs. Regressor', reg_blob['overall']))
    seq_blob = ours.get('methods', {}).get('seqpred')
    if seq_blob is not None and isinstance(seq_blob.get('overall'), dict):
        methods.append(('TaoPredictor', seq_blob['overall']))
    methods.append(('SimPoint', baselines['simpoint']))
    methods.append(('Random', baselines['random']))
    return methods


def _infer_budget_levels(blob):
    k_fractions = blob.get('k_fractions')
    if k_fractions:
        return tuple(f"{int(round(float(frac) * 100))}%" for frac in k_fractions)

    if 'methods' in blob:
        model_blob = blob.get('methods', {}).get('model', blob)
        cache_mean = model_blob.get('overall', {}).get('cache_mean', {})
    else:
        baseline_blob = blob.get('simpoint') or blob.get('random') or {}
        cache_mean = baseline_blob.get('cache_mean', {})

    budget_levels = []
    for key in cache_mean:
        if key.startswith('recall@'):
            budget_levels.append(key.split('@', 1)[1])
    if not budget_levels:
        raise ValueError('Could not infer retrieval budget levels from results JSON.')
    return tuple(sorted(budget_levels, key=lambda x: float(x.rstrip('%'))))


def _resolve_budget_levels(ours, baselines):
    ours_levels = _infer_budget_levels(ours)
    baseline_levels = _infer_budget_levels(baselines)
    if ours_levels != baseline_levels:
        raise ValueError(
            f"Budget levels mismatch between model and baselines: {ours_levels} vs {baseline_levels}"
        )
    return ours_levels


def _primary_budget_level(budget_levels):
    return '5%' if '5%' in budget_levels else budget_levels[-1]


def _method_order(ours):
    order = ['Ours']
    reg_blob = ours.get('methods', {}).get('regressor')
    if reg_blob is not None and isinstance(reg_blob.get('overall'), dict):
        order.append('Abs. Regressor')
    seq_blob = ours.get('methods', {}).get('seqpred')
    if seq_blob is not None and isinstance(seq_blob.get('overall'), dict):
        order.append('TaoPredictor')
    order.extend(['SimPoint', 'Random'])
    return order


def _method_color(method):
    return {
        'Ours': COLORS['ours'],
        'Abs. Regressor': COLORS['abs_surrogate'],
        'TaoPredictor': COLORS['seqpred'],
        'SimPoint': COLORS['simpoint'],
        'Random': COLORS['random'],
    }[method]


def _curve_rows(ours, baselines, summary_key, metric_prefix, budget_levels):
    rows = []
    for method, overall_blob in _method_entries(ours, baselines):
        blob = overall_blob[summary_key]
        for pct in budget_levels:
            rows.append({
                'Method': method,
                'Budget': pct,
                'Value': blob[f'{metric_prefix}@{pct}'],
            })
    df = pd.DataFrame(rows)
    df['Budget'] = pd.Categorical(df['Budget'], categories=list(budget_levels), ordered=True)
    return df


def _per_task_rows(ours, baselines, metric_key):
    rows = []
    for task in ALL_PLOT_TASKS:
        for method, blob in _method_entries(ours, baselines):
            rows.append({
                'Task': TASK_DISPLAY[task],
                'Method': method,
                'Value': blob[task][metric_key],
            })
    return pd.DataFrame(rows)


def _apply_compact_shared_legend(left_ax, right_ax):
    left_legend = left_ax.get_legend()
    if left_legend is not None:
        left_legend.remove()
    legend = right_ax.legend(
        frameon=True,
        framealpha=0.5,
        edgecolor='.8',
        fontsize=7,
        ncol=2,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.05),
        borderpad=0.2,
        labelspacing=0.15,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.1,
        handlelength=1.0,
    )
    return legend


def _soften_random_line(ax):
    for line in ax.lines:
        line.set_linewidth(CURVE_LINE_WIDTH)
        line.set_markersize(CURVE_MARKER_SIZE)
        if line.get_label() == 'Random':
            line.set_alpha(RANDOM_ALPHA)


def _finalize_retrieval_layout(fig, axes):
    axes[0].xaxis.label.set_fontsize(9)
    axes[0].xaxis.label.set_fontweight('bold')
    for label in axes[1].get_xticklabels():
        label.set_fontsize(7)
        label.set_fontweight('normal')
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.23, top=0.84, wspace=0.20)


def plot_recall(ours, baselines):
    fig, axes = plt.subplots(1, 2, figsize=RETRIEVAL_FIGSIZE, gridspec_kw={'width_ratios': [1.0, 1.8]})
    method_order = _method_order(ours)
    budget_levels = _resolve_budget_levels(ours, baselines)
    primary_budget = _primary_budget_level(budget_levels)

    df_curve = _curve_rows(ours, baselines, 'cache_mean', 'recall', budget_levels)
    sns.lineplot(
        data=df_curve,
        x='Budget',
        y='Value',
        hue='Method',
        hue_order=method_order,
        style='Method',
        markers=True,
        dashes=False,
        palette=[_method_color(m) for m in method_order],
        ax=axes[0],
    )
    axes[0].set_title('Cache-Mean Recall vs Budget')
    axes[0].set_xlabel('Simulation Budget')
    axes[0].set_ylabel('Cache-Mean Recall@K')
    axes[0].set_ylim(0.0, max(0.35, df_curve['Value'].max() * 1.15))
    axes[0].axhline(0.05, color='.6', linewidth=0.6, linestyle='--', zorder=0)
    _soften_random_line(axes[0])

    df_task = _per_task_rows(ours, baselines, f'recall@{primary_budget}')
    sns.barplot(
        data=df_task,
        x='Task',
        y='Value',
        hue='Method',
        hue_order=method_order,
        palette=[_method_color(m) for m in method_order],
        edgecolor='.3',
        linewidth=0.5,
        ax=axes[1],
    )
    axes[1].set_title(f'Per-Task Recall at {primary_budget} Budget')
    axes[1].set_xlabel('')
    axes[1].set_ylabel(f'Recall@{primary_budget}')
    axes[1].set_ylim(0.0, max(0.60, df_task['Value'].max() * 1.15))
    axes[1].axhline(0.05, color='.6', linewidth=0.6, linestyle='--', zorder=0)
    axes[1].tick_params(axis='x', rotation=20)
    _apply_compact_shared_legend(axes[0], axes[1])

    style_figure(fig)
    _finalize_retrieval_layout(fig, axes)
    savefig(fig, 'fig_retrieval_recall')


def plot_ndcg(ours, baselines):
    fig, axes = plt.subplots(1, 2, figsize=RETRIEVAL_FIGSIZE, gridspec_kw={'width_ratios': [1.0, 1.8]})
    method_order = _method_order(ours)
    budget_levels = _resolve_budget_levels(ours, baselines)
    primary_budget = _primary_budget_level(budget_levels)

    df_curve = _curve_rows(ours, baselines, 'cache_mean', 'ndcg', budget_levels)
    sns.lineplot(
        data=df_curve,
        x='Budget',
        y='Value',
        hue='Method',
        hue_order=method_order,
        style='Method',
        markers=True,
        dashes=False,
        palette=[_method_color(m) for m in method_order],
        ax=axes[0],
    )
    axes[0].set_title('Cache-Mean NDCG vs Budget')
    axes[0].set_xlabel('Simulation Budget')
    axes[0].set_ylabel('Cache-Mean NDCG@K')
    axes[0].set_ylim(0.0, 1.0)
    _soften_random_line(axes[0])

    df_task = _per_task_rows(ours, baselines, f'ndcg@{primary_budget}')
    sns.barplot(
        data=df_task,
        x='Task',
        y='Value',
        hue='Method',
        hue_order=method_order,
        palette=[_method_color(m) for m in method_order],
        edgecolor='.3',
        linewidth=0.5,
        ax=axes[1],
    )
    axes[1].set_title(f'Per-Task Ranking at {primary_budget} Budget')
    axes[1].set_xlabel('')
    axes[1].set_ylabel(f'NDCG@{primary_budget}')
    axes[1].set_ylim(0.0, max(0.70, df_task['Value'].max() * 1.15))
    axes[1].tick_params(axis='x', rotation=20)
    _apply_compact_shared_legend(axes[0], axes[1])

    style_figure(fig)
    _finalize_retrieval_layout(fig, axes)
    savefig(fig, 'fig_retrieval_ndcg')


def main():
    ours, baselines = load_results()
    plot_recall(ours, baselines)
    plot_ndcg(ours, baselines)
    print("Done.")


if __name__ == '__main__':
    main()
