#!/usr/bin/env python3
"""Figure 6: unseen-hardware generalization breakdown.

Outputs:
  - fig_ood_breakdown.pdf: cache/non-cache recall@5 breakdown (main text)
  - fig_ood_breakdown_per_metric.pdf: per-metric recall@5 breakdown
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from fig_common import COLORS, FIG_FULL, TASK_DISPLAY, CACHE_TASKS, savefig, style_figure

OOD_PLOT_TASKS = CACHE_TASKS + ('cpi', 'rob_stall', 'branch_mpki')
A4_WIDTH = FIG_FULL[0] / 0.9
OOD_MAIN_FIGSIZE = (A4_WIDTH * 0.5, (A4_WIDTH / 3.0) * 0.75)

STRATUM_LABELS = {
    "close": "Close\n(0-10)",
    "medium": "Medium\n(11-25)",
    "far": "Far\n(26-47)",
}

BASE_METHOD_SPECS = (
    ("model", "SensiSelect", COLORS["ours"]),
    ("simpoint", "SimPoint", COLORS["simpoint"]),
    ("random", "Random", COLORS["random"]),
)


def _strata_in_order(results):
    return [name for name in ("close", "medium", "far") if name in results]


def _valid_strata(results, strata):
    return [name for name in strata if "summary" in results.get(name, {})]


def _has_method(results, strata, method_key):
    saw_any = False
    saw_missing = False
    for stratum in strata:
        if "summary" not in results.get(stratum, {}):
            continue
        summary = results[stratum].get("summary", {}).get("cache_mean", {})
        method_summary = summary.get(method_key)
        if not isinstance(method_summary, dict):
            saw_missing = True
            continue
        saw_any = True
        for metric_key in ("recall@5%",):
            if metric_key not in method_summary:
                raise ValueError(
                    f"Partial schema for method '{method_key}' in stratum '{stratum}': "
                    f"missing {metric_key}"
                )
        metrics = results[stratum].get("metrics", {})
        for task in OOD_PLOT_TASKS:
            method_metrics = metrics.get(task, {}).get(method_key)
            if not isinstance(method_metrics, dict):
                continue  # skip tasks not in this eval (non-cache may be absent)
            if "recall@5%" not in method_metrics:
                continue
    if saw_any and saw_missing:
        raise ValueError(
            f"Partial schema for method '{method_key}': present in some valid strata and missing in others"
        )
    return saw_any


def _method_specs(results, strata):
    specs = [BASE_METHOD_SPECS[0]]
    if _has_method(results, strata, "regressor"):
        specs.append(("regressor", "Abs. Regressor", COLORS["abs_surrogate"]))
    if _has_method(results, strata, "seqpred"):
        specs.append(("seqpred", "TaoPredictor", COLORS["seqpred"]))
    specs.extend(BASE_METHOD_SPECS[1:])
    return tuple(specs)


def _summary_value(results, stratum, method_key, metric_key, group='cache_mean'):
    return (
        results.get(stratum, {})
        .get("summary", {})
        .get(group, {})
        .get(method_key, {})
        .get(metric_key, np.nan)
    )


def _task_value(results, stratum, task_key, method_key, metric_key):
    return (
        results.get(stratum, {})
        .get("metrics", {})
        .get(task_key, {})
        .get(method_key, {})
        .get(metric_key, np.nan)
    )


def _bar_panel(ax, results, strata, metric_key, title, method_specs, group='cache_mean'):
    x = np.arange(len(strata), dtype=np.float64)
    offset_center = (len(method_specs) - 1) / 2.0
    width = 0.18 if len(method_specs) >= 4 else 0.24
    for idx, (method_key, label, color) in enumerate(method_specs):
        vals = [
            _summary_value(results, stratum, method_key, metric_key, group)
            for stratum in strata
        ]
        ax.bar(
            x + (idx - offset_center) * width,
            vals,
            width=width,
            label=label,
            color=color,
            edgecolor=".3",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(title)
    ax.set_xlabel("")


def _per_metric_panel(ax, results, strata, task_key, method_specs):
    x = np.arange(len(strata), dtype=np.float64)
    width = 0.24
    offset_center = (len(method_specs) - 1) / 2.0
    width = 0.18 if len(method_specs) >= 4 else 0.24
    for idx, (method_key, label, color) in enumerate(method_specs):
        vals = [
            _task_value(results, stratum, task_key, method_key, "recall@5%")
            for stratum in strata
        ]
        ax.bar(
            x + (idx - offset_center) * width,
            vals,
            width=width,
            label=label,
            color=color,
            edgecolor=".3",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABELS.get(s, s) for s in strata])
    ax.set_ylim(0, 1.0)
    ax.set_title(TASK_DISPLAY.get(task_key, task_key))
    ax.set_ylabel("Recall@5%")
    ax.set_xlabel("")


def main():
    with open("eval_hw_generalization.json") as f:
        results = json.load(f)

    strata = _strata_in_order(results)
    valid_strata = _valid_strata(results, strata)
    if not valid_strata:
        raise ValueError("No valid close/medium/far strata found in eval_hw_generalization.json")
    method_specs = _method_specs(results, valid_strata)

    fig, axes = plt.subplots(1, 2, figsize=OOD_MAIN_FIGSIZE)
    _bar_panel(axes[0], results, valid_strata, "recall@5%", "Recall@5%", method_specs, 'cache_mean')
    _bar_panel(axes[1], results, valid_strata, "recall@5%", "", method_specs, 'nc_mean')

    axes[0].set_title("(a) Cache Retrieval")
    axes[1].set_title("(b) Non-Cache Retrieval")

    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    legend = axes[1].legend(
        handles,
        labels,
        frameon=True,
        framealpha=0.9,
        edgecolor=".8",
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderpad=0.15,
        labelspacing=0.12,
        handletextpad=0.25,
        columnspacing=0.6,
        borderaxespad=0.08,
        handlelength=0.9,
    )

    style_figure(fig)
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(7)
        text.set_fontweight("normal")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.30, top=0.86, wspace=0.25)
    savefig(fig, "fig_ood_breakdown")

    n_tasks = len(OOD_PLOT_TASKS)
    n_cols = 4 if n_tasks >= 4 else n_tasks
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig_pm, axes_pm = plt.subplots(n_rows, n_cols, figsize=(FIG_FULL[0] * 1.3, FIG_FULL[1] * 0.6 * n_rows))
    axes_flat = axes_pm.flatten() if n_tasks > 1 else [axes_pm]
    for ax, task in zip(axes_flat, OOD_PLOT_TASKS):
        _per_metric_panel(ax, results, valid_strata, task, method_specs)
    # Hide unused axes
    for ax in axes_flat[len(OOD_PLOT_TASKS):]:
        ax.set_visible(False)
    handles, labels = axes_flat[-1].get_legend_handles_labels()
    axes_flat[-1].legend(
        handles,
        labels,
        frameon=False,
        ncol=len(method_specs),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    style_figure(fig_pm)
    fig_pm.tight_layout()
    savefig(fig_pm, "fig_ood_breakdown_per_metric")
    print("Done.")


if __name__ == "__main__":
    main()
