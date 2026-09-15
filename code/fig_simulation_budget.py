#!/usr/bin/env python3
"""Figure: Minimal Simulation Budget for Bottleneck-Risk Discrimination.

Figure A — Pairwise bottleneck-ordering accuracy vs K, stratified by difficulty
Figure B — Multi-candidate bottleneck-risk discrimination vs shared K:
           top-1 worst-candidate accuracy, pairwise ordering accuracy,
           top-m set accuracy, regret
"""

import json

import matplotlib.pyplot as plt

from fig_common import FIG_FULL, COLORS, savefig, style_figure, style_text

METRIC_DISPLAY = {
    "l1_mpki": "L1D MPKI",
    "l2_mpki": "L2 MPKI",
    "l2_miss_rate": "L2 MR",
    "cpi": "CPI",
    "rob_stall": "ROB Stall",
    "branch_mpki": "Br. MPKI",
}
PLOT_METRICS = ["l1_mpki", "l2_mpki", "l2_miss_rate", "cpi", "rob_stall", "branch_mpki"]
PAIRWISE_FIGSIZE = (FIG_FULL[0], FIG_FULL[0] / 3.0)
LINE_WIDTH = 1.15
MARKER_SIZE = 3.5

BASE_METHOD_STYLE = [
    ("model", "SensiSelect", COLORS["ours"], "o-"),
    ("simpoint", "SimPoint", COLORS["simpoint"], "s--"),
    ("random", "Random", COLORS["random"], "D:"),
]


def _has_pairwise_method(data, method_key):
    pairwise = data.get("pairwise", {}).get("overall", {})
    k_values = data.get("K_values", [])
    saw_any = False
    for metric in PLOT_METRICS:
        metric_data = _kv(pairwise, metric)
        if not metric_data:
            return False
        present = [method_key in _kv(metric_data, k) for k in k_values]
        if any(present):
            saw_any = True
        if any(present) and not all(present):
            raise ValueError(
                f"Partial pairwise schema for method '{method_key}' on metric '{metric}'"
            )
    return saw_any


def _has_multi_method(data, method_key):
    multi = data.get("multi_hw", {})
    k_values = data.get("K_values", [])
    saw_any = False
    for metric in PLOT_METRICS:
        metric_data = _kv(multi, metric)
        if not metric_data:
            return False
        present = [f"top1_{method_key}" in _kv(metric_data, k) for k in k_values]
        if any(present):
            saw_any = True
        if any(present) and not all(present):
            raise ValueError(
                f"Partial multi-hw schema for method '{method_key}' on metric '{metric}'"
            )
    return saw_any


def _pairwise_method_style(data):
    style = [BASE_METHOD_STYLE[0]]
    if _has_pairwise_method(data, "regressor"):
        style.append(("regressor", "Abs. Regressor", COLORS["abs_surrogate"], "^-"))
    if _has_pairwise_method(data, "seqpred"):
        style.append(("seqpred", "TaoPredictor", COLORS["seqpred"], "v-"))
    style.extend(BASE_METHOD_STYLE[1:])
    return style


def _multi_method_style(data):
    style = [BASE_METHOD_STYLE[0]]
    if _has_multi_method(data, "regressor"):
        style.append(("regressor", "Abs. Regressor", COLORS["abs_surrogate"], "^-"))
    if _has_multi_method(data, "seqpred"):
        style.append(("seqpred", "TaoPredictor", COLORS["seqpred"], "v-"))
    style.extend(BASE_METHOD_STYLE[1:])
    return style


def _kv(d, k):
    if k in d:
        return d[k]
    sk = str(k)
    if sk in d:
        return d[sk]
    return {}


def main():
    with open("eval_simulation_budget.json") as f:
        data = json.load(f)

    k_values = data["K_values"]
    pairwise = data["pairwise"]
    multi = data["multi_hw"]
    pairwise_method_style = _pairwise_method_style(data)
    multi_method_style = _multi_method_style(data)
    n_candidates = multi.get("candidates_per_group", max(1, multi.get("hw_per_group", 8) - 1))
    top_m = multi.get("top_m", 3)
    pairwise_k_values = [k for k in k_values if k <= 30]
    pairwise_xticks = pairwise_k_values + ([30] if 30 not in pairwise_k_values else [])

    # ============================================================
    # Figure A: Pairwise bottleneck-ordering accuracy vs K
    # ============================================================
    strata = ["overall", "hard"]
    stratum_titles = [
        "Overall",
        "Hard (small risk gap)",
    ]
    n_metrics = len(PLOT_METRICS)

    fig, axes = plt.subplots(
        len(strata),
        n_metrics,
        figsize=PAIRWISE_FIGSIZE,
    )

    for row, (stratum, s_title) in enumerate(zip(strata, stratum_titles)):
        for col, metric in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            metric_data = _kv(pairwise.get(stratum, {}), metric)
            for key, label, color, fmt in pairwise_method_style:
                vals = [_kv(metric_data, k).get(key, 0) for k in pairwise_k_values]
                ax.plot(
                    pairwise_k_values,
                    vals,
                    fmt,
                    label=label,
                    color=color,
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH,
                )
            if row == 0:
                ax.set_title(METRIC_DISPLAY.get(metric, metric), pad=1.5)
            ax.set_xscale("log")
            ax.set_xlim(1, 30)
            ax.set_xticks(pairwise_xticks)
            ax.set_xticklabels([] if row < len(strata) - 1 else pairwise_xticks)
            if row == len(strata) - 1:
                ticklabels = ax.get_xticklabels()
                if len(ticklabels) >= 2:
                    ticklabels[-2].set_ha("right")
                    ticklabels[-1].set_ha("left")
            ax.set_ylim(0.4, 1.02)
            ax.axhline(y=0.5, color=".5", linewidth=0.5, linestyle="--", zorder=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.96),
        frameon=True,
        framealpha=0.9,
        edgecolor=".8",
        fontsize=7,
        prop={"family": "Times New Roman", "size": 7},
        borderpad=0.2,
        labelspacing=0.2,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.1,
        handlelength=1.0,
    )
    bottom_label = fig.text(0.5, 0.055, "Number of Simulated Intervals", ha="center")
    y_label = fig.text(
        0.02,
        0.5,
        "Pairwise Risk Accuracy",
        rotation=90,
        va="center",
        ha="center",
    )
    row_labels = [
        fig.text(0.045, 0.66, "Overall", rotation=90, va="center", ha="center"),
        fig.text(0.045, 0.28, "Hard (small risk gap)", rotation=90, va="center", ha="center"),
    ]
    title = fig.suptitle(
        "Pairwise Bottleneck-Risk Ordering Accuracy vs Simulation Budget",
        y=1.01,
    )
    style_text(bottom_label)
    style_text(y_label)
    for text in row_labels:
        style_text(text)
    style_text(title)
    style_figure(fig)
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")
        text.set_fontsize(7)
        text.set_fontweight("normal")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.14, top=0.82, wspace=0.20, hspace=0.18)
    savefig(fig, "fig_simulation_budget_pairwise")

    # ============================================================
    # Figure B: Multi-candidate bottleneck-risk discrimination vs K
    # ============================================================
    fig, axes = plt.subplots(
        4,
        n_metrics,
        figsize=(FIG_FULL[0] * 2.1, FIG_FULL[1] * 1.75),
    )

    for col, metric in enumerate(PLOT_METRICS):
        metric_data = _kv(multi, metric)

        # Row 0: top-1 worst-candidate accuracy
        ax = axes[0, col]
        for key, label, color, fmt in multi_method_style:
            vals = [_kv(metric_data, k).get(f"top1_{key}", 0) for k in k_values]
            ax.plot(
                k_values,
                vals,
                fmt,
                label=label,
                color=color,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )
        if col == 0:
            ax.set_ylabel("Top-1 Worst Acc.")
        ax.set_title(METRIC_DISPLAY.get(metric, metric))
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels([])
        ax.axhline(y=1.0 / max(1, n_candidates), color=".5", linewidth=0.5, linestyle="--", zorder=0)

        # Row 1: pairwise ordering accuracy
        ax = axes[1, col]
        for key, label, color, fmt in multi_method_style:
            vals = [_kv(metric_data, k).get(f"pairwise_acc_{key}", 0) for k in k_values]
            ax.plot(
                k_values,
                vals,
                fmt,
                label=label,
                color=color,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )
        if col == 0:
            ax.set_ylabel("Pairwise Risk Acc.")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels([])
        ax.axhline(y=0.5, color=".5", linewidth=0.5, linestyle="--", zorder=0)

        # Row 2: top-m set accuracy
        ax = axes[2, col]
        for key, label, color, fmt in multi_method_style:
            vals = [_kv(metric_data, k).get(f"topm_acc_{key}", 0) for k in k_values]
            ax.plot(
                k_values,
                vals,
                fmt,
                label=label,
                color=color,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )
        if col == 0:
            ax.set_ylabel(f"Top-{top_m} Set Acc.")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels([])
        ax.axhline(y=min(1.0, top_m / max(1, n_candidates)), color=".5", linewidth=0.5, linestyle="--", zorder=0)

        # Row 3: regret
        ax = axes[3, col]
        for key, label, color, fmt in multi_method_style:
            vals = [_kv(metric_data, k).get(f"regret_{key}", 0) for k in k_values]
            ax.plot(
                k_values,
                vals,
                fmt,
                label=label,
                color=color,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )
        if col == 0:
            ax.set_ylabel("Regret")
        ax.set_xlabel("K (shared intervals)")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels([1, 2, 5, 10, 20, 50])

    axes[0, 0].legend(frameon=True, framealpha=0.9, edgecolor=".8", fontsize=7)
    title = fig.suptitle(
        "Multi-Candidate Bottleneck-Risk Discrimination vs Shared Simulation Budget",
        y=1.01,
    )
    style_text(title)
    style_figure(fig)
    fig.tight_layout()
    savefig(fig, "fig_simulation_budget_multi_hw")

    print("Done.")


if __name__ == "__main__":
    main()
