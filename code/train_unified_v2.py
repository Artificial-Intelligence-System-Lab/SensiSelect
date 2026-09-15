"""Train absolute-performance and sensitivity models with configurable data
splits, training objectives, and checkpoint initialization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

from dataset import MicroArchDataset
from model_unified import UnifiedLoopModel, METRIC_KEYS, count_params
from model_unified_film_v2 import FiLMUnifiedModel

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

CODE_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CODE_DIR.parent
DATA_DIR = RELEASE_ROOT / "data"
DEFAULT_DATA_PATH = str(DATA_DIR / "training_dataset_ALL.pkl")
DEFAULT_HW_JSON_PATH = str(DATA_DIR / "gem5_hardware_vectors_0_304.json")


LABEL_IDX = tuple(range(10))  # all 10 metrics
HW_PARAM_NAMES = (
    "clock_frequency",
    "fetchWidth",
    "decodeWidth",
    "renameWidth",
    "dispatchWidth",
    "issueWidth",
    "commitWidth",
    "squashWidth",
    "numROBEntries_log2",
    "numIQEntries_log2",
    "LQEntries_log2",
    "SQEntries_log2",
    "fetchBufferSize",
    "fetchQueueSize",
    "btb_entries_log2",
    "ras_entries",
    "l1d_size_log2",
    "l1i_size_log2",
    "l2_size_log2",
    "l1d_assoc",
    "l1i_assoc",
    "l2_assoc",
    "l1d_mshrs",
    "l1i_mshrs",
    "l2_mshrs",
    "fu_int_alu",
    "fu_int_mult",
    "fu_fp_alu",
    "fu_fp_mult",
    "l1d_latency_log2",
    "l1i_latency_log2",
    "l2_latency_log2",
    "fetch_to_decode_delay_log2",
    "decode_to_rename_delay_log2",
    "rename_to_iew_delay_log2",
    "iew_to_commit_delay_log2",
    "int_alu_latency_log2",
    "int_mult_latency_log2",
    "int_div_latency_log2",
    "fp_alu_latency_log2",
    "fp_mult_latency_log2",
    "fp_div_latency_log2",
    "bp_BiModeBP",
    "bp_TournamentBP",
    "bp_LTAGE",
    "pf_Null",
    "pf_StridePrefetcher",
)

DEFAULT_H6_RAW_FEATURE_NAMES = (
    "fp_half_mean",
    "fp_half_tail_mean",
    "fp_half_max",
    "fp_cache2dual_mean",
    "fp_cache2dual_tail_mean",
    "fp_cache2dual_max",
    "ifp_half_mean",
    "ifp_half_tail_mean",
    "ifp_half_max",
    "ifp_cache2dual_mean",
    "ifp_cache2dual_tail_mean",
    "ifp_cache2dual_max",
)


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


def normalize_quantile(q_value):
    q = float(q_value)
    if q <= 0:
        return 0.0
    if q > 1.0:
        q = q / 100.0
    return min(max(q, 0.0), 1.0)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_spearman(y_true, y_pred):
    if len(y_true) < 3:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else 0.0


def topk_recall(true_sens, pred_sens, k=None):
    n = len(true_sens)
    if n == 0:
        return 0.0
    if k is None:
        k = max(1, int(n * 0.05))
    if n < k:
        k = n
    true_topk = set(np.argsort(true_sens)[-k:])
    pred_topk = set(np.argsort(pred_sens)[-k:])
    return len(true_topk & pred_topk) / k


def ndcg_at_k(true_sens, pred_sens, k=None):
    n = len(true_sens)
    if n == 0:
        return 0.0
    if k is None:
        k = max(1, int(n * 0.05))
    if n < k:
        k = n
    true_order = np.argsort(-true_sens)
    pred_order = np.argsort(-pred_sens)
    rel = true_sens[pred_order[:k]]
    ideal_rel = true_sens[true_order[:k]]
    dcg = np.sum(rel / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(ideal_rel / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def near_zero_false_direction_rate(true_delta, pred_delta, threshold_pct=20):
    if len(true_delta) == 0:
        return 0.0
    threshold = np.percentile(np.abs(true_delta), threshold_pct)
    near_zero_mask = np.abs(true_delta) < threshold
    if near_zero_mask.sum() == 0:
        return 0.0
    wrong = (np.sign(pred_delta[near_zero_mask]) != np.sign(true_delta[near_zero_mask])).sum()
    return float(wrong) / near_zero_mask.sum()


def direction_accuracy(true_delta, pred_delta):
    mask = np.abs(true_delta) > 1e-10
    if mask.sum() == 0:
        return 1.0
    return float((np.sign(pred_delta[mask]) == np.sign(true_delta[mask])).sum()) / mask.sum()


def stack_metric_dict(metric_dict):
    return torch.cat([metric_dict[key] for key in METRIC_KEYS], dim=-1)


def stack_outputs(output, kind):
    return stack_metric_dict(output[kind])


def resolve_loss_metric_keys(loss_metrics_mode):
    mode = loss_metrics_mode.strip().lower()
    if mode == "all":
        return tuple(METRIC_KEYS)
    if mode in ("cache", "cache_only"):
        return tuple(FiLMUnifiedModel.CACHE_METRICS)
    if mode in ("noncache", "all_noncache"):
        return tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)
    # Comma-separated metric names, e.g. "btb_miss_rate,branch_mpki"
    items = tuple(s.strip() for s in mode.split(",") if s.strip())
    invalid = [k for k in items if k not in METRIC_KEYS]
    if invalid:
        raise ValueError(f"Invalid LOSS_METRICS: {','.join(invalid)}. "
                         f"Valid: {','.join(METRIC_KEYS)}, all, cache, noncache")
    return items


def validate_cache_only_m1_recipe(
    *,
    train_mode,
    model_type,
    hw_encoder_type,
    gam_hidden,
    l2_dual_path,
    use_residual,
    c2a_alpha,
    c3a_lambda,
    l2a_rank_lambda,
    h1a_enabled,
    h2a_enabled,
    h3a_enabled,
    a1_enabled,
    cachex_enabled,
    h6a_enabled,
    n1a_enabled,
    nc3_enabled,
    j1a_enabled,
    r1_enabled,
):
    issues = []
    if train_mode != "dse":
        issues.append("TRAIN_MODE must be dse")
    if model_type != "film":
        issues.append("MODEL_TYPE must be film")
    if hw_encoder_type != "gam":
        issues.append("HW_ENCODER must be gam")
    if gam_hidden != 16:
        issues.append("GAM_HIDDEN must be 16")
    if l2_dual_path:
        issues.append("L2_DUAL_PATH must be 0")
    if use_residual:
        issues.append("USE_RESIDUAL must be 0")
    if c2a_alpha != 0.0:
        issues.append("C2A_ALPHA must be 0")
    if c3a_lambda <= 0.0:
        issues.append("C3A_LAMBDA must be > 0")
    if l2a_rank_lambda <= 0.0:
        issues.append("L2A_RANK_LAMBDA must be > 0")
    if not h1a_enabled:
        issues.append("H1A_ENABLED must be 1")

    disabled_flags = []
    if h2a_enabled:
        disabled_flags.append("H2A_ENABLED=0")
    if h3a_enabled:
        disabled_flags.append("H3A_ENABLED=0")
    if a1_enabled:
        disabled_flags.append("A1_ENABLED=0")
    if cachex_enabled:
        disabled_flags.append("CACHEX_ENABLED=0")
    if h6a_enabled:
        disabled_flags.append("H6A_ENABLED=0")
    if n1a_enabled:
        disabled_flags.append("N1A_ENABLED=0")
    if nc3_enabled:
        disabled_flags.append("NC3_ENABLED=0")
    if j1a_enabled:
        disabled_flags.append("J1A_ENABLED=0")
    if r1_enabled:
        disabled_flags.append("R1_ENABLED=0")
    if disabled_flags:
        issues.append("must disable auxiliary branches: " + ", ".join(disabled_flags))

    if issues:
        raise ValueError(
            "LOSS_METRICS=cache enables strict M1 cache-only mode and requires "
            "an h1a_c3-compatible recipe: " + "; ".join(issues)
        )


def _init_a1_top_contrib_sums(a1_knob_meta):
    if not a1_knob_meta:
        return {}
    return {
        name: np.zeros(len(knobs), dtype=np.float64)
        for name, knobs in a1_knob_meta.items()
    }


def _accumulate_a1_stats(output, cache_metric_keys, top_contrib_sums, mode="pair"):
    """Accumulate per-step A1 diagnostic statistics. Returns (baseline, add, ratio, coeff, basis) or None."""
    baseline_abs = []
    add_abs = []
    add_ratio = []
    coeff_abs = []
    basis_abs = []
    for name in cache_metric_keys:
        baseline = output["_a1_baseline"].get(name)
        coeff = output["_a1_coeff"].get(name)
        if mode == "pair":
            add_a = output["_a1_add_a"].get(name)
            add_b = output["_a1_add_b"].get(name)
            basis_a = output["_a1_basis_a"].get(name)
            basis_b = output["_a1_basis_b"].get(name)
            contrib_a = output["_a1_contrib_a"].get(name)
            contrib_b = output["_a1_contrib_b"].get(name)
            if any(v is None for v in (baseline, coeff, add_a, add_b, basis_a, basis_b, contrib_a, contrib_b)):
                continue
            pred_a = output["absolute_a"][name]
            pred_b = output["absolute_b"][name]
            baseline_abs.append(baseline.abs().mean())
            add_abs.append(0.5 * (add_a.abs().mean() + add_b.abs().mean()))
            add_ratio.append(
                0.5 * (
                    (add_a.abs() / (pred_a.abs() + 1e-8)).mean() +
                    (add_b.abs() / (pred_b.abs() + 1e-8)).mean()
                )
            )
            coeff_abs.append(coeff.abs().mean())
            basis_abs.append(0.5 * (basis_a.abs().mean() + basis_b.abs().mean()))
            if name in top_contrib_sums:
                contrib_mean = 0.5 * (
                    contrib_a.abs().mean(dim=0) + contrib_b.abs().mean(dim=0)
                )
                top_contrib_sums[name] += contrib_mean.detach().cpu().numpy().astype(np.float64)
        else:
            add = output["_a1_add"].get(name)
            basis = output["_a1_basis"].get(name)
            contrib = output["_a1_contrib"].get(name)
            if any(v is None for v in (baseline, coeff, add, basis, contrib)):
                continue
            pred = output["absolute"][name]
            baseline_abs.append(baseline.abs().mean())
            add_abs.append(add.abs().mean())
            add_ratio.append((add.abs() / (pred.abs() + 1e-8)).mean())
            coeff_abs.append(coeff.abs().mean())
            basis_abs.append(basis.abs().mean())
            if name in top_contrib_sums:
                contrib_mean = contrib.abs().mean(dim=0)
                top_contrib_sums[name] += contrib_mean.detach().cpu().numpy().astype(np.float64)

    if not baseline_abs:
        return None
    return (
        float(torch.stack(baseline_abs).mean().detach()),
        float(torch.stack(add_abs).mean().detach()),
        float(torch.stack(add_ratio).mean().detach()),
        float(torch.stack(coeff_abs).mean().detach()),
        float(torch.stack(basis_abs).mean().detach()),
    )


def _parse_h6a_targets(spec):
    spec = (spec or "all_noncache").strip().lower()
    all_noncache = tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)
    aliases = {
        "all_noncache": all_noncache,
        "all": all_noncache,
        "stall_only": ("rob_stall", "lq_stall", "sq_stall"),
        "cpi_stall": ("cpi", "rob_stall", "lq_stall", "sq_stall"),
    }
    if spec in aliases:
        return aliases[spec]
    items = tuple(s.strip() for s in spec.split(",") if s.strip())
    invalid = [k for k in items if k not in all_noncache]
    if invalid:
        raise ValueError(
            "H6A_TARGETS must reference non-cache metrics only; got "
            + ", ".join(invalid)
        )
    return items


def _parse_cachex_targets(spec):
    spec = (spec or "cpi_stall").strip().lower()
    all_noncache = tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)
    aliases = {
        "all_noncache": all_noncache,
        "all": all_noncache,
        "stall_only": ("rob_stall", "lq_stall", "sq_stall"),
        "cpi_stall": ("cpi", "rob_stall", "lq_stall", "sq_stall"),
    }
    if spec in aliases:
        return aliases[spec]
    items = tuple(s.strip() for s in spec.split(",") if s.strip())
    invalid = [k for k in items if k not in all_noncache]
    if invalid:
        raise ValueError(
            "CACHEX_TARGETS must reference non-cache metrics only; got "
            + ", ".join(invalid)
        )
    return items


def _parse_nc3_targets(spec):
    spec = (spec or "cpi_stall").strip().lower()
    all_noncache = tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)
    aliases = {
        "all_noncache": all_noncache,
        "all": all_noncache,
        "stall_only": ("rob_stall", "lq_stall", "sq_stall"),
        "cpi_stall": ("cpi", "rob_stall", "lq_stall", "sq_stall"),
    }
    if spec in aliases:
        return aliases[spec]
    items = tuple(s.strip() for s in spec.split(",") if s.strip())
    invalid = [k for k in items if k not in all_noncache]
    if invalid:
        raise ValueError(
            "NC3_TARGETS must reference non-cache metrics only; got "
            + ", ".join(invalid)
        )
    return items


def _parse_j1a_targets(spec):
    spec = (spec or "branch_cpi").strip().lower()
    all_noncache = tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)
    aliases = {
        "branch_cpi": ("branch_mpki", "btb_miss_rate", "cpi"),
        "branch_only": ("branch_mpki", "btb_miss_rate"),
        "all_noncache": all_noncache,
        "all": all_noncache,
    }
    if spec in aliases:
        return aliases[spec]
    items = tuple(s.strip() for s in spec.split(",") if s.strip())
    invalid = [k for k in items if k not in all_noncache]
    if invalid:
        raise ValueError(
            "J1A_TARGETS must reference non-cache metrics only; got "
            + ", ".join(invalid)
        )
    return items


def _shared_sensitivity_target_torch(st):
    """Aggregate per-metric sensitivities into one batch-invariant shared target.

    Labels are already z-score normalized per metric in the dataset, so a
    direct per-sample mean keeps the branch-shared target stable across batch
    size and batch composition.
    """
    if st.ndim != 2:
        raise ValueError("shared sensitivity target expects [N, M] tensor")
    return st.mean(dim=1)


def _shared_sensitivity_target_numpy(st):
    st = np.asarray(st, dtype=np.float64)
    if st.ndim != 2:
        raise ValueError("shared sensitivity target expects [N, M] array")
    return st.mean(axis=1)


def _accumulate_h6_stats(output):
    raw = output.get("_h6_raw")
    feat = output.get("_h6_feat")
    if raw is None or feat is None:
        raw = output.get("_h6a_raw")
        feat = output.get("_h6a_feat")
    if raw is None or feat is None:
        return None
    raw_f = raw.float()
    feat_abs = float(feat.abs().mean().detach())
    raw_mean = raw_f.mean(dim=0).detach().cpu().numpy().astype(np.float64)
    raw_min = float(raw_f.min().detach())
    raw_max = float(raw_f.max().detach())
    sat0 = (
        (raw_f < 1e-6)
        .float()
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    sat1 = (
        (raw_f > 1.0 - 1e-6)
        .float()
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    return feat_abs, raw_mean, raw_min, raw_max, sat0, sat1


def _accumulate_cachex_stats(output):
    feat = output.get("_cachex_feat")
    if feat is None:
        return None
    return float(feat.abs().mean().detach())


def _normalize_h6_mode(spec):
    mode = (spec or "h6a").strip().lower()
    if mode not in ("h6a", "h6c", "h6d"):
        raise ValueError("H6_MODE must be one of: h6a, h6c, h6d")
    return mode


class PairStore:
    """Full dataset tensors on the selected device."""

    def __init__(self, dataset, device):
        print(f"    stacking tensors for {len(dataset.samples)} samples ...", flush=True)
        self.fp = torch.stack([s["fp"] for s in dataset.samples]).to(device)
        self.ifp = torch.stack([s["ifp"] for s in dataset.samples]).to(device)
        self.gwfp = torch.stack([s["gwfp"] for s in dataset.samples]).to(device)
        self.gwifp = torch.stack([s["gwifp"] for s in dataset.samples]).to(device)
        self.hw = torch.stack([s["hw"] for s in dataset.samples]).to(device)
        self.rfp = torch.stack([s["rfp"] for s in dataset.samples]).to(device)
        self.bbe = torch.stack([s["bbe"] for s in dataset.samples]).to(device)
        self.br = torch.stack([s["br"] for s in dataset.samples]).to(device)
        labels = torch.stack([s["labels"] for s in dataset.samples])
        self.labels = labels[:, torch.tensor(LABEL_IDX, dtype=torch.long)].to(device)


def run_absolute_on_indices(model, store, idx, hw_override=None, no_rfp=False):
    hw = store.hw[idx] if hw_override is None else hw_override
    rfp = torch.zeros_like(store.rfp[idx]) if no_rfp else store.rfp[idx]
    output = model.forward_absolute(
        store.fp[idx], store.ifp[idx], hw,
        rfp, store.bbe[idx], store.br[idx],
        store.gwfp[idx], store.gwifp[idx],
    )
    return output, store.labels[idx]


def run_model_on_indices(model, store, h1, h2, no_rfp=False):
    rfp = torch.zeros_like(store.rfp[h1]) if no_rfp else store.rfp[h1]
    output = model(
        store.fp[h1], store.ifp[h1], store.hw[h1], store.hw[h2],
        rfp, store.bbe[h1], store.br[h1],
        store.gwfp[h1], store.gwifp[h1],
    )
    target_a = store.labels[h1]
    target_b = store.labels[h2]
    target_delta = target_b - target_a
    return output, target_delta, target_a, target_b


def parse_jepa_ratio(raw):
    labels = ("close", "medium", "far")
    parts = [float(x.strip()) for x in raw.split(":") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"JEPA_RATIO must have exactly 3 fields, got '{raw}'")
    if any(v < 0 for v in parts):
        raise ValueError(f"JEPA_RATIO entries must be >= 0, got '{raw}'")
    total = sum(parts)
    if total <= 0:
        raise ValueError(f"JEPA_RATIO sum must be > 0, got '{raw}'")
    return {k: v / total for k, v in zip(labels, parts)}


def stratified_jepa_indices(j_l0, n_target, ratios):
    """Sample close/medium/far JEPA indices without exceeding n_target."""
    if n_target <= 0 or j_l0.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=j_l0.device)

    bucket_map = {
        "close": torch.where(j_l0 <= 10)[0],
        "medium": torch.where((j_l0 > 10) & (j_l0 <= 25))[0],
        "far": torch.where(j_l0 > 25)[0],
    }
    labels = ("close", "medium", "far")

    quota_real = {label: float(n_target) * float(ratios[label]) for label in labels}
    quotas = {label: int(math.floor(quota_real[label])) for label in labels}
    remainder = n_target - sum(quotas.values())
    if remainder > 0:
        fractional_order = sorted(
            labels,
            key=lambda label: (quota_real[label] - quotas[label], quota_real[label]),
            reverse=True,
        )
        for label in fractional_order:
            if remainder <= 0:
                break
            quotas[label] += 1
            remainder -= 1

    chosen_parts = []
    leftovers = []
    for label in labels:
        where = bucket_map[label]
        if where.numel() == 0:
            continue
        perm = where[torch.randperm(where.numel(), device=where.device)]
        take = min(quotas[label], perm.numel())
        if take > 0:
            chosen_parts.append(perm[:take])
        if take < perm.numel():
            leftovers.append(perm[take:])

    if chosen_parts:
        chosen = torch.cat(chosen_parts)
    else:
        k = min(n_target, j_l0.numel())
        _, top = torch.topk(j_l0, k)
        return top

    remaining = n_target - chosen.numel()
    if remaining > 0 and leftovers:
        extra_pool = torch.cat(leftovers)
        if extra_pool.numel() > 0:
            extra_take = min(remaining, extra_pool.numel())
            extra_idx = torch.randperm(extra_pool.numel(), device=extra_pool.device)[:extra_take]
            chosen = torch.cat([chosen, extra_pool[extra_idx]])

    if chosen.numel() > n_target:
        trim_idx = torch.randperm(chosen.numel(), device=chosen.device)[:n_target]
        chosen = chosen[trim_idx]
    return chosen


def evaluate_absolute(model, store, idx_eval, batch_size, no_rfp=False):
    pred = {key: [] for key in METRIC_KEYS}
    true = {key: [] for key in METRIC_KEYS}
    with torch.no_grad():
        for start in range(0, idx_eval.numel(), batch_size):
            idx = idx_eval[start:start + batch_size]
            output, target = run_absolute_on_indices(model, store, idx, no_rfp=no_rfp)
            pred_abs = stack_outputs(output, "absolute")
            for i, key in enumerate(METRIC_KEYS):
                pred[key].extend(pred_abs[:, i].cpu().numpy().tolist())
                true[key].extend(target[:, i].cpu().numpy().tolist())
    return {key: safe_spearman(true[key], pred[key]) for key in METRIC_KEYS}


def evaluate_delta(model, store, h1_eval, h2_eval, batch_size, no_rfp=False):
    pred = {key: [] for key in METRIC_KEYS}
    true = {key: [] for key in METRIC_KEYS}
    with torch.no_grad():
        for start in range(0, h1_eval.numel(), batch_size):
            h1 = h1_eval[start:start + batch_size]
            h2 = h2_eval[start:start + batch_size]
            output, target_delta, _, _ = run_model_on_indices(model, store, h1, h2, no_rfp=no_rfp)
            pred_delta = stack_outputs(output, "delta")
            for i, key in enumerate(METRIC_KEYS):
                pred[key].extend(pred_delta[:, i].cpu().numpy().tolist())
                true[key].extend(target_delta[:, i].cpu().numpy().tolist())
    return {key: safe_spearman(true[key], pred[key]) for key in METRIC_KEYS}


def evaluate_sensitivity(model, store, h1_eval, h2_eval, batch_size, no_rfp=False):
    pred = {key: [] for key in METRIC_KEYS}
    true = {key: [] for key in METRIC_KEYS}
    with torch.no_grad():
        for start in range(0, h1_eval.numel(), batch_size):
            h1 = h1_eval[start:start + batch_size]
            h2 = h2_eval[start:start + batch_size]
            output, target_delta, _, _ = run_model_on_indices(model, store, h1, h2, no_rfp=no_rfp)
            pred_sens = stack_outputs(output, "sensitivity")
            true_sens = target_delta.abs()
            for i, key in enumerate(METRIC_KEYS):
                pred[key].extend(pred_sens[:, i].cpu().numpy().tolist())
                true[key].extend(true_sens[:, i].cpu().numpy().tolist())
    return {key: safe_spearman(true[key], pred[key]) for key in METRIC_KEYS}


def evaluate_delta_bundle(
    model,
    store,
    h1_eval,
    h2_eval,
    batch_size,
    no_rfp=False,
    topk_frac=0.05,
    near_zero_pct=20.0,
):
    delta_pred = {key: [] for key in METRIC_KEYS}
    delta_true = {key: [] for key in METRIC_KEYS}
    sens_pred = {key: [] for key in METRIC_KEYS}
    sens_true = {key: [] for key in METRIC_KEYS}
    score_pred = {key: [] for key in METRIC_KEYS}
    with torch.no_grad():
        for start in range(0, h1_eval.numel(), batch_size):
            h1 = h1_eval[start:start + batch_size]
            h2 = h2_eval[start:start + batch_size]
            output, target_delta, _, _ = run_model_on_indices(
                model, store, h1, h2, no_rfp=no_rfp
            )
            pred_delta = stack_outputs(output, "delta")
            pred_sens = stack_outputs(output, "sensitivity")
            true_sens = target_delta.abs()
            for i, key in enumerate(METRIC_KEYS):
                delta_pred[key].extend(pred_delta[:, i].cpu().numpy().tolist())
                delta_true[key].extend(target_delta[:, i].cpu().numpy().tolist())
                sens_pred[key].extend(pred_sens[:, i].cpu().numpy().tolist())
                sens_true[key].extend(true_sens[:, i].cpu().numpy().tolist())
            if "score" in output:
                for key, score_tensor in output["score"].items():
                    score_pred[key].extend(score_tensor.squeeze(-1).cpu().numpy().tolist())
            if "nc_score" in output:
                for key, score_tensor in output["nc_score"].items():
                    score_pred[key].extend(score_tensor.squeeze(-1).cpu().numpy().tolist())

    delta_sp = {key: safe_spearman(delta_true[key], delta_pred[key]) for key in METRIC_KEYS}
    sens_sp = {key: safe_spearman(sens_true[key], sens_pred[key]) for key in METRIC_KEYS}

    topk = {}
    ndcg = {}
    for key in METRIC_KEYS:
        st = np.asarray(sens_true[key], dtype=np.float64)
        if score_pred[key]:
            # C3a score head is evaluated only on ranking metrics.
            sp = np.asarray(score_pred[key], dtype=np.float64)
        else:
            sp = np.asarray(sens_pred[key], dtype=np.float64)
        if len(st) > 10:
            k = max(1, int(len(st) * topk_frac))
            topk[key] = float(topk_recall(st, sp, k=k))
            ndcg[key] = float(ndcg_at_k(st, sp, k=k))
        else:
            topk[key] = 0.0
            ndcg[key] = 0.0

    dir_acc = {}
    near_zero_rate = {}
    for key in METRIC_KEYS:
        dt = np.asarray(delta_true[key], dtype=np.float64)
        dp = np.asarray(delta_pred[key], dtype=np.float64)
        dir_acc[key] = float(direction_accuracy(dt, dp))
        near_zero_rate[key] = float(
            near_zero_false_direction_rate(dt, dp, threshold_pct=near_zero_pct)
        )

    # PDG Spearman: rank on positive degradation max(delta, 0)
    pdg_sp = {}
    for key in METRIC_KEYS:
        dt = np.asarray(delta_true[key], dtype=np.float64)
        dp = np.asarray(delta_pred[key], dtype=np.float64)
        pdg_sp[key] = safe_spearman(np.maximum(dt, 0), np.maximum(dp, 0))

    return {
        "delta_spearman": delta_sp,
        "sensitivity_spearman": sens_sp,
        "pdg_spearman": pdg_sp,
        "topk_recall": topk,
        "ndcg": ndcg,
        "direction_accuracy": dir_acc,
        "near_zero_dir_rate": near_zero_rate,
    }


def evaluate_shared_rank_bundle(
    model,
    store,
    h1_eval,
    h2_eval,
    batch_size,
    target_metric_keys,
    score_key,
    no_rfp=False,
    topk_frac=0.05,
):
    target_indices = torch.tensor(
        [list(METRIC_KEYS).index(k) for k in target_metric_keys],
        dtype=torch.long,
        device=store.labels.device,
    )
    shared_true = []
    shared_pred = []
    with torch.no_grad():
        for start in range(0, h1_eval.numel(), batch_size):
            h1 = h1_eval[start:start + batch_size]
            h2 = h2_eval[start:start + batch_size]
            output, target_delta, _, _ = run_model_on_indices(
                model, store, h1, h2, no_rfp=no_rfp
            )
            st = target_delta.abs()[:, target_indices]
            shared_true.extend(
                _shared_sensitivity_target_torch(st).detach().cpu().numpy().tolist()
            )
            score_tensor = output.get(score_key, None)
            if score_tensor is None:
                shared_pred.extend([0.0] * st.size(0))
            else:
                shared_pred.extend(score_tensor.squeeze(-1).detach().cpu().numpy().tolist())

    shared_true = np.asarray(shared_true, dtype=np.float64)
    shared_pred = np.asarray(shared_pred, dtype=np.float64)
    if shared_true.size > 10:
        k = max(1, int(shared_true.size * topk_frac))
        topk = float(topk_recall(shared_true, shared_pred, k=k))
        ndcg = float(ndcg_at_k(shared_true, shared_pred, k=k))
    else:
        topk = 0.0
        ndcg = 0.0
    return {
        "spearman": float(safe_spearman(shared_true, shared_pred)),
        "topk": topk,
        "ndcg": ndcg,
    }


def estimate_deadzone_tau(train_pair, store, sample_pairs, quantile):
    """Estimate per-metric dead-zone thresholds from the training pair distribution."""
    q = normalize_quantile(quantile)
    if q <= 0 or sample_pairs <= 0:
        return None
    with torch.no_grad():
        h1, h2 = train_pair.sample_batch(sample_pairs)
        abs_delta = (store.labels[h2] - store.labels[h1]).abs()
        return torch.quantile(abs_delta, q, dim=0)


def correlation_ranking_loss(true_sens, pred_sens, eps=1e-8):
    """Listwise correlation loss for one HW-pair group across intervals."""
    losses = []
    for mi in range(true_sens.size(1)):
        s_true = true_sens[:, mi]
        s_pred = pred_sens[:, mi]
        true_centered = s_true - s_true.mean()
        pred_centered = s_pred - s_pred.mean()
        true_var = torch.mean(true_centered.square())
        if true_var <= eps:
            continue
        true_std = torch.sqrt(true_var + eps)
        pred_std = torch.sqrt(torch.mean(pred_centered.square()) + eps)
        corr = torch.mean(true_centered * pred_centered) / (true_std * pred_std + eps)
        losses.append(1.0 - corr)
    if not losses:
        return None
    return torch.stack(losses).mean()


class SimplePairSampler:
    """Simple same-interval random-pair sampler for validation evaluation."""

    def __init__(self, dataset, split_indices, device, seed):
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.split_indices = np.asarray(split_indices, dtype=np.int64)

        group_to_indices = defaultdict(list)
        for idx in self.split_indices.tolist():
            sample = dataset.samples[int(idx)]
            gid = int(sample.get("group_id", -1))
            if gid >= 0:
                group_to_indices[gid].append(int(idx))

        self.group_to_indices = {g: idxs for g, idxs in group_to_indices.items()
                                 if len(idxs) >= 2}

    def make_eval_pairs(self, max_pairs):
        h1_list, h2_list = [], []
        groups = list(self.group_to_indices.keys())
        self.rng.shuffle(groups)
        for g in groups:
            idxs = self.group_to_indices[g]
            if len(idxs) < 2:
                continue
            n_pairs = min(len(idxs) // 2, max_pairs - len(h1_list))
            for _ in range(n_pairs):
                a, b = self.rng.choice(idxs, 2, replace=False)
                h1_list.append(a)
                h2_list.append(b)
            if len(h1_list) >= max_pairs:
                break
        return (
            torch.tensor(h1_list[:max_pairs], dtype=torch.long, device=self.device),
            torch.tensor(h2_list[:max_pairs], dtype=torch.long, device=self.device),
        )


class TrainingPairSampler:
    """Same-interval pair sampler for DSE training."""

    def __init__(self, dataset, train_indices, device, seed, enable_rank=False):
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.enable_rank = enable_rank
        self.hard_scores = []  # populated by compute_hard_scores if hard mining enabled
        group_to_idxs = defaultdict(list)
        group_to_hw_idxs = defaultdict(lambda: defaultdict(list)) if enable_rank else None
        for idx in train_indices.tolist():
            sample = dataset.samples[int(idx)]
            gid = int(sample.get("group_id", -1))
            hid = int(sample.get("hw_id", -1))
            if gid >= 0:
                group_to_idxs[gid].append(int(idx))
                if enable_rank and hid >= 0:
                    group_to_hw_idxs[gid][hid].append(int(idx))
        self.groups = [(g, idxs) for g, idxs in group_to_idxs.items()
                       if len(idxs) >= 2]
        self._gid_hw_to_samples = {}
        self.rank_candidates = []
        if enable_rank:
            pair_to_groups = defaultdict(list)
            for gid, hw_map in group_to_hw_idxs.items():
                hw_ids = sorted(hw_map.keys())
                for hid, samples in hw_map.items():
                    self._gid_hw_to_samples[(gid, hid)] = samples
                for i, hw_a in enumerate(hw_ids):
                    for hw_b in hw_ids[i + 1:]:
                        pair_to_groups[(hw_a, hw_b)].append(gid)
            self.rank_candidates = [
                (hw_a, hw_b, gids)
                for (hw_a, hw_b), gids in pair_to_groups.items()
                if len(gids) >= 2
            ]

    def compute_hard_scores(self, store, target_metrics):
        """Pre-compute group info scores for hard mining.

        For each rank candidate (hw_a, hw_b, gids) and each target metric,
        computes the mean per-metric variance of |delta| across all group
        intervals. Higher score = more informative for ranking.
        """
        if not self.enable_rank or not self.rank_candidates:
            self.hard_scores = []
            return
        self.hard_scores = []  # list of (score, candidate_idx) sorted by score
        metric_indices = [list(METRIC_KEYS).index(m) for m in target_metrics]
        labels = store.labels
        for ci, (hw_a, hw_b, gids) in enumerate(self.rank_candidates):
            if len(gids) < 2:
                continue
            delta_per_gid = []
            for gid in gids:
                sa = self._gid_hw_to_samples[(int(gid), hw_a)][0]
                sb = self._gid_hw_to_samples[(int(gid), hw_b)][0]
                delta_per_gid.append(labels[sb] - labels[sa])
            deltas = torch.stack(delta_per_gid).abs()
            per_metric_var = [float(torch.var(deltas[:, i]).item()) for i in metric_indices]
            self.hard_scores.append((np.mean(per_metric_var), ci))
        self.hard_scores.sort(key=lambda x: -x[0])  # descending by score
        self._hard_top_n = max(1, len(self.hard_scores) // 2)  # top half

    def sample_rank_batch(self, pair_count, k_intervals, hard_ratio=0.0):
        """Sample same-HW-pair examples across multiple workload intervals.

        When hard_ratio > 0, a fraction of pairs are sampled from high-info
        groups (pre-computed via compute_hard_scores). The rest are uniform.
        """
        if not self.enable_rank:
            raise ValueError("TrainingPairSampler rank sampling requested with enable_rank=False")
        if not self.rank_candidates:
            raise ValueError(
                "L2a ranking loss requires at least one HW pair appearing in >=2 intervals"
            )
        all_h1, all_h2 = [], []
        group_sizes = []
        n_hard = int(pair_count * hard_ratio) if hard_ratio > 0 and hasattr(self, 'hard_scores') and self.hard_scores else 0
        for i in range(pair_count):
            if i < n_hard:
                # Weighted sample from top half of hard_scores
                top = self.hard_scores[:self._hard_top_n]
                weights = np.array([s for s, _ in top], dtype=np.float64)
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                else:
                    weights = np.ones(len(top)) / len(top)  # fallback to uniform
                idx_in_top = self.rng.choice(len(top), p=weights)
                _, ci = top[idx_in_top]
                hw_a, hw_b, gids = self.rank_candidates[ci]
            else:
                hw_a, hw_b, gids = self.rank_candidates[
                    self.rng.integers(0, len(self.rank_candidates))
                ]
            n = min(k_intervals, len(gids))
            if n < 2:
                continue
            chosen_gids = self.rng.choice(gids, n, replace=False)
            group_sizes.append(int(n))
            for gid in chosen_gids.tolist():
                sa = self.rng.choice(self._gid_hw_to_samples[(int(gid), hw_a)])
                sb = self.rng.choice(self._gid_hw_to_samples[(int(gid), hw_b)])
                all_h1.append(int(sa))
                all_h2.append(int(sb))
        if not group_sizes:
            raise ValueError(
                "L2a ranking batch sampling failed to find >=2-interval HW pairs"
            )
        return (
            torch.tensor(all_h1, dtype=torch.long, device=self.device),
            torch.tensor(all_h2, dtype=torch.long, device=self.device),
            group_sizes,
        )

    def sample_batch(self, batch_size):
        h1_list, h2_list = [], []
        while len(h1_list) < batch_size:
            g, idxs = self.groups[self.rng.integers(0, len(self.groups))]
            a, b = self.rng.choice(idxs, 2, replace=False)
            h1_list.append(a)
            h2_list.append(b)
        return (torch.tensor(h1_list, dtype=torch.long, device=self.device),
                torch.tensor(h2_list, dtype=torch.long, device=self.device))


# Intervention-aware pair sampling ratios
INTERVENTION_RATIOS = {
    "oneknob-small": 0.40,
    "oneknob-large": 0.20,
    "interaction":   0.20,
    "random":        0.10,
    "hard":          0.10,
}
INTERVENTION_TYPES = list(INTERVENTION_RATIOS.keys())
INTERVENTION_WEIGHTS = list(INTERVENTION_RATIOS.values())


def _extract_hw_id(raw_sample):
    """Extract the hardware ID, or derive a stable hash when no ID is provided."""
    hw = raw_sample.get('hardware', {})
    hid = hw.get('id', -1)
    if hid >= 0:
        return hid
    # Fallback: derive the ID from the hardware parameters.
    from build_pair_suite import NUMERIC_PARAMS, CATEGORICAL_PARAMS
    all_params = NUMERIC_PARAMS + CATEGORICAL_PARAMS
    items = tuple(sorted((k, str(v)) for k, v in hw.items() if k in all_params))
    raw_bytes = repr(items).encode('utf-8')
    return int(hashlib.md5(raw_bytes).hexdigest()[:16], 16) % (2**31)


class InterventionPairSampler:
    """Intervention-stratified pair sampler with cache ranking batches.

    Replaces random same-interval pairs with type-stratified sampling:
      40% one-knob small-step    20% one-knob large-step
      20% physically-related two-knob   10% random multi-knob
      10% hard / high-sensitivity (falls back to random)

    Builds hardware-pair categories from configuration differences, then
    maps each hardware pair to same-interval sample index pairs.
    """

    def __init__(self, dataset, raw_train_samples, train_indices, device, seed, enable_rank=False):
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.enable_rank = enable_rank
        self.hard_scores = []  # populated by compute_hard_scores if hard mining enabled

        # Build same-interval groups from processed samples so sampling semantics
        # do not depend on raw/processed pickle order staying perfectly aligned.
        train_set = {int(i) for i in train_indices.tolist()}
        group_to_samples = defaultdict(list)  # group_id -> [(sample_idx, hw_id)]

        for si in sorted(train_set):
            sample = dataset.samples[si]
            gid = int(sample.get("group_id", -1))
            hid = int(sample.get("hw_id", -1))
            if gid >= 0 and hid >= 0:
                group_to_samples[gid].append((si, hid))

        print(f"  InterventionPairSampler: "
              f"{len(group_to_samples)} unique intervals in training set")

        # Build HW pair types from raw hardware configs keyed by processed hw_id.
        self._build_pair_types(dataset, raw_train_samples, train_set)

        # Build interval index: group_id -> {hw_id: [sample_indices]}
        interval_hw_samples = defaultdict(lambda: defaultdict(list))
        for gid, samples in group_to_samples.items():
            for si, hid in samples:
                interval_hw_samples[gid][hid].append(si)

        # For each pair type, collect (hw_a, hw_b) + shared intervals
        raw_type_pools = {}  # type -> [(hw_a, hw_b, [group_ids])]
        for ptype in INTERVENTION_TYPES:
            hw_pairs = self._get_hw_pairs(ptype)
            pool = []
            for hw_a, hw_b in hw_pairs:
                # Find intervals where both HWs appear
                shared_gids = []
                for gid in interval_hw_samples:
                    if (hw_a in interval_hw_samples[gid] and
                        hw_b in interval_hw_samples[gid]):
                        shared_gids.append(gid)
                if shared_gids:
                    pool.append((hw_a, hw_b, shared_gids))
            raw_type_pools[ptype] = pool

        random_pool = raw_type_pools.get("random", [])
        self.type_pools = {}
        for ptype in INTERVENTION_TYPES:
            pool = raw_type_pools.get(ptype, [])
            if not pool and ptype != "random":
                # Fallback: use random-hw type if a type has no valid pairs.
                pool = list(random_pool)
            self.type_pools[ptype] = pool

        # Index: (gid, hw_id) -> [sample indices] for fast lookup
        self._gid_hw_to_samples = {}
        for gid, hw_map in interval_hw_samples.items():
            for hw_id, samples in hw_map.items():
                self._gid_hw_to_samples[(gid, hw_id)] = samples

        self.rank_candidates = []
        if enable_rank:
            rank_candidate_map = {}
            for pools in raw_type_pools.values():
                for hw_a, hw_b, shared_gids in pools:
                    if len(shared_gids) >= 2:
                        rank_candidate_map[(hw_a, hw_b)] = list(shared_gids)
            self.rank_candidates = [
                (hw_a, hw_b, gids) for (hw_a, hw_b), gids in rank_candidate_map.items()
            ]

        types_with_pairs = [t for t in INTERVENTION_TYPES
                           if len(self.type_pools[t]) > 0]
        self._types = types_with_pairs
        if not self._types:
            raise ValueError(
                "InterventionPairSampler found no valid pair types in the training split"
            )
        self._weights = [INTERVENTION_RATIOS[t] for t in types_with_pairs]
        # Normalize
        wsum = sum(self._weights)
        self._weights = [w / wsum for w in self._weights]

        print(f"  InterventionPairSampler: {len(types_with_pairs)} types, "
              f"total {sum(len(self.type_pools[t]) for t in types_with_pairs)} hw-pairs")

    def _build_pair_types(self, dataset, raw_samples, train_set):
        """Build HW-level pair types for training HW configs only."""
        from build_pair_suite import (
            NUMERIC_PARAMS, CATEGORICAL_PARAMS, ALL_HW_PARAMS,
            PARAM_TO_IDX, SIZE_TO_KB, _categorical_match,
        )
        import numpy as np

        raw_hw_id_to_config = {}
        for s in raw_samples:
            hid = _extract_hw_id(s)
            if hid >= 0 and hid not in raw_hw_id_to_config:
                raw_hw_id_to_config[hid] = dict(s["hardware"])

        train_hw_ids = {
            int(dataset.samples[si].get("hw_id", -1))
            for si in train_set
            if int(dataset.samples[si].get("hw_id", -1)) >= 0
        }

        hw_id_to_config = {}
        missing_hw = []
        for hid in sorted(train_hw_ids):
            cfg = raw_hw_id_to_config.get(hid)
            if cfg is None:
                missing_hw.append(hid)
                continue
            hw_id_to_config[hid] = cfg

        if missing_hw:
            raise ValueError(
                f"Missing raw hardware configs for {len(missing_hw)} train hw_ids; "
                "raw/processed dataset sources are inconsistent"
            )

        hw_ids = sorted(hw_id_to_config.keys())
        n_hw = len(hw_ids)
        if n_hw < 2:
            self.hw_ids = []
            self._train_hw_pairs = {"oneknob-small": [], "oneknob-large": [],
                                     "interaction": [], "random": [], "hard": []}
            return

        def to_f(v):
            return float(SIZE_TO_KB.get(v, 0) if isinstance(v, str) else v) if isinstance(v, str) else float(v)

        mat = np.zeros((n_hw, len(NUMERIC_PARAMS)), dtype=np.float32)
        for i, hid in enumerate(hw_ids):
            cfg = hw_id_to_config[hid]
            for j, p in enumerate(NUMERIC_PARAMS):
                mat[i, j] = to_f(cfg.get(p, 0))

        param_values = {}
        for p in NUMERIC_PARAMS:
            vals = sorted(set(to_f(hw_id_to_config[hid].get(p, 0))
                             for hid in hw_ids), key=float)
            param_values[p] = vals

        self.hw_ids = hw_ids
        self._hw_id_to_config = hw_id_to_config
        self._mat = mat
        self._param_values = param_values
        self._n_params = len(NUMERIC_PARAMS)

        # Build oneknob-small pairs
        small_pairs = []
        for knob in NUMERIC_PARAMS:
            ki = PARAM_TO_IDX[knob]
            vals = param_values[knob]
            for i, hid_i in enumerate(hw_ids):
                cfg_i = hw_id_to_config[hid_i]
                vi = to_f(cfg_i[knob])
                try:
                    vi_r = vals.index(vi)
                except ValueError:
                    continue
                for step in [-1, 1]:
                    nb = vi_r + step
                    if nb < 0 or nb >= len(vals):
                        continue
                    for j in range(i + 1, n_hw):
                        hid_j = hw_ids[j]
                        vj = to_f(hw_id_to_config[hid_j][knob])
                        if vj != vals[nb]:
                            continue
                        ok = all(abs(mat[i, k] - mat[j, k]) < 1e-6
                                for k in range(len(NUMERIC_PARAMS)) if k != ki)
                        if ok and _categorical_match(cfg_i, hw_id_to_config[hid_j]):
                            small_pairs.append((hid_i, hid_j))
        self._train_hw_pairs = {"oneknob-small": small_pairs}

        # Build oneknob-large pairs
        large_pairs = set()
        for knob in NUMERIC_PARAMS:
            ki = PARAM_TO_IDX[knob]
            vals = param_values[knob]
            for i, hid_i in enumerate(hw_ids):
                cfg_i = hw_id_to_config[hid_i]
                vi = to_f(cfg_i[knob])
                try:
                    vi_r = vals.index(vi)
                except ValueError:
                    continue
                for j in range(n_hw):
                    if i == j:
                        continue
                    hid_j = hw_ids[j]
                    cfg_j = hw_id_to_config[hid_j]
                    vj = to_f(cfg_j[knob])
                    try:
                        vj_r = vals.index(vj)
                    except ValueError:
                        continue
                    if abs(vi_r - vj_r) < 2:
                        continue
                    ok = all(abs(mat[i, k] - mat[j, k]) < 1e-6
                            for k in range(len(NUMERIC_PARAMS)) if k != ki)
                    if ok and _categorical_match(cfg_i, cfg_j):
                        a, b = (hid_i, hid_j) if hid_i < hid_j else (hid_j, hid_i)
                        large_pairs.add((a, b))
        self._train_hw_pairs["oneknob-large"] = list(large_pairs)

        # Build interaction pairs (physically related two-knob)
        from build_pair_suite import INTERACTION_PAIRS
        interaction_pairs = set()
        for knob_list, _ in INTERACTION_PAIRS:
            ki_set = {PARAM_TO_IDX[k] for k in knob_list}
            for i, hid_i in enumerate(hw_ids):
                for j in range(i + 1, n_hw):
                    hid_j = hw_ids[j]
                    changed = set()
                    for k in range(len(NUMERIC_PARAMS)):
                        if abs(mat[i, k] - mat[j, k]) > 1e-6:
                            changed.add(k)
                    if changed == ki_set and _categorical_match(
                        hw_id_to_config[hid_i], hw_id_to_config[hid_j]):
                        interaction_pairs.add((hid_i, hid_j))
        self._train_hw_pairs["interaction"] = list(interaction_pairs)

        # Random pairs
        rng = np.random.default_rng(42)
        max_random = min(2000, n_hw * (n_hw - 1) // 2)
        random_set = set()
        while len(random_set) < max_random:
            i, j = int(rng.integers(0, n_hw)), int(rng.integers(0, n_hw))
            if i != j:
                random_set.add((hw_ids[min(i, j)], hw_ids[max(i, j)]))
        self._train_hw_pairs["random"] = sorted(random_set)

        # Initialize the hard-pair category with the random-pair pool.
        self._train_hw_pairs["hard"] = self._train_hw_pairs["random"]

    def _get_hw_pairs(self, ptype):
        return self._train_hw_pairs.get(ptype, self._train_hw_pairs.get("random", []))

    def sample_batch(self, batch_size):
        h1_list, h2_list = [], []
        while len(h1_list) < batch_size:
            # Pick pair type by ratio
            ptype = self.rng.choice(self._types, p=self._weights)
            pool = self.type_pools[ptype]
            # Pick random HW pair
            idx = int(self.rng.integers(0, len(pool)))
            hw_a, hw_b, shared_gids = pool[idx]
            # Pick random shared interval
            gid = shared_gids[int(self.rng.integers(0, len(shared_gids)))]
            # Pick one sample from each HW in this interval
            sa = self.rng.choice(self._gid_hw_to_samples[(gid, hw_a)])
            sb = self.rng.choice(self._gid_hw_to_samples[(gid, hw_b)])
            h1_list.append(sa)
            h2_list.append(sb)
        return (torch.tensor(h1_list, dtype=torch.long, device=self.device),
                torch.tensor(h2_list, dtype=torch.long, device=self.device))

    def compute_hard_scores(self, store, target_metrics):
        """Pre-compute group info scores for hard mining (intervention version).

        Same logic as TrainingPairSampler: per-metric Var(|delta|) averaged
        across all group intervals for each rank candidate. One sample per
        (gid, hw) pair since intervals are unique per HW config.
        """
        if not self.enable_rank or not self.rank_candidates:
            self.hard_scores = []
            return
        self.hard_scores = []
        metric_indices = [list(METRIC_KEYS).index(m) for m in target_metrics]
        labels = store.labels
        for ci, (hw_a, hw_b, gids) in enumerate(self.rank_candidates):
            if len(gids) < 2:
                continue
            delta_per_gid = []
            for gid in gids:
                sa = self._gid_hw_to_samples[(int(gid), hw_a)][0]
                sb = self._gid_hw_to_samples[(int(gid), hw_b)][0]
                delta_per_gid.append(labels[sb] - labels[sa])
            deltas = torch.stack(delta_per_gid).abs()
            per_metric_var = [float(torch.var(deltas[:, i]).item()) for i in metric_indices]
            self.hard_scores.append((np.mean(per_metric_var), ci))
        self.hard_scores.sort(key=lambda x: -x[0])
        self._hard_top_n = max(1, len(self.hard_scores) // 2)

    def sample_rank_batch(self, pair_count, k_intervals, hard_ratio=0.0):
        """Sample same-HW-pair examples across multiple workload intervals."""
        if not self.enable_rank:
            raise ValueError(
                "InterventionPairSampler rank sampling requested with enable_rank=False"
            )
        if not self.rank_candidates:
            raise ValueError(
                "L2a ranking loss requires at least one intervention HW pair "
                "appearing in >=2 intervals"
            )
        all_h1, all_h2 = [], []
        group_sizes = []
        n_hard = int(pair_count * hard_ratio) if hard_ratio > 0 and hasattr(self, 'hard_scores') and self.hard_scores else 0
        for i in range(pair_count):
            if i < n_hard:
                top = self.hard_scores[:self._hard_top_n]
                weights = np.array([s for s, _ in top], dtype=np.float64)
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                else:
                    weights = np.ones(len(top)) / len(top)
                idx_in_top = self.rng.choice(len(top), p=weights)
                _, ci = top[idx_in_top]
                hw_a, hw_b, gids = self.rank_candidates[ci]
            else:
                hw_a, hw_b, gids = self.rank_candidates[
                    self.rng.integers(0, len(self.rank_candidates))
                ]
            n = min(k_intervals, len(gids))
            if n < 2:
                continue
            chosen_gids = self.rng.choice(gids, n, replace=False)
            group_sizes.append(int(n))
            for gid in chosen_gids.tolist():
                sa = self.rng.choice(self._gid_hw_to_samples[(int(gid), hw_a)])
                sb = self.rng.choice(self._gid_hw_to_samples[(int(gid), hw_b)])
                all_h1.append(int(sa))
                all_h2.append(int(sb))
        if not group_sizes:
            raise ValueError(
                "L2a ranking batch sampling failed to find >=2-interval intervention pairs"
            )
        return (
            torch.tensor(all_h1, dtype=torch.long, device=self.device),
            torch.tensor(all_h2, dtype=torch.long, device=self.device),
            group_sizes,
        )


def _save_model(model, path):
    """Save state_dict, stripping _orig_mod. prefix if compiled."""
    state = model.state_dict()
    clean = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    torch.save(clean, path)


def _load_initial_state_dict(model, ckpt_path, device, strict=True):
    """Load an initialization checkpoint before training.

    Supports plain state_dict checkpoints and dict wrappers with a
    ``state_dict`` field. Also strips any ``_orig_mod.`` prefixes left by
    compiled models.
    """
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"INIT_CKPT not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")
    clean = {
        k.replace("_orig_mod.", "", 1) if isinstance(k, str) and k.startswith("_orig_mod.") else k: v
        for k, v in state.items()
    }
    skipped_mismatch = []
    if not strict:
        target_state = model.state_dict()
        filtered = {}
        for key, value in clean.items():
            if key not in target_state:
                continue
            target_value = target_state[key]
            if hasattr(value, "shape") and hasattr(target_value, "shape") and tuple(value.shape) != tuple(target_value.shape):
                skipped_mismatch.append(
                    f"{key}: ckpt{tuple(value.shape)} != model{tuple(target_value.shape)}"
                )
                continue
            filtered[key] = value
        clean = filtered
    incompatible = model.load_state_dict(clean, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(f"  init_ckpt={ckpt_path} strict={int(strict)}")
    if missing:
        print(f"  init missing keys ({len(missing)}): {', '.join(missing[:8])}" +
              (" ..." if len(missing) > 8 else ""))
    if unexpected:
        print(f"  init unexpected keys ({len(unexpected)}): {', '.join(unexpected[:8])}" +
              (" ..." if len(unexpected) > 8 else ""))
    if skipped_mismatch:
        print(f"  init skipped mismatched keys ({len(skipped_mismatch)}): {', '.join(skipped_mismatch[:4])}" +
              (" ..." if len(skipped_mismatch) > 4 else ""))


def _freeze_all_but_j1a(model):
    """Freeze the base model and keep only the J1a score head trainable."""
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        keep = name.startswith("j1a_score_head.")
        param.requires_grad = keep
        if keep:
            trainable.append(name)
        else:
            frozen.append(name)
    if not trainable:
        raise ValueError("J1A_FREEZE_BASE=1 but no j1a_score_head parameters were found")
    print(f"  J1A freeze-base mode: trainable_tensors={len(trainable)} frozen_tensors={len(frozen)}")
    print(f"  J1A trainable params: {', '.join(trainable)}")


def _set_j1a_probe_mode(model):
    """Run frozen base in eval mode and keep only J1a head in train mode."""
    model.eval()
    if not hasattr(model, "j1a_score_head") or model.j1a_score_head is None:
        raise ValueError("J1A probe mode requested but model has no j1a_score_head")
    model.j1a_score_head.train()


def _freeze_except_heads(model, train_heads, caller="freeze"):
    """Freeze all parameters, then unfreeze only heads[train_heads].

    Used for per-head fine-tuning (R1) and positive-degradation ranking (PDG):
    warm-start from a checkpoint,
    freeze everything except specific endpoint heads. Supports model.heads
    and model.cache_heads.
    """
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name in train_heads:
        found = False
        for head_dict in ("heads", "cache_heads", "stall_heads"):
            if hasattr(model, head_dict) and name in getattr(model, head_dict):
                for p in getattr(model, head_dict)[name].parameters():
                    p.requires_grad = True
                trainable.append(f"{head_dict}.{name}")
                found = True
                break
        # R2: also unfreeze rob_adapter (separate from heads)
        if hasattr(model, "rob_adapter") and name == "rob_stall":
            for p in model.rob_adapter.parameters():
                p.requires_grad = True
            trainable.append("rob_adapter")
        if not found:
            available = sorted(
                list(getattr(model, "cache_heads", {}).keys()) +
                list(getattr(model, "heads", {}).keys()) +
                list(getattr(model, "stall_heads", {}).keys()))
            raise ValueError(
                f"{caller}_TRAIN_HEADS includes '{name}' which is not in "
                f"model.heads, model.cache_heads, or model.stall_heads. "
                f"Available: {available}")
    print(f"  {caller} freeze: only heads {trainable} are trainable")


def _compute_hotl_loss(store, h1, h2, pred_abs_a, pred_abs_b, hotl_scale):
    """HOTL auxiliary loss: mr(c) ≈ d fp(l)/dl.

    Estimates miss rate from footprint slope at the cache-size-relevant window.
    NOTE: PairStore gwfp values are z-score normalized; the slope computed here
    is a statistical proxy, not the physical mr(c). Full HOTL supervision would
    require raw footprint curves + original cache sizes.

    Uses 64B-cacheline channel (index 1); applies loss symmetrically to h1 and h2.
    """
    TASK_IDX = {k: i for i, k in enumerate(METRIC_KEYS)}
    # This auxiliary approximation uses only data footprints for l2_miss_rate.
    cache_configs = [
        ("l1_mpki", "gwfp", 16),         # l1d_size
        ("l1i_mpki", "gwifp", 17),       # l1i_size
        ("l2_mpki", "gwfp", 18),         # l2_size
        ("l2_miss_rate", "gwfp", 18),    # l2_size
    ]
    losses = []
    for name, fp_key, cs_idx in cache_configs:
        for h_idx in (h1, h2):
            fp_raw = getattr(store, fp_key)[h_idx][:, 1, :]              # [B, 19] ch1=cacheline
            slope = (fp_raw[:, 2:] - fp_raw[:, :-2]) / 2                  # [B, 17]
            cache_size = store.hw[h_idx][:, cs_idx]                       # [B], z-scored
            win_idx = ((cache_size + 2.5) / 5.0 * 16).long().clamp(0, 16)
            hotl_target = slope.gather(1, win_idx.unsqueeze(1))           # [B, 1]

            ti = TASK_IDX[name]
            pred = pred_abs_a[:, ti:ti + 1] if h_idx is h1 else pred_abs_b[:, ti:ti + 1]
            pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
            targ_norm = (hotl_target - hotl_target.mean()) / (hotl_target.std() + 1e-8)
            losses.append(
                F.smooth_l1_loss(pred_norm, hotl_scale * targ_norm, reduction="mean"))
    return sum(losses) / len(losses)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    seed = env_int("SEED", 42)
    set_seed(seed)

    experiment_tag = os.getenv("EXPERIMENT_TAG", "unified_abs").strip()
    data_path = os.getenv("DATA_PATH", DEFAULT_DATA_PATH)
    epochs = env_int("TRAIN_EPOCHS", 100)
    patience = env_int("TRAIN_PATIENCE", 15)
    batch_size = env_int("BATCH_SIZE", 2048)
    hidden = env_int("HIDDEN", 256)
    dropout = env_float("DROPOUT", 0.1)
    lr = env_float("TRAIN_LR", 2e-4)
    weight_decay = env_float("TRAIN_WD", 1e-4)
    warmup_epochs = env_int("TRAIN_WARMUP_EPOCHS", 3)
    train_max_steps = env_int("TRAIN_MAX_STEPS", 0)
    noise_std = env_float("NOISE_STD", 0.01)
    loop_steps = env_int("LOOP_STEPS", 3)
    num_heads = env_int("NUM_HEADS", 4)
    ff_mult = env_float("FF_MULT", 2.0)
    split_mode = os.getenv("SPLIT_MODE", "random").strip().lower()
    train_mode = os.getenv("TRAIN_MODE", "abs").strip().lower()
    if train_mode not in ("abs", "dse", "dse_direct"):
        raise ValueError("TRAIN_MODE must be one of: abs, dse, dse_direct")
    direct_delta_enabled = train_mode == "dse_direct"
    loss_metrics_mode = os.getenv("LOSS_METRICS", "all").strip().lower()
    no_rfp = os.getenv("NO_RFP", "0").strip() == "1"
    dse_lambda_delta = env_float("DSE_LAMBDA_DELTA", 1.0)
    dse_lambda_dir = env_float("DSE_LAMBDA_DIR", 2.0)
    dse_dir_temp = env_float("DSE_DIR_TEMP", 20.0)
    dse_deadzone_q = env_float("DSE_DEADZONE_Q", 0.0)
    dse_deadzone_calib_pairs = env_int("DSE_DEADZONE_CALIB_PAIRS", 32768)
    l2a_rank_lambda = env_float("L2A_RANK_LAMBDA", 0.0)
    l2a_rank_k = env_int("L2A_RANK_K", 8)
    l2a_rank_pairs = env_int("L2A_RANK_PAIRS", 16)
    c2a_alpha = env_float("C2A_ALPHA", 0.0)
    c3a_lambda = env_float("C3A_LAMBDA", 0.0)
    nc3_enabled = os.getenv("NC3_ENABLED", "0").strip() == "1"
    nc3_lambda = env_float("NC3_LAMBDA", 0.0)
    nc3_detach_base = os.getenv("NC3_DETACH_BASE", "1").strip() != "0"
    j1a_enabled = os.getenv("J1A_ENABLED", "0").strip() == "1"
    j1a_lambda = env_float("J1A_LAMBDA", 0.0)
    j1a_detach_base = os.getenv("J1A_DETACH_BASE", "1").strip() != "0"
    a1_enabled = os.getenv("A1_ENABLED", "0").strip() == "1"
    a1_basis_hidden = env_int("A1_BASIS_HIDDEN", 16)
    a1_knob_mode = os.getenv("A1_KNOB_MODE", "full47").strip().lower()
    h1a_enabled = os.getenv("H1A_ENABLED", "0").strip() == "1"
    h2a_enabled = os.getenv("H2A_ENABLED", "0").strip() == "1"
    h2a_concave = os.getenv("H2A_CONCAVE", "1").strip() != "0"
    h2a_above_mode = os.getenv("H2A_ABOVE_MODE", "boundary").strip().lower()
    cachex_enabled = os.getenv("CACHEX_ENABLED", "0").strip() == "1"
    n1a_enabled = os.getenv("N1A_ENABLED", "0").strip() == "1"
    b1_enabled = os.getenv("B1_ENABLED", "0").strip() == "1"
    s2_enabled = os.getenv("S2_ENABLED", "0").strip() == "1"
    s2_alpha = env_float("S2_ALPHA", 0.10)
    hard_mining_enabled = os.getenv("HARD_MINING_ENABLED", "0").strip() == "1"
    hard_mining_ratio = env_float("HARD_MINING_RATIO", 0.3)
    hard_mining_targets = tuple(
        s.strip() for s in os.getenv("HARD_MINING_TARGETS",
            "cpi,rob_stall,lq_stall,sq_stall,branch_mpki,btb_miss_rate").split(",")
        if s.strip()
    )
    rank_hard_ratio = hard_mining_ratio if hard_mining_enabled else 0.0
    r2_enabled = os.getenv("R2_ENABLED", "0").strip() == "1"
    cachex_targets = _parse_cachex_targets(os.getenv("CACHEX_TARGETS", "cpi_stall"))
    h6a_enabled = os.getenv("H6A_ENABLED", "0").strip() == "1"
    h6_mode = os.getenv("H6_MODE", "h6a").strip().lower()
    if h6a_enabled:
        h6_mode = _normalize_h6_mode(h6_mode)
    h6a_proxy_scale = env_float("H6A_PROXY_SCALE", os.getenv("H6A_SLQ_SCALE", "1.0"))
    h6a_targets = _parse_h6a_targets(os.getenv("H6A_TARGETS", "all_noncache"))
    h3a_enabled = os.getenv("H3A_ENABLED", "0").strip() == "1"
    h3a_eta = env_float("H3A_ETA", 0.02)
    h3a_reg_lambda = env_float("H3A_REG_LAMBDA", 0.01)
    h3a_detach_base = os.getenv("H3A_DETACH_BASE", "1").strip() != "0"
    h3a_mode = os.getenv("H3A_MODE", "full").strip().lower()
    c3a_rank_k = env_int("C3A_RANK_K", l2a_rank_k if l2a_rank_k > 0 else 8)
    c3a_rank_pairs = env_int("C3A_RANK_PAIRS", l2a_rank_pairs if l2a_rank_pairs > 0 else 16)
    nc3_rank_k = env_int("NC3_RANK_K", c3a_rank_k)
    nc3_rank_pairs = env_int("NC3_RANK_PAIRS", c3a_rank_pairs)
    nc3_targets = _parse_nc3_targets(os.getenv("NC3_TARGETS", "cpi_stall"))
    j1a_rank_k = env_int("J1A_RANK_K", nc3_rank_k)
    j1a_rank_pairs = env_int("J1A_RANK_PAIRS", nc3_rank_pairs)
    j1a_targets = _parse_j1a_targets(os.getenv("J1A_TARGETS", "branch_cpi"))
    kd_lambda = env_float("KD_LAMBDA", 0.0)
    hotl_lambda = env_float("HOTL_LAMBDA", 0.0)
    hotl_scale = env_float("HOTL_SCALE", 0.1)
    teacher_ckpt = os.getenv("TEACHER_CKPT", "").strip()
    eval_topk_frac = normalize_quantile(env_float("EVAL_TOPK_FRAC", 0.05))
    near_zero_pct = env_float("NEAR_ZERO_PCT", 20.0)
    hw_encoder_type = os.getenv("HW_ENCODER", "attention").strip().lower()
    use_residual = os.getenv("USE_RESIDUAL", "0").strip() == "1"
    teacher_hw_enc = os.getenv("TEACHER_HW_ENCODER", hw_encoder_type).strip().lower()
    teacher_residual = os.getenv("TEACHER_USE_RESIDUAL", "1" if use_residual else "0").strip() == "1"
    gam_hidden = env_int("GAM_HIDDEN", 32)
    l2_dual_path = os.getenv("L2_DUAL_PATH", "0").strip() == "1"
    teacher_gam_hidden = env_int("TEACHER_GAM_HIDDEN", int(os.getenv("GAM_HIDDEN", "32")))
    teacher_l2_dual = os.getenv("TEACHER_L2_DUAL_PATH",
                                 os.getenv("L2_DUAL_PATH", "0")).strip() == "1"
    init_ckpt = os.getenv("INIT_CKPT", "").strip()
    init_strict = os.getenv("INIT_STRICT", "1").strip() != "0"
    if b1_enabled and init_ckpt:
        if init_strict:
            print("  WARNING: B1_ENABLED=1 adds branch_readout and changes branch head dims.")
            print("    Old checkpoints are incompatible with INIT_STRICT=1.")
            print("    Set INIT_STRICT=0 or use a B1-trained checkpoint.")
        else:
            print("  B1_ENABLED=1: branch_readout + branch head dims will train from scratch")
    j1a_freeze_base = os.getenv("J1A_FREEZE_BASE", "0").strip() == "1"

    # ---- Per-head endpoint fine-tuning (R1) ----
    r1_enabled = os.getenv("R1_ENABLED", "0").strip() == "1"
    r1_train_heads = tuple(
        s.strip() for s in os.getenv("R1_TRAIN_HEADS", "rob_stall").split(",")
        if s.strip()
    )
    r1_lambda_abs = env_float("R1_LAMBDA_ABS", 1.0)
    r1_lambda_delta = env_float("R1_LAMBDA_DELTA", 1.0)
    _r1_rank_raw = os.getenv("R1_LAMBDA_RANK", "0.05")
    r1_lambda_rank = tuple(
        float(s.strip()) for s in _r1_rank_raw.split(",")
    ) if "," in _r1_rank_raw else env_float("R1_LAMBDA_RANK", 0.05)
    if r1_enabled and not init_ckpt:
        raise ValueError("R1_ENABLED=1 requires INIT_CKPT for warm-start")

    # ---- Grouped endpoint sensitivity ranking for selected metrics (R1G) ----
    r1g_enabled = os.getenv("R1G_ENABLED", "0").strip() == "1"
    _r1g_rank_raw = os.getenv("R1G_GROUP_RANK_LAMBDA", "0.0")
    r1g_group_rank_lambda = tuple(
        float(s.strip()) for s in _r1g_rank_raw.split(",")
    ) if "," in _r1g_rank_raw else env_float("R1G_GROUP_RANK_LAMBDA", 0.0)
    r1g_rank_k = env_int("R1G_RANK_K", 8)
    r1g_rank_pairs = env_int("R1G_RANK_PAIRS", 16)
    if r1g_enabled and not r1_enabled:
        # From-scratch mode: R1g without R1. If user didn't set explicit targets,
        # default to all non-cache metrics.
        _explicit_heads = os.getenv("R1_TRAIN_HEADS", "").strip()
        if not _explicit_heads:
            if os.getenv("PDG_ENABLED", "0").strip() == "1":
                # PDG+R1G coexist: don't silently expand r1_train_heads to all non-cache.
                # Default to PDG targets, require explicit R1_TRAIN_HEADS for different set.
                r1_train_heads = pdg_train_heads
                print(f"  PDG+R1G: r1_train_heads defaulting to pdg_train_heads={list(r1_train_heads)}")
                print(f"    Set R1_TRAIN_HEADS explicitly to override.")
            else:
                r1_train_heads = tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)

    # ---- PDG: Positive-Degradation Grouped Rank ----
    # Same-HW-pair grouped rank on max(delta, 0) instead of |delta|.
    # Aligned with bottleneck-discovery (budget decision fidelity) eval.
    pdg_enabled = os.getenv("PDG_ENABLED", "0").strip() == "1"
    _pdg_rank_raw = os.getenv("PDG_GROUP_RANK_LAMBDA", "0.0")
    pdg_group_rank_lambda = tuple(
        float(s.strip()) for s in _pdg_rank_raw.split(",")
    ) if "," in _pdg_rank_raw else env_float("PDG_GROUP_RANK_LAMBDA", 0.0)
    pdg_rank_k = env_int("PDG_RANK_K", 8)
    pdg_rank_pairs = env_int("PDG_RANK_PAIRS", 16)
    pdg_tau = env_float("PDG_POS_TAU", 0.05)  # softplus temperature for pred_pos
    pdg_train_heads = tuple(
        s.strip() for s in os.getenv("PDG_TRAIN_HEADS",
            "l1_mpki,l1i_mpki,l2_mpki,l2_miss_rate,cpi,branch_mpki,btb_miss_rate").split(",")
        if s.strip()
    )
    if pdg_enabled and not init_ckpt:
        raise ValueError("PDG_ENABLED=1 requires INIT_CKPT for warm-start (chain mode)")
    if pdg_enabled and r1_enabled:
        raise ValueError("PDG_ENABLED and R1_ENABLED are mutually exclusive. "
                         "PDG uses its own freeze logic; use separate runs for chain steps.")
    if pdg_enabled and j1a_freeze_base:
        raise ValueError("PDG_ENABLED and J1A_FREEZE_BASE are mutually exclusive. "
                         "J1A's freeze conflicts with PDG's per-head freeze.")
    if pdg_enabled:
        print(f"  PDG chain mode: freezing all except {list(pdg_train_heads)}")

    # ---- GMR: Global Mixed-Pair Rank ----
    # Cross-HW-pair global sensitivity ranking, complementary to
    # same-HW-pair grouped rank. Addresses cross-pair scale calibration.
    gmr_enabled = os.getenv("GMR_ENABLED", "0").strip() == "1"
    gmr_lambda = env_float("GMR_LAMBDA", 0.0)
    gmr_pairs = env_int("GMR_PAIRS", 256)
    gmr_train_heads = tuple(
        s.strip() for s in os.getenv("GMR_TRAIN_HEADS", "").split(",")
        if s.strip()
    )
    if not gmr_train_heads:
        # Default: all metrics (cache + non-cache). Cross-pair scale calibration
        # benefits cache metrics too, even with L2A/C3A already active.
        gmr_train_heads = METRIC_KEYS
    _gmr_invalid = [k for k in gmr_train_heads if k not in METRIC_KEYS]
    if _gmr_invalid:
        raise ValueError(f"Invalid GMR_TRAIN_HEADS: {','.join(_gmr_invalid)}. "
                         f"Valid: {','.join(METRIC_KEYS)}")
    jepa_lambda = env_float("JEPA_LAMBDA", 0.0)
    ood_aug_lambda = env_float("OOD_AUG_LAMBDA", 0.0)
    if ood_aug_lambda < 0:
        raise ValueError(f"OOD_AUG_LAMBDA={ood_aug_lambda} must be >= 0")
    if ood_aug_lambda > 0 and not r1g_enabled:
        raise ValueError("OOD_AUG_LAMBDA>0 requires R1G_ENABLED=1")
    if ood_aug_lambda > 0 and (not isinstance(r1g_group_rank_lambda, tuple) and r1g_group_rank_lambda <= 0):
        raise ValueError("OOD_AUG_LAMBDA>0 requires R1G_GROUP_RANK_LAMBDA>0")
    if ood_aug_lambda > 0 and isinstance(r1g_group_rank_lambda, tuple) and all(v <= 0 for v in r1g_group_rank_lambda):
        raise ValueError("OOD_AUG_LAMBDA>0 requires at least one R1G_GROUP_RANK_LAMBDA entry > 0")
    if gmr_lambda > 0 and not gmr_enabled:
        raise ValueError("GMR_LAMBDA>0 requires GMR_ENABLED=1")
    if gmr_lambda > 0 and train_mode not in ("dse", "dse_direct"):
        raise ValueError("GMR_LAMBDA>0 requires TRAIN_MODE=dse or dse_direct")
    if gmr_enabled and gmr_pairs < 2:
        raise ValueError(f"GMR_PAIRS must be >= 2, got {gmr_pairs}")

    # ---- GMR Finetune: warm-start from checkpoint, add GMR on top ----
    gmr_finetune = os.getenv("GMR_FINETUNE", "0").strip() == "1"
    if gmr_finetune:
        if not gmr_enabled:
            raise ValueError("GMR_FINETUNE=1 requires GMR_ENABLED=1")
        if not init_ckpt:
            raise ValueError("GMR_FINETUNE=1 requires INIT_CKPT for warm-start")
        if r1_enabled or pdg_enabled:
            raise ValueError("GMR_FINETUNE is mutually exclusive with R1_ENABLED and PDG_ENABLED")
        if j1a_freeze_base:
            raise ValueError("GMR_FINETUNE is mutually exclusive with J1A_FREEZE_BASE")
    if jepa_lambda > 0 and gmr_finetune:
        raise ValueError(
            "JEPA_LAMBDA>0 is incompatible with GMR_FINETUNE=1. "
            "JEPA modules would be frozen by _freeze_except_heads.")
    if gmr_finetune:
        if os.getenv("TRAIN_LR", "").strip() == "":
            lr = 1e-4  # lower LR for finetune stability
        if os.getenv("TRAIN_EPOCHS", "").strip() == "":
            epochs = 50
        if os.getenv("TRAIN_PATIENCE", "").strip() == "":
            patience = 10
        # cache heads are frozen → cache_rank won't change during training
        if os.getenv("SCORE_MODE", "").strip() == "":
            os.environ["SCORE_MODE"] = "noncache_mean"
        print(f"  GMR finetune: freeze all except {list(gmr_train_heads)} heads")

    # GMR uses its own same-interval random pair sampler, independent of
    # PAIR_SAMPLER. This ensures GMR's cross-pair calibration distribution
    # does not change with the main training curriculum.
    gmr_pair_sampler = None  # initialized after dataset is loaded

    val_max_pairs = env_int("VAL_MAX_PAIRS", 50000)
    val_max_abs = env_int("VAL_MAX_ABS", 100000)
    steps_per_epoch = env_int("STEPS_PER_EPOCH", 0)
    default_score_mode = "mean_sens" if direct_delta_enabled else "mean_abs"
    score_mode = os.getenv("SCORE_MODE", default_score_mode).strip().lower()
    cache_guard = env_float("CACHE_GUARD", 0.0)

    if direct_delta_enabled:
        direct_issues = []
        if a1_enabled:
            direct_issues.append("A1_ENABLED=0")
        if h1a_enabled:
            direct_issues.append("H1A_ENABLED=0")
        if h2a_enabled:
            direct_issues.append("H2A_ENABLED=0")
        if h3a_enabled:
            direct_issues.append("H3A_ENABLED=0")
        if cachex_enabled:
            direct_issues.append("CACHEX_ENABLED=0")
        if n1a_enabled:
            direct_issues.append("N1A_ENABLED=0")
        if b1_enabled:
            direct_issues.append("B1_ENABLED=0")
        if hotl_lambda > 0:
            direct_issues.append("HOTL_LAMBDA=0")
        if kd_lambda > 0:
            direct_issues.append("KD_LAMBDA=0")
        if env_float("HW_LIP_EPS", 0.0) > 0 or env_float("HW_LIP_LAMBDA", 0.0) > 0:
            direct_issues.append("HW_LIP_EPS=0 and HW_LIP_LAMBDA=0")
        if direct_issues:
            raise ValueError(
                "TRAIN_MODE=dse_direct currently supports the shared encoders/backbone "
                "with direct delta/sensitivity supervision only; disable: "
                + ", ".join(direct_issues)
            )
        if score_mode == "mean_abs":
            raise ValueError(
                "TRAIN_MODE=dse_direct cannot use SCORE_MODE=mean_abs because "
                "endpoint heads are not trained. Use mean_sens, min_sens, cache_min_sens, "
                "cache_rank, or another sensitivity-based score mode."
            )

    if (dse_deadzone_q > 0 or l2a_rank_lambda > 0) and score_mode != "cache_min_sens":
        print("  WARNING: Consider SCORE_MODE=cache_min_sens for dead-zone or cache sensitivity ranking runs")
    if l2a_rank_lambda > 0 and dse_deadzone_q <= 0:
        print("  WARNING: L2a is being run without L1a dead-zone enabled")
    if l2a_rank_lambda > 0 and kd_lambda > 0:
        print("  WARNING: Evaluate cache sensitivity ranking without knowledge distillation before combining them")
    if c3a_lambda > 0 and score_mode not in ("cache_rank", "cache_ndcg", "cache_topk") and not r1_enabled and not r1g_enabled and not pdg_enabled:
        print("  NOTE: C3a rank-head run will select best checkpoint by cache_rank "
              "(mean of cache top-K recall and cache NDCG)")
    if nc3_lambda > 0 and not nc3_enabled:
        raise ValueError("NC3_LAMBDA>0 requires NC3_ENABLED=1")
    if j1a_lambda > 0 and score_mode not in ("j1a_rank", "j1a_ndcg", "j1a_topk", "j1a_spearman"):
        print("  NOTE: J1a branch-shared run will select best checkpoint by j1a_rank "
              "(mean of shared top-K recall and shared NDCG)")
    if nc3_lambda > 0 and train_mode not in ("dse", "dse_direct"):
        raise ValueError("NC3 ranking head requires TRAIN_MODE=dse or dse_direct")
    if j1a_lambda > 0 and not j1a_enabled:
        raise ValueError("J1A_LAMBDA>0 requires J1A_ENABLED=1")
    if j1a_lambda > 0 and train_mode not in ("dse", "dse_direct"):
        raise ValueError("J1A branch-shared score head requires TRAIN_MODE=dse or dse_direct")
    if j1a_freeze_base and not j1a_enabled:
        raise ValueError("J1A_FREEZE_BASE=1 requires J1A_ENABLED=1")
    if j1a_freeze_base and not init_ckpt:
        raise ValueError("J1A_FREEZE_BASE=1 requires INIT_CKPT to warm-start from a baseline checkpoint")
    loss_metric_keys = resolve_loss_metric_keys(loss_metrics_mode)
    if loss_metrics_mode in ("cache", "cache_only"):
        validate_cache_only_m1_recipe(
            train_mode=train_mode,
            model_type=os.getenv("MODEL_TYPE", "loop").strip().lower(),
            hw_encoder_type=hw_encoder_type,
            gam_hidden=gam_hidden,
            l2_dual_path=l2_dual_path,
            use_residual=use_residual,
            c2a_alpha=c2a_alpha,
            c3a_lambda=c3a_lambda,
            l2a_rank_lambda=l2a_rank_lambda,
            h1a_enabled=h1a_enabled,
            h2a_enabled=h2a_enabled,
            h3a_enabled=h3a_enabled,
            a1_enabled=a1_enabled,
            cachex_enabled=cachex_enabled,
            h6a_enabled=h6a_enabled,
            n1a_enabled=n1a_enabled,
            nc3_enabled=nc3_enabled,
            j1a_enabled=j1a_enabled,
            r1_enabled=r1_enabled,
        )
    if h2a_above_mode not in ("boundary", "zero"):
        raise ValueError("H2A_ABOVE_MODE must be 'boundary' or 'zero'")
    if a1_knob_mode not in ("full47", "min"):
        raise ValueError("A1_KNOB_MODE must be 'full47' or 'min'")
    if hotl_lambda > 0:
        raise ValueError(
            "Legacy HOTL auxiliary loss is deprecated; "
            "use H1A_ENABLED or H2A_ENABLED feature-only branches instead."
        )
    if a1_enabled and (h1a_enabled or h2a_enabled or h3a_enabled):
        raise ValueError(
            "A1_ENABLED cannot be combined with H1A_ENABLED, H2A_ENABLED, or H3A_ENABLED"
        )
    if h1a_enabled and h2a_enabled:
        raise ValueError("H1A_ENABLED and H2A_ENABLED are mutually exclusive; enable only one")
    if h3a_enabled and not h2a_enabled:
        raise ValueError("H3a requires H2a raw HOTL analytic features")
    if h3a_enabled and train_mode != "dse":
        raise ValueError("H3a is a cache-delta correction branch and requires TRAIN_MODE=dse")
    if h3a_mode not in ("full", "prior_only", "residual_only"):
        raise ValueError("H3A_MODE must be one of: full, prior_only, residual_only")
    if h3a_enabled and c2a_alpha > 0:
        print("  NOTE: Evaluate the cache-delta correction branch with C2A_ALPHA=0 first; "
              "current run enables both.")
    if h3a_enabled and c3a_lambda > 0:
        print("  NOTE: Evaluate the cache-delta correction branch without the cache-aware ranking score head first; "
              "current run enables both.")
    if h3a_enabled and h3a_reg_lambda <= 0:
        print("  WARNING: H3a is running with H3A_REG_LAMBDA<=0; "
              "formal runs should usually keep residual/correction regularization on.")

    best_model_path = f"{experiment_tag}_best_model.pth"
    history_path = f"{experiment_tag}_training_history.json"
    best_cache_rank_path = f"{experiment_tag}_best_cache_rank.pth"
    best_cache_min_path = f"{experiment_tag}_best_cache_min_sens.pth"
    best_nc3_rank_path = f"{experiment_tag}_best_nc3_rank.pth"
    best_nc3_min_sens_path = f"{experiment_tag}_best_nc3_min_sens.pth"
    best_j1a_rank_path = f"{experiment_tag}_best_j1a_rank.pth"

    print(
        f"[{experiment_tag}] device={device} lr={lr} epochs={epochs} "
        f"patience={patience} batch={batch_size} hidden={hidden} "
        f"split={split_mode} loop_steps={loop_steps} heads={num_heads}"
    )
    print(f"  loss_metrics={loss_metrics_mode} -> {','.join(loss_metric_keys)}")
    print(f"Loading dataset from {data_path} ...")
    dataset = MicroArchDataset(data_path)
    print(f"  total samples: {len(dataset)}")

    val_hw = set()
    if split_mode == "random":
        n_total = len(dataset)
        n_train = int(n_total * 0.8)
        indices = np.random.default_rng(seed).permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        val_groups = ["random"]
    elif split_mode == "hw":
        # Hardware-level split: train on 80% HW configs, test on 20%
        hw_ids = np.array([s.get("hw_id", i) for i, s in enumerate(dataset.samples)])
        unique_hw = np.unique(hw_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_hw)
        n_train_hw = int(len(unique_hw) * 0.8)
        train_hw = set(unique_hw[:n_train_hw].tolist())
        val_hw = set(unique_hw[n_train_hw:].tolist())
        train_idx = np.array([i for i in range(len(dataset)) if hw_ids[i] in train_hw])
        val_idx = np.array([i for i in range(len(dataset)) if hw_ids[i] in val_hw])
        val_groups = [f"hw_{h}" for h in sorted(val_hw)[:8]]
        print(f"  HW split: {len(train_hw)} train HW + {len(val_hw)} test HW")
    else:
        groups = dataset.get_benchmarks()
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
        train_idx, val_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
        val_groups = sorted({groups[i] for i in val_idx})

    print(
        f"  train: {len(train_idx)}  val: {len(val_idx)}  "
        f"val_groups({len(val_groups)}): {val_groups[:8]}..."
    )

    train_idx_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_abs_idx_np = np.asarray(val_idx, dtype=np.int64)
    if val_max_abs > 0 and val_abs_idx_np.size > val_max_abs:
        val_abs_idx_np = np.random.default_rng(seed + 3).choice(
            val_abs_idx_np, size=val_max_abs, replace=False)
    val_abs_idx = torch.as_tensor(val_abs_idx_np, dtype=torch.long, device=device)

    print("Building PairStore ...", flush=True)
    t0 = time.time()
    store = PairStore(dataset, device)
    print(f"  PairStore built in {time.time() - t0:.1f}s")

    pair_sampler_type = os.getenv("PAIR_SAMPLER", "random").strip().lower()
    if l2a_rank_lambda > 0 and pair_sampler_type != "intervention":
        print("  WARNING: Consider PAIR_SAMPLER=intervention for cache sensitivity ranking")
    if train_mode in ("dse", "dse_direct"):
        need_rank = (
            (l2a_rank_lambda > 0)
            or (c3a_lambda > 0)
            or (nc3_enabled and nc3_lambda > 0)
            or (j1a_enabled and j1a_lambda > 0)
            or r1g_enabled
            or pdg_enabled
        )
        if hard_mining_enabled and not need_rank:
            raise ValueError("HARD_MINING_ENABLED=1 requires at least one rank loss active"
                             " (L2A/C3A/NC3/J1A/R1G/PDG)")
        tau_pair_sampler = None
        if dse_deadzone_q > 0:
            tau_pair_sampler = TrainingPairSampler(
                dataset, train_idx, device, seed + 41, enable_rank=False
            )
        if pair_sampler_type == "intervention":
            # Load lightweight HW config JSON (305 entries, ~100KB) instead of
            # the full raw pickle (5.6GB) to avoid OOM during pair type building.
            import json as _json
            hw_json_path = os.getenv("HW_JSON", DEFAULT_HW_JSON_PATH)
            try:
                with open(hw_json_path) as f:
                    jdata = _json.load(f)
                raw_list = [{"hardware": d, "benchmark": "", "interval_id": 0}
                            for d in (jdata if isinstance(jdata, list) else jdata.values())]
                print(f"  InterventionPairSampler: loaded {len(raw_list)} HW configs from JSON")
            except FileNotFoundError:
                print("  WARNING: HW JSON not found, falling back to raw pickle (may OOM)")
                import pickle as _pickle
                with open(data_path, 'rb') as f:
                    raw_samples = _pickle.load(f)
                # Extract only unique HW configs
                seen = set()
                raw_list = []
                for s in raw_samples:
                    hid = _extract_hw_id(s)
                    if hid >= 0 and hid not in seen:
                        seen.add(hid)
                        raw_list.append(s)
                print(f"  Extracted {len(raw_list)} unique HW configs from raw data")
            train_pair = InterventionPairSampler(
                dataset, raw_list, train_idx, device, seed + 4, enable_rank=need_rank)
        else:
            train_pair = TrainingPairSampler(
                dataset, train_idx, device, seed + 4, enable_rank=need_rank
            )
    else:
        train_pair = None
        tau_pair_sampler = None
    val_pair = SimplePairSampler(dataset, val_idx, device, seed + 2)
    val_h1, val_h2 = val_pair.make_eval_pairs(val_max_pairs)
    # GMR: independent same-interval random pair sampler, decoupled from PAIR_SAMPLER
    if gmr_enabled:
        gmr_pair_sampler = TrainingPairSampler(
            dataset, train_idx, device, seed + 47, enable_rank=False)
    # Hard rank-candidate mining: pre-compute info scores for biased rank sampling
    if hard_mining_enabled and train_pair is not None:
        if hasattr(train_pair, 'compute_hard_scores'):
            print(f"  Hard mining: computing group info for {','.join(hard_mining_targets)} ...", flush=True)
            t0 = time.time()
            train_pair.compute_hard_scores(store, hard_mining_targets)
            print(f"  Hard mining: {len(train_pair.hard_scores)} groups scored in {time.time()-t0:.1f}s")
            print(f"  Hard mining: ratio={hard_mining_ratio}, top-N groups will be oversampled")
        else:
            print("  WARNING: HARD_MINING_ENABLED=1 but pair sampler does not support hard mining")
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, math.ceil(len(train_idx) / batch_size))
    print(
        f"  val delta pairs: {val_h1.numel()}  "
        f"val abs samples: {val_abs_idx.numel()}  "
        f"steps/epoch: {steps_per_epoch}"
    )

    deadzone_tau = None
    if train_mode in ("dse", "dse_direct") and dse_deadzone_q > 0:
        deadzone_tau = estimate_deadzone_tau(
            tau_pair_sampler, store, dse_deadzone_calib_pairs, dse_deadzone_q
        )
        if deadzone_tau is None:
            print("  L1a dead-zone disabled: failed to estimate per-metric tau")
        else:
            tau_np = deadzone_tau.detach().cpu().numpy()
            q_print = normalize_quantile(dse_deadzone_q)
            print(
                f"  L1a dead-zone q={q_print:.3f}: "
                + ", ".join(f"{k}={tau_np[i]:.4f}" for i, k in enumerate(METRIC_KEYS))
            )

    model_type = os.getenv("MODEL_TYPE", "loop").strip().lower()
    compiled = os.getenv("TORCH_COMPILE", "0") == "1"
    h6_raw_feature_names = DEFAULT_H6_RAW_FEATURE_NAMES

    if (h6a_enabled or nc3_enabled or j1a_enabled) and model_type not in ("film", "swiglu"):
        raise ValueError("H6/NC3/J1A features are only implemented for MODEL_TYPE=film/swiglu")

    if model_type in ("film", "swiglu"):
        if no_rfp:
            print("  NOTE: NO_RFP=1 ignored for film/swiglu — RFP intentionally unused by design")
        interaction = "swiglu" if model_type == "swiglu" else "film"
        hotl_stats = None
        if h1a_enabled or h2a_enabled:
            hotl_stats = {
                "gwfp_mean": float(dataset.stats["gwfp_mean"]),
                "gwfp_std": float(dataset.stats["gwfp_std"]),
                "gwifp_mean": float(dataset.stats["gwifp_mean"]),
                "gwifp_std": float(dataset.stats["gwifp_std"]),
                "cache_hw_mean": dataset.stats["hw_mean"][16:19].tolist(),
                "cache_hw_std": dataset.stats["hw_std"][16:19].tolist(),
            }
        h6a_stats = None
        if h6a_enabled:
            h6a_stats = {
                "gwfp_ratio_half_mean": float(dataset.stats["gwfp_ratio_half_mean"]),
                "gwfp_ratio_half_std": float(dataset.stats["gwfp_ratio_half_std"]),
                "gwfp_ratio_cache2dual_mean": float(dataset.stats["gwfp_ratio_cache2dual_mean"]),
                "gwfp_ratio_cache2dual_std": float(dataset.stats["gwfp_ratio_cache2dual_std"]),
                "gwifp_ratio_half_mean": float(dataset.stats["gwifp_ratio_half_mean"]),
                "gwifp_ratio_half_std": float(dataset.stats["gwifp_ratio_half_std"]),
                "gwifp_ratio_cache2dual_mean": float(dataset.stats["gwifp_ratio_cache2dual_mean"]),
                "gwifp_ratio_cache2dual_std": float(dataset.stats["gwifp_ratio_cache2dual_std"]),
            }
        model = FiLMUnifiedModel(
            embed_dim=hidden, dropout=dropout, noise_std=noise_std,
            interaction_type=interaction, hw_encoder=hw_encoder_type,
            use_residual=use_residual,
            c2a_alpha=c2a_alpha, c3a_enabled=(c3a_lambda > 0),
            nc3_enabled=nc3_enabled,
            nc3_target_metrics=nc3_targets,
            nc3_detach_base=nc3_detach_base,
            j1a_enabled=j1a_enabled,
            j1a_target_metrics=j1a_targets,
            j1a_detach_base=j1a_detach_base,
            a1_enabled=a1_enabled, a1_basis_hidden=a1_basis_hidden, a1_knob_mode=a1_knob_mode,
            h1a_enabled=h1a_enabled,
            h1a_stats=hotl_stats if h1a_enabled else None,
            h2a_enabled=h2a_enabled,
            h2a_stats=hotl_stats if h2a_enabled else None,
            h2a_concave=h2a_concave,
            h2a_above_mode=h2a_above_mode,
            cachex_enabled=cachex_enabled,
            cachex_target_metrics=cachex_targets,
            h6a_enabled=h6a_enabled,
            h6a_stats=h6a_stats,
            h6_mode=h6_mode,
            h6a_proxy_scale=h6a_proxy_scale,
            h6a_target_metrics=h6a_targets,
            h3a_enabled=h3a_enabled,
            h3a_eta=h3a_eta,
            h3a_detach_base=h3a_detach_base,
            h3a_mode=h3a_mode,
            n1a_enabled=n1a_enabled,
            r2_enabled=r2_enabled,
            b1_enabled=b1_enabled,
            s2_enabled=s2_enabled,
            s2_alpha=s2_alpha,
            jepa_enabled=(jepa_lambda > 0),
            direct_delta_enabled=direct_delta_enabled,
        ).to(device)
        h6_raw_feature_names = tuple(getattr(model, "h6_raw_feature_names", DEFAULT_H6_RAW_FEATURE_NAMES))
        a1_knob_meta = None
        if a1_enabled:
            _cache_m = ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate")
            a1_knob_meta = {
                name: tuple(HW_PARAM_NAMES[i] for i in model.a1_cache_heads[name].knob_indices)
                for name in _cache_m
            }
        c2a_str = f" c2a_alpha={c2a_alpha}" if c2a_alpha > 0 else ""
        c3a_str = f" c3a" if c3a_lambda > 0 else ""
        cachex_str = f" cachex[targets={','.join(cachex_targets)}]" if cachex_enabled else ""
        nc3_str = f" nc3[{','.join(nc3_targets)}]" if nc3_enabled else ""
        j1a_str = f" j1a[{','.join(j1a_targets)}]" if j1a_enabled else ""
        n1a_str = " n1a[rob_stall,lq_stall,sq_stall]" if n1a_enabled else ""
        r2_str = " r2[rob_adapter]" if r2_enabled else ""
        b1_str = " b1[branch_readout]" if b1_enabled else ""
        s2_str = f" s2[α={s2_alpha:g}]" if s2_enabled else ""
        r1_rank_display = ",".join(f"{v:g}" for v in r1_lambda_rank) if isinstance(r1_lambda_rank, tuple) else f"{r1_lambda_rank:g}"
        r1_str = f" r1[train={','.join(r1_train_heads)},rank={r1_rank_display}]" if r1_enabled else ""
        _r1g_display = ",".join(f"{v:g}" for v in r1g_group_rank_lambda) if isinstance(r1g_group_rank_lambda, tuple) else f"{r1g_group_rank_lambda:g}"
        _r1g_targets = f",targets={','.join(r1_train_heads)}" if not r1_enabled else ""
        r1g_str = f" r1g[group_rank={_r1g_display}{_r1g_targets}]" if r1g_enabled else ""
        _pdg_display = ",".join(f"{v:g}" for v in pdg_group_rank_lambda) if isinstance(pdg_group_rank_lambda, tuple) else f"{pdg_group_rank_lambda:g}"
        pdg_str = f" pdg[group_rank={_pdg_display},targets={','.join(pdg_train_heads)}]" if pdg_enabled else ""
        gmr_str = f" gmr[λ={gmr_lambda:g},pairs={gmr_pairs},targets={','.join(gmr_train_heads)}]" if gmr_enabled else ""
        a1_str = f" a1[{a1_knob_mode},basis={a1_basis_hidden}]" if a1_enabled else ""
        h1a_str = " h1a" if h1a_enabled else ""
        h2a_str = " h2a" if h2a_enabled else ""
        h6a_str = (
            f" h6[{h6_mode},proxy_scale={h6a_proxy_scale:g},targets={','.join(h6a_targets)}]"
            if h6a_enabled else ""
        )
        h3a_str = f" h3a[{h3a_mode}]_eta={h3a_eta:g} detach={int(h3a_detach_base)} reg={h3a_reg_lambda:g}" if h3a_enabled else ""
        ood_aug_str = f" ood_aug={ood_aug_lambda}" if ood_aug_lambda > 0 else ""
        jepa_str = f" jepa={jepa_lambda}" if jepa_lambda > 0 else ""
        direct_str = " direct_delta" if direct_delta_enabled else ""
        print(f"  model={model_type} hw_enc={hw_encoder_type} residual={use_residual}{direct_str}"
              f"{c2a_str}{c3a_str}{cachex_str}{nc3_str}{j1a_str}{n1a_str}{r2_str}{b1_str}{s2_str}{jepa_str}{r1_str}{r1g_str}{pdg_str}{gmr_str}{ood_aug_str}{a1_str}{h1a_str}{h2a_str}{h6a_str}{h3a_str} hotl={hotl_lambda} params={count_params(model):,}")
        if h2a_enabled:
            print(f"  H2a config: concave={int(h2a_concave)} above_mode={h2a_above_mode}")
        if h6a_enabled:
            if h6_mode == "h6a":
                print("  H6a uses gwfp/gwifp ratio channels 3 and 4 as half/cache and cache/dual spatial-compression proxies")
            elif h6_mode == "h6c":
                print("  H6c bins those gwfp/gwifp ratio proxies across execution-window scales to approximate reuse-bin sensitivity")
            else:
                print("  H6d focuses on cache/dual-block spatial-compression proxies to emphasize adjacent-block and prefetch-related behavior")
        if init_ckpt:
            _load_initial_state_dict(model, init_ckpt, device, strict=init_strict)
        if r1_enabled:
            _freeze_except_heads(model, r1_train_heads, caller="R1")
        if pdg_enabled:
            # When PDG+R1G coexist, merge both head sets so both losses get gradients
            _pdg_heads = set(pdg_train_heads)
            if r1g_enabled:
                _pdg_heads |= set(r1_train_heads)
                print(f"  PDG+R1G joint: unfreezing heads {sorted(_pdg_heads)}")
            _freeze_except_heads(model, tuple(sorted(_pdg_heads)), caller="PDG")
        if gmr_finetune:
            _freeze_except_heads(model, gmr_train_heads, caller="GMR")
        if j1a_freeze_base:
            _freeze_all_but_j1a(model)
            print("  J1A probe runtime mode: frozen base=eval, j1a_score_head=train")
        if compiled:
            # Save original forward_absolute before compile wraps __call__
            _orig_forward_absolute = model.forward_absolute
            _model_uncompiled = model  # kept for R1G/GMR/JEPA forward to avoid recompilation
            model = torch.compile(model, mode="default")
            # Also compile forward_absolute so abs-only training, eval, HW smoothing,
            # and KD student/teacher paths benefit from kernel fusion.
            model.forward_absolute = torch.compile(
                _orig_forward_absolute, mode="reduce-overhead")
        else:
            _model_uncompiled = model

        # Optional knowledge distillation, disabled when KD_LAMBDA=0.
        # The teacher supplies targets for student predictions on held-out hardware.
        # Hardware-holdout validity depends on the teacher's training data: a teacher
        # trained on test-hardware labels can transfer test information to the student.
        # Teacher predictions alone do not establish the absence of label leakage.
        teacher = None
        kd_hw = None
        if kd_lambda > 0 and teacher_ckpt and split_mode == "hw" and val_hw:
            _saved_gam = os.environ.get("GAM_HIDDEN", None)
            _saved_l2 = os.environ.get("L2_DUAL_PATH", None)
            os.environ["GAM_HIDDEN"] = str(teacher_gam_hidden)
            os.environ["L2_DUAL_PATH"] = "1" if teacher_l2_dual else "0"
            teacher = FiLMUnifiedModel(
                embed_dim=hidden, dropout=0.0, noise_std=0.0,
                interaction_type=interaction, hw_encoder=teacher_hw_enc,
                use_residual=teacher_residual,
                b1_enabled=b1_enabled,
            ).to(device)
            for _k, _saved in [("GAM_HIDDEN", _saved_gam), ("L2_DUAL_PATH", _saved_l2)]:
                if _saved is not None:
                    os.environ[_k] = _saved
                else:
                    os.environ.pop(_k, None)
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
            teacher.eval()
            if compiled:
                _orig_teacher_forward = teacher.forward_absolute
                teacher = torch.compile(teacher, mode="reduce-overhead")
                teacher.forward_absolute = torch.compile(
                    _orig_teacher_forward, mode="reduce-overhead")
            for p in teacher.parameters():
                p.requires_grad = False
            # Build held-out HW vectors: hw_id -> hw_vector
            kd_hw = {}
            for h in val_hw:
                for s in dataset.samples:
                    if s.get("hw_id", -1) == h:
                        kd_hw[h] = s["hw"].to(device)
                        break
            kd_hw_list = sorted(kd_hw.keys())
            print(f"  KD teacher loaded, {len(kd_hw_list)} held-out HW configs")
            # GP augmentation: add GP-validated synthetic HW configs
            gp_augment = os.getenv("GP_AUGMENT", "").strip()
            if gp_augment:
                gp_data = pickle.load(open(gp_augment, 'rb'))
                for synth_id, hw_vec in gp_data['hw_dict'].items():
                    kd_hw[synth_id] = torch.as_tensor(hw_vec, device=device)
                kd_hw_list = sorted(kd_hw.keys())
                print(f"  GP augment: +{len(gp_data['hw_dict'])} synthetic HW, "
                      f"total {len(kd_hw_list)} KD targets")
    else:
        model = UnifiedLoopModel(
            embed_dim=hidden, dropout=dropout, loop_steps=loop_steps,
            num_heads=num_heads, ff_mult=ff_mult, noise_std=noise_std,
        ).to(device)
        a1_knob_meta = None
        if compiled:
            _orig_forward_absolute = model.forward_absolute
            model = torch.compile(model, mode="default")
            model.forward_absolute = torch.compile(
                _orig_forward_absolute, mode="reduce-overhead")
        print(f"  model=loop params={count_params(model):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying freeze settings")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        p = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "train_loss": [],
        "val_abs_spearman": [],
        "val_delta_spearman": [],
        "val_sensitivity_spearman": [],
        "val_pdg_spearman": [],
        "val_topk_recall": [],
        "val_ndcg": [],
        "val_direction_accuracy": [],
        "val_near_zero_dir_rate": [],
        "loss_metrics_mode": loss_metrics_mode,
        "loss_metric_keys": list(loss_metric_keys),
    }
    if a1_enabled:
        history["train_a1_baseline_abs"] = []
        history["train_a1_add_abs"] = []
        history["train_a1_add_ratio"] = []
        history["train_a1_coeff_abs_mean"] = []
        history["train_a1_basis_abs_mean"] = []
        history["train_a1_top_knob_contrib"] = []
    if h6a_enabled:
        history["h6_mode"] = h6_mode
        history["h6_raw_feature_names"] = list(h6_raw_feature_names)
        history["train_h6_feat_abs_mean"] = []
        history["train_h6_raw_summary"] = []
        history["train_h6_sat0_rate"] = []
        history["train_h6_sat1_rate"] = []
        history["train_h6_sat0_per_feature"] = []
        history["train_h6_sat1_per_feature"] = []
    if cachex_enabled:
        history["train_cachex_feat_abs_mean"] = []
    if h3a_enabled:
        history["train_h3_residual_abs"] = []
        history["train_h3_correction_abs"] = []
        history["train_h3_base_delta_abs"] = []
        history["train_h3_correction_ratio"] = []
    best_score = -1.0
    best_cache_rank = -1.0
    best_cache_min_sens = -1.0
    best_nc3_rank = -1.0
    best_nc3_min_sens = -1.0
    best_j1a_rank = -1.0
    best_epoch = 0
    bad_epochs = 0

    cache_metric_keys = ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate")
    cache_metric_idx = torch.tensor(
        [list(METRIC_KEYS).index(m) for m in cache_metric_keys],
        device=device,
        dtype=torch.long,
    )
    loss_metric_idx = torch.tensor(
        [list(METRIC_KEYS).index(m) for m in loss_metric_keys],
        device=device,
        dtype=torch.long,
    )
    nc3_metric_keys = tuple(nc3_targets)
    nc3_metric_idx = torch.tensor(
        [list(METRIC_KEYS).index(m) for m in nc3_metric_keys],
        device=device,
        dtype=torch.long,
    )
    j1a_metric_keys = tuple(j1a_targets)
    j1a_metric_idx = torch.tensor(
        [list(METRIC_KEYS).index(m) for m in j1a_metric_keys],
        device=device,
        dtype=torch.long,
    )

    def _ood_weight(hw_a, hw_b):
        """Per-pair weight by L0 HW distance, same definition as OOD eval."""
        if ood_aug_lambda <= 0:
            return 1.0
        l0 = (torch.abs(hw_a - hw_b) > 1e-6).sum(dim=-1).float()
        l0_norm = l0 / float(hw_a.shape[-1])
        raw_w = 1.0 + ood_aug_lambda * l0_norm
        return raw_w / (1.0 + 0.5 * ood_aug_lambda)

    print("Starting training loop ...", flush=True)
    for epoch in range(epochs):
        model.train()
        if j1a_freeze_base:
            _set_j1a_probe_mode(model)
        epoch_start = time.time()
        total_loss = 0.0
        n_steps = 0
        a1_baseline_abs_total = 0.0
        a1_add_abs_total = 0.0
        a1_add_ratio_total = 0.0
        a1_coeff_abs_total = 0.0
        a1_basis_abs_total = 0.0
        a1_stats_steps = 0
        a1_top_contrib_sums = _init_a1_top_contrib_sums(a1_knob_meta)
        h6a_feat_abs_total = 0.0
        h6a_raw_sum = np.zeros(len(h6_raw_feature_names), dtype=np.float64)
        h6a_raw_min = float('inf')
        h6a_raw_max = float('-inf')
        h6a_sat0_sum = np.zeros(len(h6_raw_feature_names), dtype=np.float64)
        h6a_sat1_sum = np.zeros(len(h6_raw_feature_names), dtype=np.float64)
        h6a_stats_steps = 0
        cachex_feat_abs_total = 0.0
        cachex_stats_steps = 0
        h3_residual_abs_total = 0.0
        h3_correction_abs_total = 0.0
        h3_base_abs_total = 0.0
        h3_stats_steps = 0

        for _ in range(steps_per_epoch):
            if train_mode in ("dse", "dse_direct"):
                h1, h2 = train_pair.sample_batch(batch_size)
                output, target_delta, target_a, target_b = run_model_on_indices(
                    model, store, h1, h2, no_rfp=no_rfp)
                pred_delta = stack_outputs(output, "delta")
                pred_delta_l = pred_delta.index_select(1, loss_metric_idx)
                target_delta_l = target_delta.index_select(1, loss_metric_idx)
                delta_loss = F.smooth_l1_loss(pred_delta_l, target_delta_l,
                                               reduction="mean")
                dir_loss_raw = F.binary_cross_entropy_with_logits(
                    pred_delta_l * dse_dir_temp,
                    (target_delta_l > 0).float(), reduction="none")
                # L1a: margin-aware dead-zone weight
                if deadzone_tau is not None:
                    abs_delta = target_delta_l.abs()
                    tau = deadzone_tau.index_select(0, loss_metric_idx).unsqueeze(0)
                    s_val = torch.clamp(tau / 2.0, min=1e-6)
                    w_dir = torch.sigmoid((abs_delta - tau) / s_val)
                    dir_loss = (w_dir * dir_loss_raw).mean()
                else:
                    dir_loss = dir_loss_raw.mean()
                if direct_delta_enabled:
                    loss = dse_lambda_delta * delta_loss + dse_lambda_dir * dir_loss
                else:
                    pred_abs_a = stack_outputs(output, "absolute_a")
                    pred_abs_b = stack_outputs(output, "absolute_b")
                    pred_abs_a_l = pred_abs_a.index_select(1, loss_metric_idx)
                    pred_abs_b_l = pred_abs_b.index_select(1, loss_metric_idx)
                    target_a_l = target_a.index_select(1, loss_metric_idx)
                    target_b_l = target_b.index_select(1, loss_metric_idx)
                    abs_loss = 0.5 * (
                        F.smooth_l1_loss(pred_abs_a_l, target_a_l, reduction="mean") +
                        F.smooth_l1_loss(pred_abs_b_l, target_b_l, reduction="mean"))
                    loss = abs_loss + dse_lambda_delta * delta_loss + dse_lambda_dir * dir_loss

                # R1: per-metric endpoint sensitivity ranking loss
                if r1_enabled and (isinstance(r1_lambda_rank, tuple) or r1_lambda_rank > 0):
                    r1_rank_loss = 0.0
                    for i, name in enumerate(r1_train_heads):
                        m_idx = list(METRIC_KEYS).index(name)
                        pred_d = pred_delta[:, m_idx]
                        true_d = target_delta[:, m_idx]
                        pred_s = pred_d.abs()
                        true_s = true_d.abs()
                        # Pearson correlation loss: 1 - corr
                        pred_z = (pred_s - pred_s.mean()) / (pred_s.std() + 1e-8)
                        true_z = (true_s - true_s.mean()) / (true_s.std() + 1e-8)
                        corr_loss = 1.0 - (pred_z * true_z).mean()
                        w = r1_lambda_rank[i] if isinstance(r1_lambda_rank, tuple) else r1_lambda_rank
                        r1_rank_loss = r1_rank_loss + w * corr_loss
                    loss = loss + r1_rank_loss

                # Grouped endpoint sensitivity ranking within each hardware pair (R1G).
                # Each metric in r1_train_heads gets independent grouped rank loss.
                if r1g_enabled and (isinstance(r1g_group_rank_lambda, tuple) or r1g_group_rank_lambda > 0):
                    r_h1, r_h2, r_group_sizes = train_pair.sample_rank_batch(
                        r1g_rank_pairs, r1g_rank_k, hard_ratio=rank_hard_ratio)
                    r_out, r_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, r_h1, r_h2, no_rfp=no_rfp)
                    r_pred_sens = stack_outputs(r_out, "sensitivity")
                    r_true_sens = r_tgt_delta.abs()
                    r1g_losses = []
                    for i, name in enumerate(r1_train_heads):
                        m_idx = METRIC_KEYS.index(name)
                        w = r1g_group_rank_lambda[i] if isinstance(r1g_group_rank_lambda, tuple) else r1g_group_rank_lambda
                        if w <= 0:
                            continue
                        offset = 0
                        for group_size in r_group_sizes:
                            next_offset = offset + group_size
                            st = r_true_sens[offset:next_offset, m_idx]
                            sp = r_pred_sens[offset:next_offset, m_idx]
                            st_c = st - st.mean()
                            sp_c = sp - sp.mean()
                            st_var = torch.mean(st_c.square())
                            if st_var > 1e-8:
                                st_std = torch.sqrt(st_var + 1e-8)
                                sp_std = torch.sqrt(torch.mean(sp_c.square()) + 1e-8)
                                corr = torch.mean(st_c * sp_c) / (st_std * sp_std + 1e-8)
                                if ood_aug_lambda > 0:
                                    ood_w = _ood_weight(
                                        store.hw[r_h1[offset:offset + 1]],
                                        store.hw[r_h2[offset:offset + 1]],
                                    )
                                    r1g_losses.append((ood_w * w * (1.0 - corr), ood_w))
                                else:
                                    r1g_losses.append(w * (1.0 - corr))
                            offset = next_offset
                    if r1g_losses:
                        if ood_aug_lambda > 0:
                            weighted_sum = torch.stack([item[0] for item in r1g_losses]).sum()
                            weight_sum = torch.stack([item[1] for item in r1g_losses]).sum()
                            r1g_loss = weighted_sum / (weight_sum + 1e-8)
                        else:
                            r1g_loss = torch.stack(r1g_losses).mean()
                        loss = loss + r1g_loss

                # PDG: Positive-Degradation Grouped Rank
                # Same-HW-pair grouped rank on max(delta, 0), aligned with
                # bottleneck-discovery (budget decision fidelity) eval.
                if pdg_enabled and (isinstance(pdg_group_rank_lambda, tuple) or pdg_group_rank_lambda > 0):
                    r_h1, r_h2, r_group_sizes = train_pair.sample_rank_batch(
                        pdg_rank_pairs, pdg_rank_k, hard_ratio=rank_hard_ratio)
                    r_out, r_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, r_h1, r_h2, no_rfp=no_rfp)
                    r_pred_delta = stack_outputs(r_out, "delta")
                    # softplus: positive-degradation score with gradient for negative preds
                    # relu would zero-out gradient for pred_delta < 0, skipping sign-miss samples
                    r_true_pos = torch.relu(r_tgt_delta)
                    r_pred_pos = pdg_tau * F.softplus(r_pred_delta / pdg_tau)
                    pdg_losses = []
                    for i, name in enumerate(pdg_train_heads):
                        m_idx = METRIC_KEYS.index(name)
                        w = pdg_group_rank_lambda[i] if isinstance(pdg_group_rank_lambda, tuple) else pdg_group_rank_lambda
                        if w <= 0:
                            continue
                        offset = 0
                        for group_size in r_group_sizes:
                            next_offset = offset + group_size
                            st = r_true_pos[offset:next_offset, m_idx]
                            sp = r_pred_pos[offset:next_offset, m_idx]
                            st_c = st - st.mean()
                            sp_c = sp - sp.mean()
                            st_var = torch.mean(st_c.square())
                            if st_var > 1e-8:
                                st_std = torch.sqrt(st_var + 1e-8)
                                sp_std = torch.sqrt(torch.mean(sp_c.square()) + 1e-8)
                                corr = torch.mean(st_c * sp_c) / (st_std * sp_std + 1e-8)
                                pdg_losses.append(w * (1.0 - corr))
                            offset = next_offset
                    if pdg_losses:
                        pdg_loss = torch.stack(pdg_losses).mean()
                        loss = loss + pdg_loss

                # GMR: Global Mixed-Pair Rank
                # Cross-HW-pair global sensitivity ranking. Unlike same-HW-pair
                # grouped ranking via sample_rank_batch, GMR uses its own independent
                # same-interval random pair sampler (gmr_pair_sampler), decoupled
                # from PAIR_SAMPLER, then computes a flat global Pearson correlation
                # without grouping. Addresses cross-pair scale calibration.
                if gmr_enabled and gmr_lambda > 0:
                    g_h1, g_h2 = gmr_pair_sampler.sample_batch(gmr_pairs)
                    g_out, g_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, g_h1, g_h2, no_rfp=no_rfp)
                    g_pred_sens = stack_outputs(g_out, "sensitivity")
                    g_true_sens = g_tgt_delta.abs()
                    gmr_losses = []
                    for name in gmr_train_heads:
                        m_idx = METRIC_KEYS.index(name)
                        st = g_true_sens[:, m_idx]
                        sp = g_pred_sens[:, m_idx]
                        st_c = st - st.mean()
                        sp_c = sp - sp.mean()
                        st_var = torch.mean(st_c.square())
                        if st_var > 1e-8:
                            st_std = torch.sqrt(st_var + 1e-8)
                            sp_std = torch.sqrt(torch.mean(sp_c.square()) + 1e-8)
                            corr = torch.mean(st_c * sp_c) / (st_std * sp_std + 1e-8)
                            gmr_losses.append(1.0 - corr)
                    if gmr_losses:
                        gmr_loss = gmr_lambda * torch.stack(gmr_losses).mean()
                        loss = loss + gmr_loss

                # JEPA: learn hardware transition in representation space.
                # Default (non-stratified): reuses DSE forward output (same h1/h2),
                # acting as a smoothness regularizer since most pairs are close.
                # JEPA_STRATIFIED=1: stratified close/medium/far, ratio controlled by
                # JEPA_RATIO env var (default "1:1:1", e.g. "4:3:3").
                if jepa_lambda > 0:
                    jepa_stratified = os.getenv("JEPA_STRATIFIED", "0").strip() == "1"
                    if jepa_stratified:
                        jepa_pairs = env_int("JEPA_PAIRS", 256)
                        # Parse JEPA_RATIO: "4:3:3" → close/medium/far fractions
                        _ratio_str = os.getenv("JEPA_RATIO", "1:1:1")
                        _ratios = parse_jepa_ratio(_ratio_str)
                        n_target = jepa_pairs
                        pool_size = n_target * 6
                        j_h1, j_h2 = train_pair.sample_batch(max(pool_size, batch_size))
                        j_l0 = (torch.abs(store.hw[j_h1] - store.hw[j_h2]) > 1e-6).sum(dim=-1)
                        chosen = stratified_jepa_indices(j_l0, n_target, _ratios)
                        j_h1 = j_h1[chosen]
                        j_h2 = j_h2[chosen]
                        hw_a = store.hw[j_h1]
                        hw_b = store.hw[j_h2]
                        j_out, _, _, _ = run_model_on_indices(
                            _model_uncompiled, store, j_h1, j_h2, no_rfp=no_rfp)
                        repr_a = j_out.get("_jepa_repr_a")
                        repr_b = j_out.get("_jepa_repr_b")
                    else:
                        # Non-stratified: reuse DSE output (zero extra forward)
                        repr_a = output.get("_jepa_repr_a")
                        repr_b = output.get("_jepa_repr_b")
                        hw_a = store.hw[h1]
                        hw_b = store.hw[h2]
                    if repr_a is not None and repr_b is not None:
                        delta = hw_b - hw_a
                        abs_delta = delta.abs()
                        changed = (abs_delta > 1e-6).float()
                        l0 = changed.sum(dim=-1, keepdim=True) / float(hw_a.shape[-1])
                        close = (changed.sum(dim=-1, keepdim=True) <= 10).float()
                        far = (changed.sum(dim=-1, keepdim=True) > 25).float()
                        medium = 1.0 - close - far
                        pair_feat = torch.cat(
                            [delta, abs_delta, changed, l0, close, medium, far], dim=-1)

                        jepa_hat = model.jepa_predict(repr_a, pair_feat)
                        jepa_loss = (1.0 - F.cosine_similarity(
                            jepa_hat, repr_b.detach(), dim=-1)).mean()
                        jepa_var = F.relu(0.1 - repr_a.std(dim=0)).mean()
                        loss = loss + jepa_lambda * jepa_loss + 0.01 * jepa_var
                        if not hasattr(_ood_weight, '_jepa_cnt'):
                            _ood_weight._jepa_loss = 0.0
                            _ood_weight._jepa_var = 0.0
                            _ood_weight._jepa_cnt = 0
                        _ood_weight._jepa_loss += float(jepa_loss.detach())
                        _ood_weight._jepa_var += float(jepa_var.detach())
                        _ood_weight._jepa_cnt += 1

                # HW smoothing: large perturbation → pred should change smoothly
                hw_lip_default = 0.0 if direct_delta_enabled else 0.3
                hw_lip_eps = env_float("HW_LIP_EPS", 0.0)
                hw_lip_lambda = env_float("HW_LIP_LAMBDA", hw_lip_default)
                if hw_lip_lambda > 0:
                    hw_noisy = store.hw[h1] + torch.randn_like(store.hw[h1]) * hw_lip_eps
                    lip_output, _ = run_absolute_on_indices(
                        model, store, h1, hw_override=hw_noisy, no_rfp=no_rfp)
                    pred_lip = stack_outputs(lip_output, "absolute")
                    pred_lip_l = pred_lip.index_select(1, loss_metric_idx)
                    lip_loss = F.smooth_l1_loss(
                        pred_lip_l, pred_abs_a_l.detach(), reduction="mean"
                    )
                    loss = loss + hw_lip_lambda * lip_loss

                # KD loss: for each training interval, randomly sample a held-out
                # HW config (unseen during HW-split training). Student and teacher
                # both predict on (interval, held-out HW). smooth_l1(student, teacher)
                # encourages student to mimic teacher's HW→performance mapping.
                if teacher is not None:
                    ids = torch.randint(len(kd_hw_list), (batch_size,), device=device)
                    hw_test = torch.stack([kd_hw[kd_hw_list[i.item()]] for i in ids])
                    student_out, _ = run_absolute_on_indices(
                        model, store, h1, hw_override=hw_test, no_rfp=no_rfp)
                    student_pred = stack_outputs(student_out, "absolute")
                    with torch.no_grad():
                        teacher_out, _ = run_absolute_on_indices(
                            teacher, store, h1, hw_override=hw_test, no_rfp=False)
                        teacher_pred = stack_outputs(teacher_out, "absolute")
                    student_pred_l = student_pred.index_select(1, loss_metric_idx)
                    teacher_pred_l = teacher_pred.index_select(1, loss_metric_idx)
                    kd_loss = F.smooth_l1_loss(
                        student_pred_l, teacher_pred_l, reduction="mean"
                    )
                    loss = loss + kd_lambda * kd_loss

                # HOTL-inspired auxiliary loss using normalized footprint slopes.
                # This is a statistical proxy, not a physical miss-rate estimate;
                # see _compute_hotl_loss for normalization constraints.
                if hotl_lambda > 0:
                    _hotl_loss = _compute_hotl_loss(
                        store, h1, h2, pred_abs_a, pred_abs_b, hotl_scale)
                    loss = loss + hotl_lambda * _hotl_loss

                # L2a: cache-only sensitivity ranking loss
                if l2a_rank_lambda > 0:
                    r_h1, r_h2, rank_group_sizes = train_pair.sample_rank_batch(
                        l2a_rank_pairs, l2a_rank_k, hard_ratio=rank_hard_ratio)
                    r_out, r_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, r_h1, r_h2, no_rfp=no_rfp)
                    r_pred_sens = stack_outputs(r_out, "sensitivity")
                    r_sens_true = r_tgt_delta.abs()
                    rank_losses = []
                    offset = 0
                    for group_size in rank_group_sizes:
                        next_offset = offset + group_size
                        st = r_sens_true[offset:next_offset][:, cache_metric_idx]
                        sp = r_pred_sens[offset:next_offset][:, cache_metric_idx]
                        group_loss = correlation_ranking_loss(st, sp)
                        if group_loss is not None:
                            if ood_aug_lambda > 0:
                                ood_w = _ood_weight(
                                    store.hw[r_h1[offset:offset + 1]],
                                    store.hw[r_h2[offset:offset + 1]],
                                )
                                rank_losses.append((ood_w * group_loss, ood_w))
                            else:
                                rank_losses.append(group_loss)
                        offset = next_offset
                    if rank_losses:
                        if ood_aug_lambda > 0:
                            weighted_sum = torch.stack([item[0] for item in rank_losses]).sum()
                            weight_sum = torch.stack([item[1] for item in rank_losses]).sum()
                            l2a_loss = weighted_sum / (weight_sum + 1e-8)
                        else:
                            l2a_loss = torch.stack(rank_losses).mean()
                        loss = loss + l2a_rank_lambda * l2a_loss

                # C3a: cache-only score head trained on same-HW-pair rank batches
                if c3a_lambda > 0:
                    c3_h1, c3_h2, c3_group_sizes = train_pair.sample_rank_batch(
                        c3a_rank_pairs, c3a_rank_k, hard_ratio=rank_hard_ratio
                    )
                    c3_out, c3_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, c3_h1, c3_h2, no_rfp=no_rfp
                    )
                    score_dict = c3_out.get("score", {})
                    c3_losses = []
                    offset = 0
                    for group_size in c3_group_sizes:
                        next_offset = offset + group_size
                        st = c3_tgt_delta.abs()[offset:next_offset][:, cache_metric_idx]
                        for mi, name in enumerate(cache_metric_keys):
                            if name not in score_dict:
                                continue
                            s_true = st[:, mi]
                            true_centered = s_true - s_true.mean()
                            true_var = torch.mean(true_centered.square())
                            if true_var <= 1e-8:
                                continue
                            true_std = torch.sqrt(true_var + 1e-8)
                            s_pred = score_dict[name][offset:next_offset].squeeze(-1)
                            pred_centered = s_pred - s_pred.mean()
                            pred_std = torch.sqrt(torch.mean(pred_centered.square()) + 1e-8)
                            corr = torch.mean(true_centered * pred_centered) / (true_std * pred_std + 1e-8)
                            c3_losses.append(1.0 - corr)
                        offset = next_offset
                    if c3_losses:
                        loss = loss + c3a_lambda * torch.stack(c3_losses).mean()

                # NC3: non-cache score head trained on same-HW-pair rank batches
                if nc3_enabled and nc3_lambda > 0:
                    nc3_h1, nc3_h2, nc3_group_sizes = train_pair.sample_rank_batch(
                        nc3_rank_pairs, nc3_rank_k, hard_ratio=rank_hard_ratio
                    )
                    nc3_out, nc3_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, nc3_h1, nc3_h2, no_rfp=no_rfp
                    )
                    nc3_score_dict = nc3_out.get("nc_score", {})
                    nc3_losses = []
                    offset = 0
                    for group_size in nc3_group_sizes:
                        next_offset = offset + group_size
                        st = nc3_tgt_delta.abs()[offset:next_offset][:, nc3_metric_idx]
                        for mi, name in enumerate(nc3_metric_keys):
                            if name not in nc3_score_dict:
                                continue
                            s_true = st[:, mi]
                            true_centered = s_true - s_true.mean()
                            true_var = torch.mean(true_centered.square())
                            if true_var <= 1e-8:
                                continue
                            true_std = torch.sqrt(true_var + 1e-8)
                            s_pred = nc3_score_dict[name][offset:next_offset].squeeze(-1)
                            pred_centered = s_pred - s_pred.mean()
                            pred_std = torch.sqrt(torch.mean(pred_centered.square()) + 1e-8)
                            corr = torch.mean(true_centered * pred_centered) / (true_std * pred_std + 1e-8)
                            nc3_losses.append(1.0 - corr)
                        offset = next_offset
                    if nc3_losses:
                        loss = loss + nc3_lambda * torch.stack(nc3_losses).mean()

                # J1a: branch-shared score head trained on aggregated non-cache sensitivity
                if j1a_enabled and j1a_lambda > 0:
                    j1a_h1, j1a_h2, j1a_group_sizes = train_pair.sample_rank_batch(
                        j1a_rank_pairs, j1a_rank_k, hard_ratio=rank_hard_ratio
                    )
                    j1a_out, j1a_tgt_delta, _, _ = run_model_on_indices(
                        _model_uncompiled, store, j1a_h1, j1a_h2, no_rfp=no_rfp
                    )
                    j1a_score = j1a_out.get("j_score", None)
                    j1a_losses = []
                    offset = 0
                    for group_size in j1a_group_sizes:
                        next_offset = offset + group_size
                        st = j1a_tgt_delta.abs()[offset:next_offset][:, j1a_metric_idx]
                        s_true = _shared_sensitivity_target_torch(st)
                        true_centered = s_true - s_true.mean()
                        true_var = torch.mean(true_centered.square())
                        if true_var <= 1e-8 or j1a_score is None:
                            offset = next_offset
                            continue
                        true_std = torch.sqrt(true_var + 1e-8)
                        s_pred = j1a_score[offset:next_offset].squeeze(-1)
                        pred_centered = s_pred - s_pred.mean()
                        pred_std = torch.sqrt(torch.mean(pred_centered.square()) + 1e-8)
                        corr = torch.mean(true_centered * pred_centered) / (true_std * pred_std + 1e-8)
                        j1a_losses.append(1.0 - corr)
                        offset = next_offset
                    if j1a_losses:
                        loss = loss + j1a_lambda * torch.stack(j1a_losses).mean()

                if h3a_enabled and "_h3_residual" in output:
                    h3_reg_terms = []
                    residual_abs = []
                    correction_abs = []
                    base_abs = []
                    for name in cache_metric_keys:
                        if name in output["_h3_residual"]:
                            res = output["_h3_residual"][name]
                            corr = output["_h3_correction"][name]
                            base = output["_h3_base_delta"][name]
                            h3_reg_terms.append(res.square().mean())
                            h3_reg_terms.append(corr.square().mean())
                            residual_abs.append(res.abs().mean())
                            correction_abs.append(corr.abs().mean())
                            base_abs.append(base.abs().mean())
                    if h3_reg_terms and h3a_reg_lambda > 0:
                        loss = loss + h3a_reg_lambda * torch.stack(h3_reg_terms).mean()
                    if residual_abs and correction_abs and base_abs:
                        h3_residual_abs_total += float(torch.stack(residual_abs).mean().detach())
                        h3_correction_abs_total += float(torch.stack(correction_abs).mean().detach())
                        h3_base_abs_total += float(torch.stack(base_abs).mean().detach())
                        h3_stats_steps += 1
                if a1_enabled and "_a1_baseline" in output:
                    a1_step = _accumulate_a1_stats(output, cache_metric_keys, a1_top_contrib_sums, mode="pair")
                    if a1_step is not None:
                        b, a, r, c, bs = a1_step
                        a1_baseline_abs_total += b
                        a1_add_abs_total += a
                        a1_add_ratio_total += r
                        a1_coeff_abs_total += c
                        a1_basis_abs_total += bs
                        a1_stats_steps += 1
                if h6a_enabled:
                    h6a_step = _accumulate_h6_stats(output)
                    if h6a_step is not None:
                        feat_abs, raw_mean, raw_min, raw_max, sat0, sat1 = h6a_step
                        h6a_feat_abs_total += feat_abs
                        h6a_raw_sum += raw_mean
                        if h6a_raw_min > raw_min:
                            h6a_raw_min = raw_min
                        if h6a_raw_max < raw_max:
                            h6a_raw_max = raw_max
                        h6a_sat0_sum += sat0
                        h6a_sat1_sum += sat1
                        h6a_stats_steps += 1
                if cachex_enabled:
                    cachex_step = _accumulate_cachex_stats(output)
                    if cachex_step is not None:
                        cachex_feat_abs_total += cachex_step
                        cachex_stats_steps += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.detach())
                n_steps += 1
            else:
                abs_rows = torch.randint(train_idx_tensor.numel(), (batch_size,),
                                         device=device)
                abs_idx = train_idx_tensor[abs_rows]
                output, target = run_absolute_on_indices(model, store, abs_idx, no_rfp=no_rfp)
                pred_abs = stack_outputs(output, "absolute")
                pred_abs_l = pred_abs.index_select(1, loss_metric_idx)
                target_l = target.index_select(1, loss_metric_idx)
                abs_loss = F.smooth_l1_loss(pred_abs_l, target_l, reduction="mean")

                # Lipschitz: small hw perturbation → small prediction change
                lip_eps = env_float("LIP_EPS", 0.02)
                lip_lambda = env_float("LIP_LAMBDA", 0.5)
                hw_orig = store.hw[abs_idx]
                hw_noisy = hw_orig + torch.randn_like(hw_orig) * lip_eps
                lip_output, _ = run_absolute_on_indices(
                    model, store, abs_idx, hw_override=hw_noisy, no_rfp=no_rfp,
                )
                pred_lip = stack_outputs(lip_output, "absolute")
                pred_lip_l = pred_lip.index_select(1, loss_metric_idx)
                lip_loss = F.smooth_l1_loss(pred_lip_l, pred_abs_l.detach(), reduction="mean")

                loss = abs_loss + lip_lambda * lip_loss

                if a1_enabled and "_a1_baseline" in output:
                    a1_step = _accumulate_a1_stats(output, cache_metric_keys, a1_top_contrib_sums, mode="abs")
                    if a1_step is not None:
                        b, a, r, c, bs = a1_step
                        a1_baseline_abs_total += b
                        a1_add_abs_total += a
                        a1_add_ratio_total += r
                        a1_coeff_abs_total += c
                        a1_basis_abs_total += bs
                        a1_stats_steps += 1
                if h6a_enabled:
                    h6a_step = _accumulate_h6_stats(output)
                    if h6a_step is not None:
                        feat_abs, raw_mean, raw_min, raw_max, sat0, sat1 = h6a_step
                        h6a_feat_abs_total += feat_abs
                        h6a_raw_sum += raw_mean
                        if h6a_raw_min > raw_min:
                            h6a_raw_min = raw_min
                        if h6a_raw_max < raw_max:
                            h6a_raw_max = raw_max
                        h6a_sat0_sum += sat0
                        h6a_sat1_sum += sat1
                        h6a_stats_steps += 1
                if cachex_enabled:
                    cachex_step = _accumulate_cachex_stats(output)
                    if cachex_step is not None:
                        cachex_feat_abs_total += cachex_step
                        cachex_stats_steps += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.detach())
                n_steps += 1
            if epoch == 0 and n_steps in (1, 50, 500):
                print(f"    ep0 step {n_steps}: loss={float(loss.detach()):.4f}")
            if train_max_steps > 0 and n_steps >= train_max_steps:
                break

        scheduler.step()
        model.eval()

        abs_sp = evaluate_absolute(model, store, val_abs_idx, batch_size, no_rfp=no_rfp)
        delta_bundle = evaluate_delta_bundle(
            model,
            store,
            val_h1,
            val_h2,
            batch_size,
            no_rfp=no_rfp,
            topk_frac=eval_topk_frac,
            near_zero_pct=near_zero_pct,
        )
        delta_sp = delta_bundle["delta_spearman"]
        sens_sp = delta_bundle["sensitivity_spearman"]
        topk_recall_metrics = delta_bundle["topk_recall"]
        ndcg_metrics = delta_bundle["ndcg"]
        direction_acc_metrics = delta_bundle["direction_accuracy"]
        near_zero_dir_metrics = delta_bundle["near_zero_dir_rate"]
        pdg_sp = delta_bundle["pdg_spearman"]

        mean_abs = float(np.mean([abs_sp[k] for k in METRIC_KEYS]))
        mean_delta = float(np.mean([delta_sp[k] for k in METRIC_KEYS]))
        mean_sens = float(np.mean([sens_sp[k] for k in METRIC_KEYS]))
        min_sens = float(np.min([sens_sp[k] for k in METRIC_KEYS]))

        # Aggregate metrics over cache tasks.
        cache_abs = float(np.mean([abs_sp[k] for k in cache_metric_keys]))
        cache_sens = float(np.mean([sens_sp[k] for k in cache_metric_keys]))
        cache_min_sens = float(np.min([sens_sp[k] for k in cache_metric_keys]))
        cache_topk = float(np.mean([topk_recall_metrics[k] for k in cache_metric_keys]))
        cache_ndcg = float(np.mean([ndcg_metrics[k] for k in cache_metric_keys]))
        cache_near_zero = float(np.mean([near_zero_dir_metrics[k] for k in cache_metric_keys]))
        cache_dir_acc = float(np.mean([direction_acc_metrics[k] for k in cache_metric_keys]))

        # Non-cache aggregate for per-head fine-tuning and grouped ranking.
        noncache_metric_keys = tuple(k for k in METRIC_KEYS if k not in FiLMUnifiedModel.CACHE_METRICS)
        noncache_mean_sens = float(np.mean([sens_sp[k] for k in noncache_metric_keys]))
        _ep_jepa_loss = (0.0 if not jepa_lambda > 0 or not hasattr(_ood_weight, '_jepa_cnt')
                         or _ood_weight._jepa_cnt == 0
                         else _ood_weight._jepa_loss / _ood_weight._jepa_cnt)

        # Non-cache (NC3) eval metrics
        if nc3_enabled and len(nc3_metric_keys) > 0:
            nc3_topk = float(np.mean([topk_recall_metrics[k] for k in nc3_metric_keys]))
            nc3_ndcg = float(np.mean([ndcg_metrics[k] for k in nc3_metric_keys]))
            nc3_min_sens_val = float(np.min([sens_sp[k] for k in nc3_metric_keys]))
            nc3_sens = float(np.mean([sens_sp[k] for k in nc3_metric_keys]))
        else:
            nc3_topk = 0.0
            nc3_ndcg = 0.0
            nc3_min_sens_val = 0.0
            nc3_sens = 0.0
        nc3_rank_score = 0.5 * (nc3_topk + nc3_ndcg)
        if j1a_enabled and len(j1a_metric_keys) > 0:
            j1a_bundle = evaluate_shared_rank_bundle(
                model,
                store,
                val_h1,
                val_h2,
                batch_size,
                target_metric_keys=j1a_metric_keys,
                score_key="j_score",
                no_rfp=no_rfp,
                topk_frac=eval_topk_frac,
            )
            j1a_topk = j1a_bundle["topk"]
            j1a_ndcg = j1a_bundle["ndcg"]
            j1a_spearman = j1a_bundle["spearman"]
        else:
            j1a_bundle = {"spearman": 0.0, "topk": 0.0, "ndcg": 0.0}
            j1a_topk = 0.0
            j1a_ndcg = 0.0
            j1a_spearman = 0.0
        j1a_rank_score = 0.5 * (j1a_topk + j1a_ndcg)
        best_score_mode = score_mode
        if c3a_lambda > 0 and not r1_enabled and not r1g_enabled and not pdg_enabled and best_score_mode not in ("cache_rank", "cache_ndcg", "cache_topk"):
            best_score_mode = "cache_rank"
        if nc3_enabled and nc3_lambda > 0 and best_score_mode not in ("nc3_rank", "nc3_topk", "nc3_ndcg", "nc3_min_sens"):
            best_score_mode = "nc3_rank"
        if j1a_enabled and j1a_lambda > 0 and best_score_mode not in ("j1a_rank", "j1a_topk", "j1a_ndcg", "j1a_spearman"):
            best_score_mode = "j1a_rank"

        if best_score_mode == "mean_abs":
            score = mean_abs
        elif best_score_mode == "single_metric":
            if r1_enabled and len(r1_train_heads) == 1:
                sm_name = r1_train_heads[0]
            elif pdg_enabled and len(pdg_train_heads) == 1:
                sm_name = pdg_train_heads[0]
            elif gmr_finetune and len(gmr_train_heads) == 1:
                sm_name = gmr_train_heads[0]
            else:
                raise ValueError(
                    "SCORE_MODE=single_metric requires (R1_ENABLED=1 or PDG_ENABLED=1"
                    " or GMR_FINETUNE=1) with exactly 1 train head")
            score = float(sens_sp[sm_name])
        elif best_score_mode == "single_metric_pdg":
            if not (pdg_enabled and len(pdg_train_heads) == 1):
                raise ValueError(
                    "SCORE_MODE=single_metric_pdg requires PDG_ENABLED=1"
                    " with exactly 1 PDG_TRAIN_HEADS")
            sm_name = pdg_train_heads[0]
            score = float(pdg_sp[sm_name])
        elif best_score_mode == "noncache_mean":
            score = noncache_mean_sens
        elif best_score_mode == "cache_guarded_noncache_mean":
            score = noncache_mean_sens if cache_min_sens >= cache_guard else -1e9
        elif best_score_mode == "cache_guarded_mean_sens":
            score = mean_sens if cache_min_sens >= cache_guard else -1e9
        elif best_score_mode == "mean_sens":
            score = mean_sens
        elif best_score_mode == "min_sens":
            score = min_sens
        elif best_score_mode == "cache_min_sens":
            score = cache_min_sens
        elif best_score_mode == "cache_topk":
            score = cache_topk
        elif best_score_mode == "cache_ndcg":
            score = cache_ndcg
        elif best_score_mode == "cache_rank":
            score = 0.5 * (cache_topk + cache_ndcg)
        elif best_score_mode == "nc3_rank":
            score = 0.5 * (nc3_topk + nc3_ndcg)
        elif best_score_mode == "nc3_topk":
            score = nc3_topk
        elif best_score_mode == "nc3_ndcg":
            score = nc3_ndcg
        elif best_score_mode == "nc3_min_sens":
            score = nc3_min_sens_val
        elif best_score_mode == "j1a_rank":
            score = j1a_rank_score
        elif best_score_mode == "j1a_topk":
            score = j1a_topk
        elif best_score_mode == "j1a_ndcg":
            score = j1a_ndcg
        elif best_score_mode == "j1a_spearman":
            score = j1a_spearman
        else:
            if direct_delta_enabled:
                raise ValueError(
                    f"Unsupported SCORE_MODE={best_score_mode!r} for TRAIN_MODE=dse_direct. "
                    "Use a sensitivity-based score mode such as mean_sens, min_sens, "
                    "cache_min_sens, cache_topk, cache_ndcg, cache_rank, noncache_mean, "
                    "or the active ranking-head score modes."
                )
            score = mean_abs

        dt = time.time() - epoch_start
        denom = max(1, n_steps)
        stage = "dse" if train_mode in ("dse", "dse_direct") else "abs_only"
        j1a_main = ""
        if j1a_enabled and j1a_lambda > 0 and len(j1a_metric_keys) > 0:
            j1a_main = (
                f" | j1_topk@{eval_topk_frac:.0%}={j1a_topk:.3f}"
                f" | j1_ndcg@{eval_topk_frac:.0%}={j1a_ndcg:.3f}"
                f" | j1_sp={j1a_spearman:.3f}"
            )
        abs_main = "" if direct_delta_enabled else f"abs_rho mean={mean_abs:.3f} | "
        print(
            f"Ep {epoch:3d} [{stage}] | {dt:.1f}s | loss={total_loss/denom:.4f} | "
            f"{abs_main}"
            f"delta_rho mean={mean_delta:.3f} | "
            f"sens_rho mean={mean_sens:.3f} min={min_sens:.3f} nc_mean={noncache_mean_sens:.3f}"
            + (f" {r1_train_heads[0]}={sens_sp[r1_train_heads[0]]:.3f}" if r1_enabled and len(r1_train_heads) == 1 else "")
            + (f" pdg_{pdg_train_heads[0]}={pdg_sp[pdg_train_heads[0]]:.3f}" if pdg_enabled and len(pdg_train_heads) == 1 else "")
            + (f" jepa={_ep_jepa_loss:.4f}" if jepa_lambda > 0 else "")
            + " | "
            f"cache_sens mean={cache_sens:.3f} min={cache_min_sens:.3f} | "
            f"topk@{eval_topk_frac:.0%} cache={cache_topk:.3f} | "
            f"ndcg@{eval_topk_frac:.0%} cache={cache_ndcg:.3f} | "
            f"near0 cache={cache_near_zero:.3f} | "
            f"dir cache={cache_dir_acc:.3f}"
            f"{j1a_main}"
        )
        if nc3_enabled and nc3_lambda > 0 and len(nc3_metric_keys) > 0:
            print(
                f"       nc3_rank={nc3_rank_score:.4f} nc3_min_sens={nc3_min_sens_val:.4f} "
                f"nc3_topk={nc3_topk:.4f} nc3_ndcg={nc3_ndcg:.4f} "
                f"targets=({','.join(nc3_metric_keys)})"
            )
        if j1a_enabled and j1a_lambda > 0 and len(j1a_metric_keys) > 0:
            print(
                f"       j1a_rank={j1a_rank_score:.4f} j1a_spearman={j1a_spearman:.4f} "
                f"j1a_topk={j1a_topk:.4f} j1a_ndcg={j1a_ndcg:.4f} "
                f"targets=({','.join(j1a_metric_keys)})"
            )
        if h3a_enabled and h3_stats_steps > 0:
            mean_h3_resid = h3_residual_abs_total / h3_stats_steps
            mean_h3_corr = h3_correction_abs_total / h3_stats_steps
            mean_h3_base = h3_base_abs_total / h3_stats_steps
            corr_ratio = mean_h3_corr / max(mean_h3_base, 1e-8)
            print(
                f"       H3 |resid|={mean_h3_resid:.5f} "
                f"|corr|={mean_h3_corr:.5f} |base_delta|={mean_h3_base:.5f} "
                f"corr/base={corr_ratio:.3f}"
            )
        if a1_enabled and a1_stats_steps > 0:
            mean_a1_base = a1_baseline_abs_total / a1_stats_steps
            mean_a1_add = a1_add_abs_total / a1_stats_steps
            mean_a1_ratio = a1_add_ratio_total / a1_stats_steps
            mean_a1_coeff = a1_coeff_abs_total / a1_stats_steps
            mean_a1_basis = a1_basis_abs_total / a1_stats_steps
            a1_top_summary = {}
            top_parts = []
            for name in cache_metric_keys:
                contrib_sums = a1_top_contrib_sums.get(name)
                knob_names = a1_knob_meta.get(name) if a1_knob_meta else None
                if contrib_sums is None or knob_names is None or len(knob_names) == 0:
                    continue
                top_idx = int(np.argmax(contrib_sums))
                top_val = float(contrib_sums[top_idx] / a1_stats_steps)
                top_knob = knob_names[top_idx]
                a1_top_summary[name] = {"knob": top_knob, "value": top_val}
                top_parts.append(f"{name}:{top_knob}({top_val:.5f})")
            print(
                f"       A1 |base|={mean_a1_base:.5f} |add|={mean_a1_add:.5f} "
                f"add/pred={mean_a1_ratio:.3f} |coeff|={mean_a1_coeff:.5f} "
                f"|basis|={mean_a1_basis:.5f}"
            )
            if top_parts:
                print("       A1 top=" + ", ".join(top_parts))
        if h6a_enabled and h6a_stats_steps > 0:
            mean_h6a_feat = h6a_feat_abs_total / h6a_stats_steps
            mean_h6a_raw = h6a_raw_sum / h6a_stats_steps
            mean_sat0_vec = h6a_sat0_sum / h6a_stats_steps
            mean_sat1_vec = h6a_sat1_sum / h6a_stats_steps
            mean_sat0 = float(mean_sat0_vec.mean())
            mean_sat1 = float(mean_sat1_vec.mean())
            h6_pairs = ", ".join(
                f"{name}={mean_h6a_raw[i]:.4f}" for i, name in enumerate(h6_raw_feature_names)
            )
            top_sat0_idx = int(np.argmax(mean_sat0_vec))
            top_sat1_idx = int(np.argmax(mean_sat1_vec))
            print(f"       H6[{h6_mode}] |feat|={mean_h6a_feat:.5f} "
                  f"min={h6a_raw_min:.4f} max={h6a_raw_max:.4f} "
                  f"sat0={mean_sat0:.3f} sat1={mean_sat1:.3f} "
                  f"sat0_max={h6_raw_feature_names[top_sat0_idx]}({mean_sat0_vec[top_sat0_idx]:.3f}) "
                  f"sat1_max={h6_raw_feature_names[top_sat1_idx]}({mean_sat1_vec[top_sat1_idx]:.3f})")
            print(f"       H6[{h6_mode}] {h6_pairs}")
        if cachex_enabled and cachex_stats_steps > 0:
            mean_cachex = cachex_feat_abs_total / cachex_stats_steps
            print(f"       CacheX |feat|={mean_cachex:.5f}")

        history["train_loss"].append(total_loss / denom)
        if jepa_lambda > 0 and hasattr(_ood_weight, '_jepa_cnt') and _ood_weight._jepa_cnt > 0:
            history.setdefault("train_jepa_loss", []).append(
                _ood_weight._jepa_loss / _ood_weight._jepa_cnt)
            history.setdefault("train_jepa_var", []).append(
                _ood_weight._jepa_var / _ood_weight._jepa_cnt)
            _ood_weight._jepa_loss = 0.0
            _ood_weight._jepa_var = 0.0
            _ood_weight._jepa_cnt = 0
        history["val_abs_spearman"].append(abs_sp)
        history["val_delta_spearman"].append(delta_sp)
        history["val_sensitivity_spearman"].append(sens_sp)
        history["val_topk_recall"].append(topk_recall_metrics)
        history["val_ndcg"].append(ndcg_metrics)
        history["val_direction_accuracy"].append(direction_acc_metrics)
        history["val_near_zero_dir_rate"].append(near_zero_dir_metrics)
        history["val_pdg_spearman"].append(pdg_sp)
        if j1a_enabled:
            history.setdefault("val_j1a", []).append(j1a_bundle)
        if a1_enabled:
            if a1_stats_steps > 0:
                mean_a1_base = a1_baseline_abs_total / a1_stats_steps
                mean_a1_add = a1_add_abs_total / a1_stats_steps
                mean_a1_ratio = a1_add_ratio_total / a1_stats_steps
                mean_a1_coeff = a1_coeff_abs_total / a1_stats_steps
                mean_a1_basis = a1_basis_abs_total / a1_stats_steps
                a1_top_summary = {}
                for name in cache_metric_keys:
                    contrib_sums = a1_top_contrib_sums.get(name)
                    knob_names = a1_knob_meta.get(name) if a1_knob_meta else None
                    if contrib_sums is None or knob_names is None or len(knob_names) == 0:
                        continue
                    top_idx = int(np.argmax(contrib_sums))
                    a1_top_summary[name] = {
                        "knob": knob_names[top_idx],
                        "value": float(contrib_sums[top_idx] / a1_stats_steps),
                    }
            else:
                mean_a1_base = 0.0
                mean_a1_add = 0.0
                mean_a1_ratio = 0.0
                mean_a1_coeff = 0.0
                mean_a1_basis = 0.0
                a1_top_summary = {}
            history["train_a1_baseline_abs"].append(mean_a1_base)
            history["train_a1_add_abs"].append(mean_a1_add)
            history["train_a1_add_ratio"].append(mean_a1_ratio)
            history["train_a1_coeff_abs_mean"].append(mean_a1_coeff)
            history["train_a1_basis_abs_mean"].append(mean_a1_basis)
            history["train_a1_top_knob_contrib"].append(a1_top_summary)
        if h6a_enabled:
            if h6a_stats_steps > 0:
                mean_h6a_feat = h6a_feat_abs_total / h6a_stats_steps
                mean_h6a_raw = h6a_raw_sum / h6a_stats_steps
                mean_sat0_vec = h6a_sat0_sum / h6a_stats_steps
                mean_sat1_vec = h6a_sat1_sum / h6a_stats_steps
                mean_sat0 = float(mean_sat0_vec.mean())
                mean_sat1 = float(mean_sat1_vec.mean())
                h6a_summary = {
                    name: float(mean_h6a_raw[i])
                    for i, name in enumerate(h6_raw_feature_names)
                }
                h6a_sat0_summary = {
                    name: float(mean_sat0_vec[i])
                    for i, name in enumerate(h6_raw_feature_names)
                }
                h6a_sat1_summary = {
                    name: float(mean_sat1_vec[i])
                    for i, name in enumerate(h6_raw_feature_names)
                }
            else:
                mean_h6a_feat = 0.0
                mean_sat0 = 0.0
                mean_sat1 = 0.0
                h6a_summary = {name: 0.0 for name in h6_raw_feature_names}
                h6a_sat0_summary = {name: 0.0 for name in h6_raw_feature_names}
                h6a_sat1_summary = {name: 0.0 for name in h6_raw_feature_names}
            history["train_h6_feat_abs_mean"].append(mean_h6a_feat)
            history["train_h6_raw_summary"].append(h6a_summary)
            history["train_h6_sat0_rate"].append(round(mean_sat0, 6))
            history["train_h6_sat1_rate"].append(round(mean_sat1, 6))
            history["train_h6_sat0_per_feature"].append(h6a_sat0_summary)
            history["train_h6_sat1_per_feature"].append(h6a_sat1_summary)
        if cachex_enabled:
            mean_cachex = cachex_feat_abs_total / cachex_stats_steps if cachex_stats_steps > 0 else 0.0
            history["train_cachex_feat_abs_mean"].append(mean_cachex)
        if h3a_enabled:
            if h3_stats_steps > 0:
                mean_h3_resid = h3_residual_abs_total / h3_stats_steps
                mean_h3_corr = h3_correction_abs_total / h3_stats_steps
                mean_h3_base = h3_base_abs_total / h3_stats_steps
                corr_ratio = mean_h3_corr / max(mean_h3_base, 1e-8)
            else:
                mean_h3_resid = 0.0
                mean_h3_corr = 0.0
                mean_h3_base = 0.0
                corr_ratio = 0.0
            history["train_h3_residual_abs"].append(mean_h3_resid)
            history["train_h3_correction_abs"].append(mean_h3_corr)
            history["train_h3_base_delta_abs"].append(mean_h3_base)
            history["train_h3_correction_ratio"].append(corr_ratio)

        # Save history after every epoch so data survives unexpected termination
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        cache_rank_score = 0.5 * (cache_topk + cache_ndcg)
        if c3a_lambda > 0:
            if cache_rank_score > best_cache_rank:
                best_cache_rank = cache_rank_score
                _save_model(model, best_cache_rank_path)
                print(f"       -> best@{epoch} cache_rank={cache_rank_score:.4f}, saved {best_cache_rank_path}")
            if cache_min_sens > best_cache_min_sens:
                best_cache_min_sens = cache_min_sens
                _save_model(model, best_cache_min_path)
                print(f"       -> best@{epoch} cache_min_sens={cache_min_sens:.4f}, saved {best_cache_min_path}")
        if nc3_enabled and nc3_lambda > 0:
            if nc3_rank_score > best_nc3_rank:
                best_nc3_rank = nc3_rank_score
                _save_model(model, best_nc3_rank_path)
                print(f"       -> best@{epoch} nc3_rank={nc3_rank_score:.4f}, saved {best_nc3_rank_path}")
            if nc3_min_sens_val > best_nc3_min_sens:
                best_nc3_min_sens = nc3_min_sens_val
                _save_model(model, best_nc3_min_sens_path)
                print(f"       -> best@{epoch} nc3_min_sens={nc3_min_sens_val:.4f}, saved {best_nc3_min_sens_path}")
        if j1a_enabled and j1a_lambda > 0:
            if j1a_rank_score > best_j1a_rank:
                best_j1a_rank = j1a_rank_score
                _save_model(model, best_j1a_rank_path)
                print(f"       -> best@{epoch} j1a_rank={j1a_rank_score:.4f}, saved {best_j1a_rank_path}")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            _save_model(model, best_model_path)
            _bm_label = best_score_mode
            if best_score_mode == "single_metric":
                if r1_enabled and len(r1_train_heads) == 1:
                    _bm_label = r1_train_heads[0]
                elif pdg_enabled and len(pdg_train_heads) == 1:
                    _bm_label = pdg_train_heads[0]
                elif gmr_finetune and len(gmr_train_heads) == 1:
                    _bm_label = gmr_train_heads[0]
            elif best_score_mode == "single_metric_pdg":
                _bm_label = f"pdg_{pdg_train_heads[0]}"
            print(f"       -> best@{epoch} {_bm_label}={score:.4f}, saved")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch} (best={best_score:.4f} @ {best_epoch})")
                break

    print(f"\nBest {best_score_mode}: {best_score:.4f} at epoch {best_epoch}")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-path", default=None, help="Path to the unified interval-level dataset pickle.")
    parser.add_argument("--hw-json", default=None, help="Path to the held-out-HW hardware-vector JSON.")
    parser.add_argument("--init-ckpt", default=None, help="Warm-start checkpoint path.")
    parser.add_argument("--teacher-ckpt", default=None, help="Teacher checkpoint path.")
    cli_args, _ = parser.parse_known_args()
    if cli_args.data_path:
        os.environ["DATA_PATH"] = cli_args.data_path
    if cli_args.hw_json:
        os.environ["HW_JSON"] = cli_args.hw_json
    if cli_args.init_ckpt:
        os.environ["INIT_CKPT"] = cli_args.init_ckpt
    if cli_args.teacher_ckpt:
        os.environ["TEACHER_CKPT"] = cli_args.teacher_ckpt
    train()
