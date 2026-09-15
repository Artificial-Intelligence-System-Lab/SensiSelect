#!/usr/bin/env python3
"""Minimal Simulation Budget for Bottleneck-Risk Discrimination.

Scenario 1 — Pairwise bottleneck discrimination
  Q: Between two held-out HWs, how many intervals are needed to decide
     which one is more risky on metric m?

Scenario 2 — Multi-candidate bottleneck-risk ranking
  Q: For a structured group of 1 base HW + 7 candidate HWs, how many
     shared intervals are needed to identify the most risky candidate?

Methods: SensiSelect | Abs. Regressor | SimPoint | Random

Usage:
    python eval_film_simulation_budget.py \
        --checkpoint PATH --output eval_simulation_budget.json \
        --batch_size 256
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import MicroArchDataset
from eval_film_common import (
    SEED,
    TASK_KEYS,
    TASK_INDEX,
    add_model_override_args,
    add_dataset_path_arg,
    add_hw_json_arg,
    add_regressor_override_mode_arg,
    collect_model_overrides,
    collect_predictions_from_indices,
    collect_pair_bundle,
    build_group_hw_map,
    compute_pair_scores,
    hw_holdout_split,
    load_film_model,
    positive_degradation,
    bottleneck_top_indices,
    set_seed,
    validate_abs_regressor_checkpoint,
    DEFAULT_ABS_REGRESSOR_CHECKPOINT,
    resolve_regressor_model_overrides,
    resolve_data_path,
)


def _resolve_hw_json_path(raw_path=None):
    candidates = []
    if raw_path:
        candidates.append(Path(raw_path))
    env_path = os.getenv("HW_JSON_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        Path(resolve_data_path("gem5_hardware_vectors_0_304.json")),
    ])
    seen = set()
    for cand in candidates:
        resolved = cand.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)
    raise FileNotFoundError(f"No HW JSON found. Searched: {candidates}")


# Mechanism categories for structured candidate set construction.
MECHANISM_CATEGORIES = {
    "cache_size": {22, 26, 30},
    "cache_assoc": {23, 27, 31},
    "cache_mshr": {24, 28, 32},
    "width": {1, 2, 3, 4, 5, 6, 7},
    "buffer": {8, 9, 10, 11, 12, 13},
    "btb_ras": {20, 21},
    "functional": {14, 15, 16, 17, 18, 19},
}

K_VALUES = [1, 2, 5, 10, 20, 50]
N_PAIRS = 400
N_HW_GROUPS = 50
HW_PER_GROUP = 8
N_SIMPOINT_SEEDS = 5
N_RANDOM_SEEDS = 5
TAIL_FRAC = 0.05
TOP_M = 3


def _tail_risk_from_delta(delta_vec, tail_frac):
    """Mean top-tail positive degradation for a 1D delta vector."""
    severity = positive_degradation(delta_vec)
    if severity.size == 0:
        return 0.0
    tail_idx = bottleneck_top_indices(severity, tail_frac)
    if tail_idx.size == 0:
        return 0.0
    return float(np.mean(severity[tail_idx]))


def _ranknorm_1d(x):
    """Rank-normalize one candidate's interval scores to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(max(1, n - 1))


def _ranknorm_cols(score_mat):
    """Column-wise rank normalization for [n_intervals, n_candidates]."""
    score_mat = np.asarray(score_mat, dtype=np.float64)
    z = np.zeros_like(score_mat, dtype=np.float64)
    for j in range(score_mat.shape[1]):
        col = score_mat[:, j]
        valid = np.isfinite(col)
        if not np.any(valid):
            continue
        z_col = np.zeros_like(col, dtype=np.float64)
        z_col[valid] = _ranknorm_1d(col[valid])
        z[:, j] = z_col
    return z


def _aggregate_shared_score(score_mat, mode):
    """Aggregate [n_intervals, n_candidates] into one shared interval score."""
    score_mat = np.asarray(score_mat, dtype=np.float64)
    if score_mat.size == 0:
        return np.array([], dtype=np.float64)
    if score_mat.ndim != 2:
        raise ValueError(f"score_mat must be 2D, got shape {score_mat.shape}")

    if mode == "mean_raw":
        return np.nanmean(score_mat, axis=1)

    z = _ranknorm_cols(score_mat)
    if mode == "mean_ranknorm":
        return np.nanmean(z, axis=1)
    if mode == "max_ranknorm":
        return np.nanmax(z, axis=1)
    if mode == "top2mean_ranknorm":
        k = min(2, z.shape[1])
        return np.mean(np.sort(z, axis=1)[:, -k:], axis=1)
    if mode == "logsumexp_ranknorm":
        tau = 0.2
        return tau * np.log(np.mean(np.exp(z / tau), axis=1))

    raise ValueError(f"Unknown multi-selector mode: {mode}")


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


def _riskier_label(score_a, score_b, eps=1e-12):
    """Return +1 if B is more risky, -1 if A is more risky, 0 if tied."""
    if score_b > score_a + eps:
        return 1
    if score_a > score_b + eps:
        return -1
    return 0


def _top1_worst_accuracy(true_scores, pred_scores):
    """Fraction where highest-risk candidate is correctly identified."""
    return float(np.mean(np.argmax(true_scores, axis=1) == np.argmax(pred_scores, axis=1)))


def _pairwise_risk_ordering_accuracy(true_scores, pred_scores):
    """Fraction of candidate pairs whose risk ordering is correct."""
    n_groups, n_hws = true_scores.shape
    total, correct = 0, 0
    for g in range(n_groups):
        for i in range(n_hws):
            for j in range(i + 1, n_hws):
                td = true_scores[g, i] - true_scores[g, j]
                pd = pred_scores[g, i] - pred_scores[g, j]
                if abs(td) <= 1e-12:
                    continue
                total += 1
                if (td > 0 and pd > 0) or (td < 0 and pd < 0):
                    correct += 1
    return correct / total if total > 0 else 0.5


def _topm_set_accuracy(true_scores, pred_scores, top_m):
    """Mean overlap accuracy between true/pred top-m worst candidate sets."""
    n_groups, n_hws = true_scores.shape
    m = max(1, min(int(top_m), n_hws))
    scores = []
    for g in range(n_groups):
        true_top = set(np.argsort(true_scores[g])[-m:])
        pred_top = set(np.argsort(pred_scores[g])[-m:])
        scores.append(len(true_top & pred_top) / m)
    return float(np.mean(scores)) if scores else 0.0


def _worst_candidate_regret(true_scores, pred_scores):
    """Normalized regret for choosing the wrong highest-risk candidate."""
    n = len(true_scores)
    true_worst = np.argmax(true_scores, axis=1)
    pred_worst = np.argmax(pred_scores, axis=1)
    worst_vals = true_scores[np.arange(n), true_worst]
    chosen_vals = true_scores[np.arange(n), pred_worst]
    rng = np.max(true_scores, axis=1) - np.min(true_scores, axis=1)
    rng = np.where(rng < 1e-8, 1.0, rng)
    return float(np.mean(np.maximum(worst_vals - chosen_vals, 0) / rng))


def _difficulty_label(gap, terciles):
    if gap <= terciles[0]:
        return "hard"
    if gap <= terciles[1]:
        return "medium"
    return "easy"


def _subset_pair_bundle(groups, h_a, h_b, target_gids=None):
    bundle_gids, items_a, items_b, _, pred_mat = collect_pair_bundle(groups, h_a, h_b)
    if not bundle_gids:
        return [], [], [], None
    if target_gids is None:
        return bundle_gids, items_a, items_b, pred_mat
    pos = {g: i for i, g in enumerate(bundle_gids)}
    idx = [pos[g] for g in target_gids if g in pos]
    gids = [bundle_gids[i] for i in idx]
    items_a = [items_a[i] for i in idx]
    items_b = [items_b[i] for i in idx]
    pred = pred_mat[idx] if pred_mat is not None and idx else None
    return gids, items_a, items_b, pred


def precompute_bbv_lookup(dataset):
    """gid -> BBV vector for SimPoint-style k-means selection."""
    interval_bbvs = {}
    for idx, sample in enumerate(dataset.samples):
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


def _build_simpoint_for_gids(bbv_lookup, gid_set, k, seed):
    """BBV k-means -> select 1 interval per cluster (K intervals total)."""
    if k >= len(gid_set):
        return set(gid_set)
    gids = sorted(gid_set)
    bbvs = {gid: bbv_lookup[gid] for gid in gids if gid in bbv_lookup}
    if not bbvs:
        return set(gids[:k])
    gids_sorted = sorted(bbvs.keys())
    mat = np.array([bbvs[g] for g in gids_sorted])
    proj_dim = min(128, mat.shape[1])
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((proj_dim, 20)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    mat20 = mat[:, :proj_dim] @ proj
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    km.fit(mat20)
    selected = set()
    for center in km.cluster_centers_:
        dist = np.linalg.norm(mat20 - center, axis=1)
        selected.add(gids_sorted[int(np.argmin(dist))])
    if len(selected) < k:
        extras = [g for g in gids_sorted if g not in selected]
        selected.update(extras[: max(0, k - len(selected))])
    return set(list(selected)[:k])


def _compute_group_score_mats(groups, model, device, gids, h_0, h_candidates, method_name):
    """Per-task candidate score matrices with shape [n_intervals, n_candidates]."""
    n_gids = len(gids)
    n_candidates = len(h_candidates)
    out = {
        task: np.full((n_gids, n_candidates), np.nan, dtype=np.float64)
        for task in TASK_KEYS
    }

    for cj, h_j in enumerate(h_candidates):
        bg, items_a, items_b, pred_mat = _subset_pair_bundle(groups, h_0, h_j, gids)
        _ensure_exact_gids(method_name, gids, bg, f"multi_hw ({h_0}, {h_j})")
        if pred_mat is None:
            raise ValueError(
                f"{method_name} bundle missing prediction matrix for multi_hw ({h_0}, {h_j})"
            )
        ps = compute_pair_scores(model, items_a, items_b, device) if model and bg else {}
        bp = {g: i for i, g in enumerate(bg)} if bg else {}

        for task in TASK_KEYS:
            t_idx = TASK_INDEX[task]
            score_arr = np.zeros(n_gids, dtype=np.float64)
            if task in ps and bp:
                src = np.asarray(ps[task], dtype=np.float64)
                for gi, g in enumerate(gids):
                    if g in bp:
                        score_arr[gi] = src[bp[g]]
            elif pred_mat is not None and bp:
                for gi, g in enumerate(gids):
                    if g in bp:
                        score_arr[gi] = abs(pred_mat[bp[g], t_idx])
            out[task][:, cj] = score_arr
    return out


def _compute_pair_task_scores(groups, model, device, gids, h_a, h_b, method_name):
    """Per-task interval scores aligned to `gids` for one HW pair."""
    bg, items_a, items_b, pred_mat = _subset_pair_bundle(groups, h_a, h_b, gids)
    _ensure_exact_gids(method_name, gids, bg, f"pairwise ({h_a}, {h_b})")
    if pred_mat is None:
        raise ValueError(
            f"{method_name} bundle missing prediction matrix for pairwise ({h_a}, {h_b})"
        )
    ps = compute_pair_scores(model, items_a, items_b, device) if model and bg else {}
    bp = {g: i for i, g in enumerate(bg)} if bg else {}
    out = {}
    for task in TASK_KEYS:
        t_idx = TASK_INDEX[task]
        score_arr = np.zeros(len(gids), dtype=np.float64)
        if task in ps and bp:
            src = np.asarray(ps[task], dtype=np.float64)
            for gi, g in enumerate(gids):
                if g in bp:
                    score_arr[gi] = src[bp[g]]
        elif pred_mat is not None and bp:
            for gi, g in enumerate(gids):
                if g in bp:
                    score_arr[gi] = abs(pred_mat[bp[g], t_idx])
        out[task] = score_arr
    return out


def _build_structured_hw_groups(
    groups,
    held_out_hw,
    hw_json_path,
    n_groups=N_HW_GROUPS,
    hw_per_group=HW_PER_GROUP,
    rng=None,
):
    """Build strict 1-base + 7-candidate structured groups.

    Groups must:
      - contain exactly `hw_per_group` HWs
      - mix multiple mechanism categories
      - avoid collapsing to trivial monotonic sweeps

    To keep the construction feasible on held-out HWs, candidates are assigned
    to the mechanism category with the largest overlap in changed parameters.
    Each group must cover at least `min_distinct_cats` categories, then any
    remaining slots are filled from already-seen structured categories.
    """
    from build_pair_suite import NUMERIC_PARAMS, SIZE_TO_KB

    hw_json_path = _resolve_hw_json_path(hw_json_path)
    with open(hw_json_path) as f:
        jdata = json.load(f)
    hw_configs = jdata if isinstance(jdata, list) else list(jdata.values())

    def _to_f(v):
        if isinstance(v, str):
            return float(SIZE_TO_KB.get(v, 0))
        return float(v)

    hw_id_to_vec = {}
    for d in hw_configs:
        hid = int(d.get("id", -1))
        if hid in held_out_hw:
            vec = np.zeros(len(NUMERIC_PARAMS), dtype=np.float32)
            for j, p in enumerate(NUMERIC_PARAMS):
                vec[j] = _to_f(d.get(p, 0))
            hw_id_to_vec[hid] = vec

    hw_ids = sorted(hw_id_to_vec.keys())
    if len(hw_ids) < hw_per_group:
        print(f"  WARNING: only {len(hw_ids)} held-out HWs, need {hw_per_group}")
        return []

    hw_groups = []
    base_order = list(hw_ids)
    if rng is not None:
        rng.shuffle(base_order)

    ordered_cats = [
        "cache_size",
        "cache_assoc",
        "width",
        "buffer",
        "btb_ras",
        "functional",
        "cache_mshr",
    ]
    min_distinct_cats = 4

    for h_0 in base_order:
        if len(hw_groups) >= n_groups:
            break
        v0 = hw_id_to_vec[h_0]
        candidates_by_cat = {cat: [] for cat in ordered_cats}
        for h_j in hw_ids:
            if h_j == h_0:
                continue
            vj = hw_id_to_vec[h_j]
            diff_indices = set(np.where(np.abs(v0 - vj) > 1e-6)[0])
            if not diff_indices:
                continue

            best_cat, best_overlap = None, 0
            for cat in ordered_cats:
                overlap = len(diff_indices & MECHANISM_CATEGORIES[cat])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cat = cat
            if best_cat and best_overlap > 0:
                candidates_by_cat[best_cat].append(h_j)

        non_empty_cats = [cat for cat in ordered_cats if candidates_by_cat[cat]]
        if len(non_empty_cats) < min_distinct_cats:
            continue

        grp = [h_0]
        used = {h_0}

        # First, guarantee multi-category coverage.
        for cat in non_empty_cats:
            if len(grp) >= hw_per_group:
                break
            cands = [c for c in candidates_by_cat[cat] if c not in used]
            if not cands:
                continue
            pick = cands[0] if rng is None else cands[int(rng.integers(len(cands)))]
            grp.append(pick)
            used.add(pick)

        if len(grp) < 1 + min_distinct_cats:
            continue

        # Then fill remaining slots from structured candidates, preferring
        # categories with fewer picks so the group stays mixed.
        while len(grp) < hw_per_group:
            progress = False
            cat_order = sorted(
                non_empty_cats,
                key=lambda cat: sum(1 for hw in grp[1:] if hw in candidates_by_cat[cat]),
            )
            for cat in cat_order:
                cands = [c for c in candidates_by_cat[cat] if c not in used]
                if not cands:
                    continue
                pick = cands[0] if rng is None else cands[int(rng.integers(len(cands)))]
                grp.append(pick)
                used.add(pick)
                progress = True
                if len(grp) >= hw_per_group:
                    break
            if not progress:
                break

        if len(grp) != hw_per_group:
            continue

        common = None
        for hw in grp:
            hw_gids = set(g for g, hw_map in groups.items() if hw in hw_map)
            common = hw_gids if common is None else (common & hw_gids)
        if common and len(common) >= 20:
            hw_groups.append((grp, sorted(common)))

    return hw_groups


def evaluate_pairwise(
    groups_by_method,
    models_by_method,
    device,
    bbv_lookup,
    held_out_hw,
    n_pairs=N_PAIRS,
    tail_frac=TAIL_FRAC,
    max_intervals_per_pair=None,
):
    """Pairwise bottleneck discrimination with difficulty stratification."""
    ref_groups = groups_by_method["model"]
    learned_methods = list(models_by_method.keys())
    hw_ids = sorted(held_out_hw)
    rng = np.random.default_rng(SEED + 400)

    hw_pairs = []
    for _ in range(n_pairs * 3):
        a, b = int(rng.choice(hw_ids)), int(rng.choice(hw_ids))
        if a != b and (a, b) not in hw_pairs and (b, a) not in hw_pairs:
            hw_pairs.append((a, b))
        if len(hw_pairs) >= n_pairs:
            break

    pair_records = []

    for pi, (h_a, h_b) in enumerate(hw_pairs):
        if pi % 50 == 0:
            print(f"  pair {pi}/{len(hw_pairs)}...")

        full_bundle_gids, _, _, full_pred = _subset_pair_bundle(ref_groups, h_a, h_b, None)
        if not full_bundle_gids:
            continue

        if max_intervals_per_pair and len(full_bundle_gids) > max_intervals_per_pair:
            work_idx = rng.choice(len(full_bundle_gids), max_intervals_per_pair, replace=False)
            work_gids = [full_bundle_gids[i] for i in work_idx]
        else:
            work_gids = full_bundle_gids

        method_scores = {
            method_name: _compute_pair_task_scores(
                groups_by_method[method_name],
                models_by_method[method_name],
                device,
                work_gids,
                h_a,
                h_b,
                method_name,
            )
            for method_name in learned_methods
        }
        n_intervals = len(work_gids)

        sp_selections = {}
        rand_selections = {}
        for k in K_VALUES:
            if k > n_intervals:
                continue
            sp_selections[k] = [
                _build_simpoint_for_gids(bbv_lookup, set(work_gids), k, SEED + 2000 + si)
                for si in range(N_SIMPOINT_SEEDS)
            ]
            rand_selections[k] = [
                rng.choice(n_intervals, k, replace=False) for _ in range(N_RANDOM_SEEDS)
            ]

        record = {}
        for task in TASK_KEYS:
            t_idx = TASK_INDEX[task]
            full_delta = np.array(
                [
                    ref_groups[g][h_b]["label"][t_idx] - ref_groups[g][h_a]["label"][t_idx]
                    for g in full_bundle_gids
                ],
                dtype=np.float64,
            )

            true_risk_b = _tail_risk_from_delta(full_delta, tail_frac)
            true_risk_a = _tail_risk_from_delta(-full_delta, tail_frac)
            true_gap = abs(true_risk_b - true_risk_a)
            true_riskier = _riskier_label(true_risk_a, true_risk_b)
            if true_riskier == 0 or true_gap <= 1e-12:
                continue

            work_delta = np.array(
                [
                    ref_groups[g][h_b]["label"][t_idx] - ref_groups[g][h_a]["label"][t_idx]
                    for g in work_gids
                ],
                dtype=np.float64,
            )

            per_k = {}
            for k in K_VALUES:
                if k > n_intervals:
                    continue

                per_k[k] = {}
                for method_name in learned_methods:
                    s_method = np.asarray(method_scores[method_name][task], dtype=np.float64)
                    im = np.argsort(s_method)[::-1][:k]
                    est_b = _tail_risk_from_delta(work_delta[im], tail_frac)
                    est_a = _tail_risk_from_delta(-work_delta[im], tail_frac)
                    pred_riskier = _riskier_label(est_a, est_b)
                    per_k[k][method_name] = 1.0 if pred_riskier == true_riskier else 0.0

                sp_accs = []
                for sp_set in sp_selections[k]:
                    isp = np.array([i for i, g in enumerate(work_gids) if g in sp_set], dtype=np.int64)
                    if isp.size == 0:
                        continue
                    est_b = _tail_risk_from_delta(work_delta[isp], tail_frac)
                    est_a = _tail_risk_from_delta(-work_delta[isp], tail_frac)
                    sp_accs.append(
                        1.0 if _riskier_label(est_a, est_b) == true_riskier else 0.0
                    )
                per_k[k]["simpoint"] = float(np.mean(sp_accs)) if sp_accs else 0.5

                rand_accs = []
                for ir in rand_selections[k]:
                    est_b = _tail_risk_from_delta(work_delta[ir], tail_frac)
                    est_a = _tail_risk_from_delta(-work_delta[ir], tail_frac)
                    rand_accs.append(
                        1.0 if _riskier_label(est_a, est_b) == true_riskier else 0.0
                    )
                per_k[k]["random"] = float(np.mean(rand_accs)) if rand_accs else 0.5

            record[task] = {"risk_gap": true_gap, "per_k": per_k}

        if record:
            pair_records.append(record)

    def _agg_pairs(task_records):
        out = {}
        for k in K_VALUES:
            out[k] = {}
            for method in [*learned_methods, "simpoint", "random"]:
                vals = [
                    rec["per_k"][k][method]
                    for rec in task_records
                    if k in rec["per_k"]
                ]
                out[k][method] = round(float(np.mean(vals)), 4) if vals else 0.0
        return out

    all_gaps = {task: [] for task in TASK_KEYS}
    for rec in pair_records:
        for task in TASK_KEYS:
            if task in rec:
                all_gaps[task].append(rec[task]["risk_gap"])

    terciles = {}
    for task in TASK_KEYS:
        vals = sorted(all_gaps[task])
        n = len(vals)
        if n >= 3:
            terciles[task] = (vals[n // 3], vals[2 * n // 3])
        else:
            terciles[task] = (0.0, float("inf"))

    agg = {"overall": {}, "hard": {}, "medium": {}, "easy": {}, "n_pairs": len(pair_records)}
    agg["tail_frac"] = float(tail_frac)
    agg["terciles"] = {task: [float(v) for v in tc] for task, tc in terciles.items()}
    agg["n_pairs_by_task"] = {}

    for task in TASK_KEYS:
        task_records = [rec[task] for rec in pair_records if task in rec]
        agg["n_pairs_by_task"][task] = len(task_records)
        agg["overall"][task] = _agg_pairs(task_records)

        tc = terciles[task]
        for stratum in ["hard", "medium", "easy"]:
            stratum_records = [
                rec[task]
                for rec in pair_records
                if task in rec and _difficulty_label(rec[task]["risk_gap"], tc) == stratum
            ]
            agg[stratum][task] = _agg_pairs(stratum_records)

    return agg


def evaluate_multi_hw(
    groups_by_method,
    models_by_method,
    device,
    bbv_lookup,
    held_out_hw,
    n_groups=N_HW_GROUPS,
    hw_per_group=HW_PER_GROUP,
    structured=True,
    hw_json_path=None,
    tail_frac=TAIL_FRAC,
    top_m=TOP_M,
    max_intervals_per_pair=None,
    multi_selector="max_ranknorm",
):
    """Rank candidate HWs by bottleneck risk using a shared set of K intervals."""
    ref_groups = groups_by_method["model"]
    learned_methods = list(models_by_method.keys())
    hw_ids = sorted(held_out_hw)
    rng = np.random.default_rng(SEED + 500)

    hw_groups = []
    if structured:
        hw_json = _resolve_hw_json_path(hw_json_path)
        print(f"  Building structured HW groups from {hw_json} ...")
        hw_groups = _build_structured_hw_groups(
            ref_groups,
            held_out_hw,
            hw_json,
            n_groups=n_groups,
            hw_per_group=hw_per_group,
            rng=rng,
        )
    else:
        raise ValueError(
            "Simulation budget evaluation requires structured candidate groups; random fallback is disabled."
        )

    n_actual = len(hw_groups)
    print(f"  Sampled {n_actual} HW groups (target: {n_groups}, group size: {hw_per_group})")
    if n_actual < n_groups:
        raise RuntimeError(
            f"Only {n_actual}/{n_groups} structured groups found. "
            "Simulation budget evaluation requires genuinely discriminative structured candidate groups; "
            "reduce n_groups or relax group construction, but do not fall back to random groups."
        )

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for group_idx, (grp_hws, common_gids) in enumerate(hw_groups):
        if group_idx % 10 == 0:
            print(f"  group {group_idx}/{n_actual}...")
        if len(grp_hws) != hw_per_group:
            continue

        all_gids = common_gids
        if max_intervals_per_pair and len(all_gids) > max_intervals_per_pair:
            sel_idx = rng.choice(len(all_gids), max_intervals_per_pair, replace=False)
            sel_gids = [all_gids[i] for i in sel_idx]
        else:
            sel_gids = all_gids

        h_0 = grp_hws[0]
        h_candidates = grp_hws[1:]
        n_candidates = len(h_candidates)
        n_sel = len(sel_gids)

        task_score_mats_by_method = {
            method_name: _compute_group_score_mats(
                groups_by_method[method_name],
                models_by_method[method_name],
                device,
                sel_gids,
                h_0,
                h_candidates,
                method_name,
            )
            for method_name in learned_methods
        }

        sp_selections = {}
        rand_selections = {}
        for k in K_VALUES:
            if k > n_sel:
                continue
            sp_selections[k] = [
                _build_simpoint_for_gids(bbv_lookup, set(sel_gids), k, SEED + 3000 + si)
                for si in range(N_SIMPOINT_SEEDS)
            ]
            rand_selections[k] = [
                rng.choice(n_sel, k, replace=False) for _ in range(N_RANDOM_SEEDS)
            ]

        for task in TASK_KEYS:
            t_idx = TASK_INDEX[task]

            true_risks = np.zeros(n_candidates, dtype=np.float64)
            for ci, hw in enumerate(h_candidates):
                delta = np.array(
                    [
                        ref_groups[g][hw]["label"][t_idx] - ref_groups[g][h_0]["label"][t_idx]
                        for g in all_gids
                    ],
                    dtype=np.float64,
                )
                true_risks[ci] = _tail_risk_from_delta(delta, tail_frac)

            if np.max(true_risks) - np.min(true_risks) <= 1e-12:
                continue

            sel_deltas = np.zeros((n_sel, n_candidates), dtype=np.float64)
            for gi, g in enumerate(sel_gids):
                base_val = ref_groups[g][h_0]["label"][t_idx]
                for ci, hw in enumerate(h_candidates):
                    sel_deltas[gi, ci] = ref_groups[g][hw]["label"][t_idx] - base_val

            for k in K_VALUES:
                if k > n_sel:
                    continue

                for method_name in learned_methods:
                    s_group = _aggregate_shared_score(
                        task_score_mats_by_method[method_name][task],
                        multi_selector,
                    )
                    im = np.argsort(s_group)[::-1][:k]
                    est_method = np.array(
                        [_tail_risk_from_delta(sel_deltas[im, ci], tail_frac) for ci in range(n_candidates)],
                        dtype=np.float64,
                    )
                    results[task][k][f"top1_{method_name}"].append(
                        _top1_worst_accuracy(true_risks[None, :], est_method[None, :])
                    )
                    results[task][k][f"pairwise_acc_{method_name}"].append(
                        _pairwise_risk_ordering_accuracy(true_risks[None, :], est_method[None, :])
                    )
                    results[task][k][f"topm_acc_{method_name}"].append(
                        _topm_set_accuracy(true_risks[None, :], est_method[None, :], top_m)
                    )
                    results[task][k][f"regret_{method_name}"].append(
                        _worst_candidate_regret(true_risks[None, :], est_method[None, :])
                    )

                # SimPoint
                sp_top1, sp_pair, sp_topm, sp_reg = [], [], [], []
                for sp_set in sp_selections[k]:
                    isp = np.array([i for i, g in enumerate(sel_gids) if g in sp_set], dtype=np.int64)
                    if isp.size == 0:
                        continue
                    est_sp = np.array(
                        [_tail_risk_from_delta(sel_deltas[isp, ci], tail_frac) for ci in range(n_candidates)],
                        dtype=np.float64,
                    )
                    sp_top1.append(_top1_worst_accuracy(true_risks[None, :], est_sp[None, :]))
                    sp_pair.append(
                        _pairwise_risk_ordering_accuracy(true_risks[None, :], est_sp[None, :])
                    )
                    sp_topm.append(_topm_set_accuracy(true_risks[None, :], est_sp[None, :], top_m))
                    sp_reg.append(_worst_candidate_regret(true_risks[None, :], est_sp[None, :]))
                if sp_top1:
                    results[task][k]["top1_simpoint"].append(float(np.mean(sp_top1)))
                    results[task][k]["pairwise_acc_simpoint"].append(float(np.mean(sp_pair)))
                    results[task][k]["topm_acc_simpoint"].append(float(np.mean(sp_topm)))
                    results[task][k]["regret_simpoint"].append(float(np.mean(sp_reg)))

                # Random
                rand_top1, rand_pair, rand_topm, rand_reg = [], [], [], []
                for ir in rand_selections[k]:
                    est_rand = np.array(
                        [_tail_risk_from_delta(sel_deltas[ir, ci], tail_frac) for ci in range(n_candidates)],
                        dtype=np.float64,
                    )
                    rand_top1.append(_top1_worst_accuracy(true_risks[None, :], est_rand[None, :]))
                    rand_pair.append(
                        _pairwise_risk_ordering_accuracy(true_risks[None, :], est_rand[None, :])
                    )
                    rand_topm.append(
                        _topm_set_accuracy(true_risks[None, :], est_rand[None, :], top_m)
                    )
                    rand_reg.append(_worst_candidate_regret(true_risks[None, :], est_rand[None, :]))
                results[task][k]["top1_random"].append(float(np.mean(rand_top1)))
                results[task][k]["pairwise_acc_random"].append(float(np.mean(rand_pair)))
                results[task][k]["topm_acc_random"].append(float(np.mean(rand_topm)))
                results[task][k]["regret_random"].append(float(np.mean(rand_reg)))

    agg = {}
    for task in TASK_KEYS:
        agg[task] = {}
        for k in K_VALUES:
            agg[task][k] = {}
            for prefix in ["top1", "pairwise_acc", "topm_acc", "regret"]:
                for method in [*learned_methods, "simpoint", "random"]:
                    key = f"{prefix}_{method}"
                    vals = results[task][k].get(key, [])
                    agg[task][k][key] = round(float(np.mean(vals)), 4) if vals else 0.0

    agg["n_groups"] = n_actual
    agg["hw_per_group"] = hw_per_group
    agg["candidates_per_group"] = hw_per_group - 1
    agg["tail_frac"] = float(tail_frac)
    agg["top_m"] = int(min(top_m, hw_per_group - 1))
    agg["multi_selector"] = multi_selector
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument(
        "--regressor-checkpoint",
        default=DEFAULT_ABS_REGRESSOR_CHECKPOINT,
    )
    parser.add_argument("--output", default="eval_simulation_budget.json")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n_hw_groups", type=int, default=N_HW_GROUPS)
    parser.add_argument("--tail-frac", type=float, default=TAIL_FRAC)
    parser.add_argument("--top-m", type=int, default=TOP_M)
    parser.add_argument("--max-intervals-per-pair", type=int, default=None)
    selector_choices = [
        "mean_raw",
        "mean_ranknorm",
        "max_ranknorm",
        "top2mean_ranknorm",
        "logsumexp_ranknorm",
    ]
    parser.add_argument("--multi-selector", default="max_ranknorm", choices=selector_choices)
    parser.add_argument(
        "--group-score-mode",
        dest="multi_selector",
        choices=selector_choices,
        help=argparse.SUPPRESS,
    )
    add_dataset_path_arg(parser)
    add_hw_json_arg(parser)
    add_model_override_args(parser)
    add_regressor_override_mode_arg(parser)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_data_path(args.data_path)
    print(f"Loading dataset from {data_path} ...")
    dataset = MicroArchDataset(data_path)
    print(f"  total samples: {len(dataset)}")

    print("Precomputing BBV lookup ...")
    bbv_lookup = precompute_bbv_lookup(dataset)
    print(f"  BBV entries: {len(bbv_lookup)}")

    train_idx, val_idx, train_hw, val_hw = hw_holdout_split(dataset)
    val_idx_list = val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx)
    print(f"  HW split: {len(train_hw)} train + {len(val_hw)} test")

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

    method_data = {}
    groups_by_method = {}
    held_out_hw = None
    for method_name, method_model in method_models.items():
        print(f"Running inference on held-out HW samples for {method_name} ...")
        all_data = collect_predictions_from_indices(
            method_model, dataset, val_idx_list, device, args.batch_size
        )
        print(f"  {len(all_data)} predictions")
        method_data[method_name] = all_data
        groups_by_method[method_name] = build_group_hw_map(all_data)
        if held_out_hw is None:
            held_out_hw = {item["hw_id"] for item in all_data}

    print(f"\n{'=' * 60}")
    print("Scenario 1: Pairwise Bottleneck Discrimination vs K")
    print(f"{'=' * 60}")
    pairwise = evaluate_pairwise(
        groups_by_method,
        method_models,
        device,
        bbv_lookup,
        held_out_hw,
        n_pairs=args.n_pairs,
        tail_frac=args.tail_frac,
        max_intervals_per_pair=args.max_intervals_per_pair,
    )

    print(f"\n  n_pairs: {pairwise['n_pairs']}")
    for stratum in ["overall", "hard", "medium", "easy"]:
        sd = pairwise[stratum].get("l2_mpki", {})
        print(f"\n  [{stratum}] l2_mpki bottleneck-ordering accuracy:")
        header = [f"{'K':<6}", f"{'SensiSelect':>10}"]
        if "regressor" in method_models:
            header.append(f"{'AbsReg':>10}")
        header.extend([f"{'SimPoint':>10}", f"{'Random':>10}"])
        print("  " + " ".join(header))
        for k in K_VALUES:
            row = [f"{k:<6}", f"{sd.get(k, {}).get('model', 0):10.4f}"]
            if "regressor" in method_models:
                row.append(f"{sd.get(k, {}).get('regressor', 0):10.4f}")
            row.extend([
                f"{sd.get(k, {}).get('simpoint', 0):10.4f}",
                f"{sd.get(k, {}).get('random', 0):10.4f}",
            ])
            print("  " + " ".join(row))

    print(f"\n{'=' * 60}")
    print("Scenario 2: Multi-Candidate Bottleneck-Risk Ranking vs Shared Budget")
    print(f"{'=' * 60}")
    multi = evaluate_multi_hw(
        groups_by_method,
        method_models,
        device,
        bbv_lookup,
        held_out_hw,
        n_groups=args.n_hw_groups,
        structured=True,
        tail_frac=args.tail_frac,
        top_m=args.top_m,
        max_intervals_per_pair=args.max_intervals_per_pair,
        multi_selector=args.multi_selector,
        hw_json_path=args.hw_json_path,
    )

    print(
        f"\n  n_groups: {multi['n_groups']}, hw_per_group: {multi['hw_per_group']}, "
        f"candidates_per_group: {multi['candidates_per_group']}"
    )
    for metric_name in ["l1_mpki", "l2_mpki", "l2_miss_rate", "cpi"]:
        md = multi.get(metric_name, {})
        print(f"\n  [{metric_name}] Top-1 / Top-{args.top_m} worst-candidate accuracy:")
        header = [f"{'K':<6}", f"{'Top1-Ours':>10}"]
        if "regressor" in method_models:
            header.append(f"{'Top1-Reg':>10}")
        header.extend([f"{'Top1-SP':>10}", f"{'Top1-Rand':>10}", f"{'TopM-Ours':>10}"])
        if "regressor" in method_models:
            header.append(f"{'TopM-Reg':>10}")
        header.extend([f"{'TopM-SP':>10}", f"{'TopM-Rand':>10}"])
        print("  " + " ".join(header))
        for k in K_VALUES:
            row = [
                f"{k:<6}",
                f"{md.get(k, {}).get('top1_model', 0):10.4f}",
            ]
            if "regressor" in method_models:
                row.append(f"{md.get(k, {}).get('top1_regressor', 0):10.4f}")
            row.extend([
                f"{md.get(k, {}).get('top1_simpoint', 0):10.4f}",
                f"{md.get(k, {}).get('top1_random', 0):10.4f}",
                f"{md.get(k, {}).get('topm_acc_model', 0):10.4f}",
            ])
            if "regressor" in method_models:
                row.append(f"{md.get(k, {}).get('topm_acc_regressor', 0):10.4f}")
            row.extend([
                f"{md.get(k, {}).get('topm_acc_simpoint', 0):10.4f}",
                f"{md.get(k, {}).get('topm_acc_random', 0):10.4f}",
            ])
            print("  " + " ".join(row))

    output = {
        "checkpoint": args.checkpoint,
        "regressor_checkpoint": args.regressor_checkpoint,
        "K_values": K_VALUES,
        "tail_frac": float(args.tail_frac),
        "top_m": int(args.top_m),
        "max_intervals_per_pair": args.max_intervals_per_pair,
        "multi_selector": args.multi_selector,
        "pairwise": pairwise,
        "multi_hw": multi,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
