"""Evaluation harness for SeqPredictor — mirrors the FiLM metrics
(train_unified_v2.py:evaluate_delta_bundle / safe_spearman / topk_recall).

Metrics (all in normalized label space, matching the FiLM model):
  - abs_rho[metric]      = Spearman(true_abs, pred_abs) over all test endpoints.
  - sens_rho[metric]     = Spearman(true_sens, pred_sens) over all test HW pairs,
                           per (bench, interval_id); pair sensitivity = |b - a|.
  - topk_recall@5%       = recall of top-5% true-sensitivity pairs in top-5% pred.

Reports mean over 10, min, and non-cache mean (matching the FiLM summary line).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch

from model_unified import METRIC_KEYS
from .data import CACHE_METRICS

NON_CACHE_METRICS = tuple(k for k in METRIC_KEYS if k not in CACHE_METRICS)


# ---- Reproduce the repo's metric primitives exactly (train_unified_v2.py) ----

def safe_spearman(y_true, y_pred) -> float:
    from scipy.stats import spearmanr
    if len(y_true) < 3:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else 0.0


def topk_recall(true_sens, pred_sens, k=None) -> float:
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


@torch.no_grad()
def collect_predictions(model, loader, device) -> Dict[Tuple[str, int, int], dict]:
    """Run the model over a loader; return preds/targets keyed by
    (bench, interval_id, hw_id) -> {'pred': [10], 'true': [10]}.
    """
    model.eval()
    out: Dict[Tuple[str, int, int], dict] = {}
    use_amp = device.type == "cuda"
    for batch in loader:
        feats = {k: v.to(device, non_blocking=True) for k, v in batch["feats"].items()}
        hw = batch["hw"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            preds = model(feats, hw)
        preds = preds.float().cpu().numpy()
        labels = batch["label"].numpy()
        hw_ids = batch["hw_ids"].numpy()
        # Reconstruct (bench, iid) per row from meta.
        rows = []
        for (bench, iid, h) in batch["meta"]:
            rows.extend([(bench, iid)] * h)
        for r, (bench, iid) in enumerate(rows):
            out[(bench, iid, int(hw_ids[r]))] = {
                "pred": preds[r], "true": labels[r],
            }
    return out


def compute_metrics(
    preds: Dict[Tuple[str, int, int], dict],
    topk_frac: float = 0.05,
    max_pairs_per_interval: int = 0,
    seed: int = 42,
) -> dict:
    """Compute abs_rho / sens_rho / topk_recall from collected predictions.

    HW pairs are formed within the TEST HW set, per (bench, interval_id).
    max_pairs_per_interval>0 caps the enumerated pairs per interval (sampled).
    """
    rng = np.random.default_rng(seed)

    # ---- abs_rho: over all endpoints ----
    abs_true = {k: [] for k in METRIC_KEYS}
    abs_pred = {k: [] for k in METRIC_KEYS}
    for v in preds.values():
        for i, k in enumerate(METRIC_KEYS):
            abs_true[k].append(float(v["true"][i]))
            abs_pred[k].append(float(v["pred"][i]))
    abs_rho = {k: safe_spearman(abs_true[k], abs_pred[k]) for k in METRIC_KEYS}

    # ---- group endpoints by (bench, interval_id) ----
    groups: Dict[Tuple[str, int], List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (bench, iid, hw_id), v in preds.items():
        groups[(bench, iid)].append((hw_id, v["pred"], v["true"]))

    # ---- sens pairs ----
    sens_true = {k: [] for k in METRIC_KEYS}
    sens_pred = {k: [] for k in METRIC_KEYS}
    n_pairs = 0
    for key, endpoints in groups.items():
        if len(endpoints) < 2:
            continue
        pairs = list(combinations(range(len(endpoints)), 2))
        if max_pairs_per_interval > 0 and len(pairs) > max_pairs_per_interval:
            sel = rng.choice(len(pairs), size=max_pairs_per_interval, replace=False)
            pairs = [pairs[s] for s in sel]
        for a, b in pairs:
            pa, ta = endpoints[a][1], endpoints[a][2]
            pb, tb = endpoints[b][1], endpoints[b][2]
            ps = np.abs(pb - pa)
            ts = np.abs(tb - ta)
            for i, k in enumerate(METRIC_KEYS):
                sens_pred[k].append(float(ps[i]))
                sens_true[k].append(float(ts[i]))
            n_pairs += 1

    sens_rho = {k: safe_spearman(sens_true[k], sens_pred[k]) for k in METRIC_KEYS}

    topk = {}
    for k in METRIC_KEYS:
        st = np.asarray(sens_true[k], dtype=np.float64)
        sp = np.asarray(sens_pred[k], dtype=np.float64)
        if len(st) > 10:
            kk = max(1, int(len(st) * topk_frac))
            topk[k] = float(topk_recall(st, sp, k=kk))
        else:
            topk[k] = 0.0

    def _mean(d, keys):
        vals = [d[k] for k in keys]
        return float(np.mean(vals)) if vals else 0.0

    sens_vals = [sens_rho[k] for k in METRIC_KEYS]
    return {
        "abs_rho": abs_rho,
        "sens_rho": sens_rho,
        "topk_recall": topk,
        "abs_rho_mean": _mean(abs_rho, METRIC_KEYS),
        "sens_rho_mean": float(np.mean(sens_vals)),
        "sens_rho_min": float(np.min(sens_vals)),
        "sens_rho_nc_mean": _mean(sens_rho, NON_CACHE_METRICS),
        "topk_mean": _mean(topk, METRIC_KEYS),
        "n_endpoints": len(preds),
        "n_pairs": n_pairs,
        "n_intervals": len(groups),
    }


def print_metrics(m: dict, tag: str = "") -> None:
    print(f"\n=== SeqPredictor eval {tag} ===")
    print(f"  endpoints={m['n_endpoints']}  intervals={m['n_intervals']}  pairs={m['n_pairs']}")
    print(f"  {'metric':<14} {'abs_rho':>9} {'sens_rho':>9} {'topk@5%':>9}")
    for k in METRIC_KEYS:
        mark = "*" if k in CACHE_METRICS else " "
        print(f"  {mark}{k:<13} {m['abs_rho'][k]:>9.4f} {m['sens_rho'][k]:>9.4f} {m['topk_recall'][k]:>9.4f}")
    print(
        f"  SUMMARY: abs_rho mean={m['abs_rho_mean']:.4f} | "
        f"sens_rho mean={m['sens_rho_mean']:.4f} min={m['sens_rho_min']:.4f} "
        f"nc_mean={m['sens_rho_nc_mean']:.4f} | topk_mean={m['topk_mean']:.4f}"
    )
    print("  (* = cache metric)")
