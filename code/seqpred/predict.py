"""Produce SeqPredictor predictions in the same list-of-dicts format the FiLM
eval scripts consume (eval_film_common.collect_predictions_from_indices).

Output format
-------------
The FiLM eval helper returns, per (interval, hw), a dict:
    {'group_id': int, 'hw_id': int, 'pred': np.array(10), 'label': np.array(10)}
and downstream evaluation (pairing, recall@K, ndcg, budget, OOD strata) is
generic on `pred`.  This module produces the same list-of-dicts but with
SeqPredictor's 10-vector `pred`, for the same (interval, hw) items, so the
metric machinery can be reused verbatim by passing `model=None` to the FiLM
eval functions (which then fall back to |pred_delta| as the ranking signal).

Data alignment
--------------
* group_id / hw_id / label are taken from the FiLM dataset
  (`dataset.samples[idx]`), so group_ids align across FiLM and SeqPred and the
  ground-truth deltas are identical (label is FiLM-normalized; metrics are
  rank-based on |label_a - label_b| so the GT object is shared).
* `pred` is SeqPredictor's output in its own normalized label space — we do not
  match FiLM's normalization (the metrics are monotonic/rank-based).
* The 47-dim HW vector fed to SeqPred is rebuilt from the raw hardware dict in
  training_dataset_ALL.pkl (keyed by hw_id) using seqpred/data.py's extraction
  + the saved hw_mean/hw_std from best.pt, as in training.
* Trace features come from tao_features keyed by (benchmark, interval_id).
  Intervals without trace features are skipped and counted in the coverage report.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seqpred.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_PKL,
    Stats,
    extract_hw_onehot,
    extract_hw_raw,
    load_interval_features,
    make_opcode_lut,
    scan_manifests,
    vectorize_hw,
)
from seqpred.model import SeqPredictor  # noqa: E402

# TaoPredictor checkpoint, resolved relative to the release root
# (code/seqpred/ -> ../../checkpoints/). Override with TAOPREDICTOR_CHECKPOINT.
DEFAULT_CHECKPOINT = os.environ.get(
    "TAOPREDICTOR_CHECKPOINT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "checkpoints", "taopredictor_best.pt"),
)


@dataclass
class SeqPredBundle:
    model: SeqPredictor
    lut: np.ndarray
    stats: Stats
    vocab: Dict[int, int]
    seq_len: int
    refs: Dict[Tuple[str, int], object]
    hw_raw_by_id: Dict[int, np.ndarray]
    hw_onehot_by_id: Dict[int, np.ndarray]


def _load_stats_from_ckpt(raw_stats: dict) -> Stats:
    return Stats(
        hw_mean=np.asarray(raw_stats["hw_mean"], dtype=np.float32),
        hw_std=np.asarray(raw_stats["hw_std"], dtype=np.float32),
        label_mean=np.asarray(raw_stats["label_mean"], dtype=np.float32),
        label_std=np.asarray(raw_stats["label_std"], dtype=np.float32),
        accdist_std=float(raw_stats["accdist_std"]),
    )


def _build_hw_maps(pkl_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """One raw-continuous (42) + one-hot (5) HW vector per hw_id, from the pkl.

    Replicates seqpred/data.py extraction so the HW encoding is identical to
    training.  The raw hardware dict for a given hw_id is fixed (same config
    everywhere it appears), so we record the first occurrence.
    """
    data = pickle.load(open(pkl_path, "rb"))
    hw_raw_by_id: Dict[int, np.ndarray] = {}
    hw_onehot_by_id: Dict[int, np.ndarray] = {}
    for item in data:
        hw = item["hardware"]
        hw_id = int(hw.get("id", -1))
        if hw_id in hw_raw_by_id:
            continue
        hw_raw_by_id[hw_id] = np.asarray(extract_hw_raw(hw), dtype=np.float32)
        hw_onehot_by_id[hw_id] = np.asarray(extract_hw_onehot(hw), dtype=np.float32)
    return hw_raw_by_id, hw_onehot_by_id


def load_seqpred(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: Optional[torch.device] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    pkl_path: str = DEFAULT_PKL,
) -> SeqPredBundle:
    """Load the trained SeqPredictor + all artefacts needed to predict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = {int(k): int(v) for k, v in ck["vocab"].items()}
    stats = _load_stats_from_ckpt(ck["stats"])
    cfg = ck.get("cfg", {})
    seq_len = int(cfg.get("seq_len", 256))
    d = int(cfg.get("d", 128))

    lut = make_opcode_lut(vocab)
    model = SeqPredictor(opcode_vocab_size=len(vocab) + 1, d=d).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    refs = scan_manifests(data_dir)
    hw_raw_by_id, hw_onehot_by_id = _build_hw_maps(pkl_path)

    return SeqPredBundle(
        model=model,
        lut=lut,
        stats=stats,
        vocab=vocab,
        seq_len=seq_len,
        refs=refs,
        hw_raw_by_id=hw_raw_by_id,
        hw_onehot_by_id=hw_onehot_by_id,
    )


def _sample_meta(dataset, idx: int):
    """(benchmark, interval_id, hw_id, group_id, label[10]) from a FiLM sample."""
    s = dataset.samples[idx]
    bench = s.get("benchmark", "unknown")
    iid = int(s.get("interval_id", -1))
    hw_id = int(s.get("hw_id", -1))
    group_id = int(s.get("group_id", -1))
    labels = s["labels"]
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    label = np.asarray(labels, dtype=np.float32)[:10]
    return bench, iid, hw_id, group_id, label


@torch.no_grad()
def collect_predictions_seqpred(
    bundle: SeqPredBundle,
    dataset,
    indices,
    device: Optional[torch.device] = None,
    hw_chunk: int = 64,
    feature_cache: Optional[dict] = None,
    progress_every: int = 0,
):
    """Run SeqPredictor over the same dataset indices the FiLM eval uses.

    Returns (all_data, coverage) where:
      all_data : list of {'group_id', 'hw_id', 'pred'(10), 'label'(10)}
      coverage : dict with served/dropped counts and reasons.

    Items whose interval has no available trace features are skipped (recorded in
    coverage), since SeqPredictor cannot predict them.  An interval's features
    are loaded once and reused across all its HW configs (cached).
    """
    if device is None:
        device = next(bundle.model.parameters()).device
    if feature_cache is None:
        feature_cache = {}

    # Group requested indices by interval key so each .bin is read once.
    by_interval: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for idx in indices:
        bench, iid, hw_id, gid, _ = _sample_meta(dataset, idx)
        by_interval[(bench, iid)].append(idx)

    all_data = []
    n_requested = len(indices)
    n_served = 0
    n_drop_no_feats = 0
    n_drop_no_hw = 0
    dropped_intervals = set()
    dropped_interval_items = 0
    served_intervals = set()
    feat_keys = ["opcode_idx", "bhist", "regbm", "accdist", "mask"]

    processed = 0
    for key, idx_list in by_interval.items():
        ref = bundle.refs.get(key)
        if ref is None:
            n_drop_no_feats += len(idx_list)
            dropped_interval_items += len(idx_list)
            dropped_intervals.add(key)
            continue

        if key in feature_cache:
            feats_np = feature_cache[key]
        else:
            feats_np = load_interval_features(
                ref, bundle.lut, bundle.seq_len, accdist_std=bundle.stats.accdist_std
            )
            feature_cache[key] = feats_np
        served_intervals.add(key)

        # Build HW vectors + metadata for every requested (interval, hw) item.
        hw_vecs = []
        metas = []  # (group_id, hw_id, label)
        for idx in idx_list:
            bench, iid, hw_id, gid, label = _sample_meta(dataset, idx)
            raw = bundle.hw_raw_by_id.get(hw_id)
            onehot = bundle.hw_onehot_by_id.get(hw_id)
            if raw is None or onehot is None:
                n_drop_no_hw += 1
                continue
            hw_vecs.append(
                vectorize_hw(raw, onehot, bundle.stats.hw_mean, bundle.stats.hw_std)
            )
            metas.append((gid, hw_id, label))

        if not hw_vecs:
            continue

        # Pre-build per-interval feature tensors (shared across this interval's HW).
        base_feats = {k: torch.from_numpy(feats_np[k]) for k in feat_keys}

        # Chunk this interval's HW configs through the model.
        for s in range(0, len(hw_vecs), hw_chunk):
            e = min(s + hw_chunk, len(hw_vecs))
            h = e - s
            hw_t = torch.from_numpy(np.stack(hw_vecs[s:e]).astype(np.float32)).to(device)
            feats_b = {
                k: base_feats[k].unsqueeze(0).expand(h, *base_feats[k].shape).to(device)
                for k in feat_keys
            }
            use_amp = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = bundle.model(feats_b, hw_t)
            preds = preds.float().cpu().numpy()
            for r in range(h):
                gid, hw_id, label = metas[s + r]
                all_data.append({
                    "group_id": int(gid),
                    "hw_id": int(hw_id),
                    "pred": preds[r].astype(np.float32, copy=False),
                    "label": label,
                })
                n_served += 1

        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"    seqpred: {processed}/{len(by_interval)} intervals, "
                  f"{n_served} items served", flush=True)

    coverage = {
        "n_requested_items": int(n_requested),
        "n_served_items": int(n_served),
        "n_dropped_no_features": int(n_drop_no_feats),
        "n_dropped_no_hw_vector": int(n_drop_no_hw),
        "n_intervals_requested": int(len(by_interval)),
        "n_intervals_served": int(len(served_intervals)),
        "n_intervals_dropped_no_features": int(len(dropped_intervals)),
        "served_fraction": float(n_served) / float(max(1, n_requested)),
    }
    return all_data, coverage
