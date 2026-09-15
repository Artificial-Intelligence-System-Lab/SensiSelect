"""Interval-grouped DataLoader for SeqPredictor.

Many HW configs share one interval's trace features, and the .bin files are the
expensive thing to read.  So we group endpoints by (bench, interval_id): each
__getitem__ loads ONE interval's features once and attaches a minibatch of that
interval's endpoints (HW vectors + labels) for the requested split.

CPU workers do the memmap + on-the-fly transform; pin_memory in the loader.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (
    Example,
    IntervalRef,
    Stats,
    load_interval_features,
    normalize_hw,
    normalize_label,
)


class IntervalGroupedDataset(Dataset):
    """One item = one interval + up to `hw_per_interval` of its endpoints."""

    def __init__(
        self,
        examples: List[Example],
        example_idx: List[int],
        refs: Dict[Tuple[str, int], IntervalRef],
        opcode_lut: np.ndarray,
        stats: Stats,
        seq_len: int,
        hw_per_interval: int = 32,
        seed: int = 42,
        shuffle_hw: bool = True,
    ):
        self.examples = examples
        self.refs = refs
        self.opcode_lut = opcode_lut
        self.stats = stats
        self.seq_len = seq_len
        self.hw_per_interval = hw_per_interval
        self.shuffle_hw = shuffle_hw
        self._rng = np.random.default_rng(seed)

        # Group the requested example indices by interval key.
        groups: Dict[Tuple[str, int], List[int]] = {}
        for i in example_idx:
            e = examples[i]
            groups.setdefault((e.bench, e.interval_id), []).append(i)

        # Materialize items: each interval becomes one or more items (chunked by
        # hw_per_interval so no single item is too large).
        self.items: List[Tuple[Tuple[str, int], List[int]]] = []
        for key, idxs in groups.items():
            for s in range(0, len(idxs), hw_per_interval):
                self.items.append((key, idxs[s:s + hw_per_interval]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        key, ex_idxs = self.items[i]
        if self.shuffle_hw and len(ex_idxs) > 1:
            ex_idxs = list(ex_idxs)
            self._rng.shuffle(ex_idxs)
        ref = self.refs[key]
        feats = load_interval_features(
            ref, self.opcode_lut, self.seq_len, accdist_std=self.stats.accdist_std
        )
        hw = np.stack([
            normalize_hw(self.examples[j], self.stats) for j in ex_idxs
        ]).astype(np.float32)
        lbl = np.stack([
            normalize_label(self.examples[j].label_raw, self.stats) for j in ex_idxs
        ]).astype(np.float32)
        hw_ids = np.array([self.examples[j].hw_id for j in ex_idxs], dtype=np.int64)
        return {
            "feats": {k: torch.from_numpy(v) for k, v in feats.items()},
            "hw": torch.from_numpy(hw),       # [H, 47]
            "label": torch.from_numpy(lbl),   # [H, 10]
            "hw_ids": torch.from_numpy(hw_ids),
            "bench": ref.benchmark,
            "interval_id": ref.interval_id,
        }


def collate_intervals(batch: List[dict]) -> dict:
    """Stack a batch of interval-items.

    The per-interval feature tensors are tiled to match each interval's endpoint
    count, so the model sees one (interval-features, hw) pair per endpoint.  This
    keeps the model interface simple while reading each .bin only once per item.
    """
    feat_keys = ["opcode_idx", "bhist", "regbm", "accdist", "mask"]
    feats_out = {k: [] for k in feat_keys}
    hw_list, lbl_list, hwid_list = [], [], []
    meta = []  # (bench, interval_id, n_endpoints)

    for item in batch:
        h = item["hw"].shape[0]
        for k in feat_keys:
            f = item["feats"][k]  # [W, L, ...]
            feats_out[k].append(f.unsqueeze(0).expand(h, *f.shape))
        hw_list.append(item["hw"])
        lbl_list.append(item["label"])
        hwid_list.append(item["hw_ids"])
        meta.append((item["bench"], item["interval_id"], h))

    feats = {k: torch.cat(v, dim=0).contiguous() for k, v in feats_out.items()}
    return {
        "feats": feats,
        "hw": torch.cat(hw_list, dim=0),
        "label": torch.cat(lbl_list, dim=0),
        "hw_ids": torch.cat(hwid_list, dim=0),
        "meta": meta,
    }
