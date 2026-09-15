"""Data layer for SeqPredictor.

Responsibilities:
  - build the global opcode vocab (union of discriminant keys across all benches present),
  - load per-interval trace features from the compact on-disk .bin form via memmap,
    transforming on the fly (no full float expansion),
  - replicate the repo's HW encoding (dataset.py:_extract_hw_raw) and label
    normalization (log1p MPKI -> z-score) exactly,
  - replicate the HW-holdout split (train_unified_v2.py hardware-holdout splitting; seed 42, 80/20 by hw_id),
  - produce (benchmark, interval_id, hw) training examples grouped by interval to
    minimise re-reading the same .bin files.

The compact .bin form is the cache; we never pre-expand features to float on disk.
"""

from __future__ import annotations

import glob
import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- Repo conventions (import read-only) ----
from model_unified import METRIC_KEYS  # 10 endpoints, this order
from dataset import MicroArchDataset    # for _extract_hw_raw (called unbound)

# Cache metrics (matches FiLMUnifiedModel.CACHE_METRICS / train_unified_v2 eval).
CACHE_METRICS = ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate")

# MPKI metrics get log1p before z-score (dataset.py:_extract_labels_raw).
MPKI_METRIC_IDX = tuple(
    METRIC_KEYS.index(k) for k in ("branch_mpki", "l1_mpki", "l1i_mpki", "l2_mpki")
)

# Resolve data locations relative to the release root (code/seqpred/ -> ../../).
# Override with the TAO_FEATURES_DIR / TRAINING_PKL environment variables.
_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.environ.get(
    "TAO_FEATURES_DIR", os.path.join(_RELEASE_ROOT, "data", "tao_features"))
DEFAULT_PKL = os.environ.get(
    "TRAINING_PKL", os.path.join(_RELEASE_ROOT, "data", "training_dataset_ALL.pkl"))

REGBM_WORDS = 4      # 256-bit reg bitmap = 4 u64 words
REGBM_BITS = 256
ACCDIST_NM = 32      # nm per record
N_WINDOWS = 8
WINDOW_LEN = 2048
HW_CONT_DIM = 42     # continuous HW features from _extract_hw_raw (z-scored)
HW_ONEHOT_DIM = 5    # BP type (3) + prefetcher (2), NOT z-scored
HW_DIM = HW_CONT_DIM + HW_ONEHOT_DIM  # 47 — matches FiLMUnifiedModel.HW_DIM


# ============================================================
# HW encoding (replicate dataset.py:_extract_hw_raw exactly)
# ============================================================

# Call the repo's unbound method so encoding is guaranteed identical.  The
# method only touches `hw` + `math` (no instance state), so a dummy self is fine.
class _HWProxy:
    pass


def extract_hw_raw(hw: Dict) -> List[float]:
    """Return the 42-dim raw continuous HW vector, identical to
    dataset.py:_extract_hw_raw (used for z-score stats)."""
    return MicroArchDataset._extract_hw_raw(_HWProxy(), hw)


# BP-type / prefetcher one-hot maps (replicate dataset.py:_vectorize_hardware).
_BP_MAP = {"BiModeBP": [1, 0, 0], "TournamentBP": [0, 1, 0], "LTAGE": [0, 0, 1]}
_PF_MAP = {"Null": [1, 0], "StridePrefetcher": [0, 1]}


def extract_hw_onehot(hw: Dict) -> List[float]:
    """Return the 5-dim one-hot HW categorical (BP type 3 + prefetcher 2),
    identical to dataset.py:_vectorize_hardware (NOT z-scored)."""
    onehot = list(_BP_MAP.get(hw["bp_type"], [0, 0, 0]))
    onehot.extend(_PF_MAP.get(hw.get("l1d_prefetcher"), [0, 0]))
    return [float(x) for x in onehot]


def vectorize_hw(hw_cont_raw: np.ndarray, hw_onehot: np.ndarray,
                 hw_mean: np.ndarray, hw_std: np.ndarray) -> np.ndarray:
    """Build the full 47-dim HW vector: z-score the 42 continuous features
    (using TRAIN-split mean/std) then concat the 5 un-normalized one-hot dims.
    Mirrors dataset.py:_vectorize_hardware exactly."""
    cont = (hw_cont_raw - hw_mean) / hw_std
    return np.concatenate([cont, hw_onehot]).astype(np.float32)


def extract_labels_raw(lbl: Dict) -> List[float]:
    """Return the 10-dim raw label vector (log1p on MPKI), identical to dataset.py."""
    return MicroArchDataset._extract_labels_raw(_HWProxy(), lbl)


# ============================================================
# Opcode vocab
# ============================================================

def build_opcode_vocab(data_dir: str = DEFAULT_DATA_DIR) -> Dict[int, int]:
    """Union the keys of every tao_features/*/opcode_dict.json -> dense map.

    Returns discriminant(int) -> dense_idx(int), with idx 0 reserved for
    unknown/pad. So vocab size for the embedding = len(map) + 1.
    """
    discriminants = set()
    for fp in sorted(glob.glob(os.path.join(data_dir, "*", "opcode_dict.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for k in d.keys():
            discriminants.add(int(k))
    vocab = {disc: i + 1 for i, disc in enumerate(sorted(discriminants))}  # 0 = unk/pad
    return vocab


def make_opcode_lut(vocab: Dict[int, int], max_disc: Optional[int] = None) -> np.ndarray:
    """Build a dense lookup table disc -> idx for O(1) vectorized mapping.

    Discriminants are sparse (0..~4123). LUT covers [0, max_disc]; out-of-range
    or unseen discriminants map to 0 (unknown).
    """
    if max_disc is None:
        max_disc = max(vocab.keys()) if vocab else 0
    lut = np.zeros(max_disc + 1, dtype=np.int64)
    for disc, idx in vocab.items():
        if 0 <= disc <= max_disc:
            lut[disc] = idx
    return lut


# ============================================================
# Per-interval feature loading (memmap + on-the-fly transform)
# ============================================================

@dataclass
class IntervalRef:
    benchmark: str
    interval_id: int
    num_records: int
    opcode_file: str
    regbm_file: str
    bhist_file: str
    accdist_file: str


def scan_manifests(data_dir: str = DEFAULT_DATA_DIR) -> Dict[Tuple[str, int], IntervalRef]:
    """Scan all available manifests; return (bench, interval_id) -> IntervalRef.

    Only includes intervals whose four .bin files all exist with non-zero size
    and num_records > 0 (skips ragged/empty tail intervals).
    """
    refs: Dict[Tuple[str, int], IntervalRef] = {}
    for mf in sorted(glob.glob(os.path.join(data_dir, "*", "manifest.json"))):
        bench_dir = os.path.dirname(mf)
        bench = os.path.basename(bench_dir)
        try:
            m = json.load(open(mf))
        except Exception:
            continue
        for iv in m.get("intervals", []):
            nrec = int(iv.get("num_records", 0))
            if nrec <= 0:
                continue
            files = {
                "opcode": os.path.join(bench_dir, iv["opcode_file"]),
                "regbm": os.path.join(bench_dir, iv["regbm_file"]),
                "bhist": os.path.join(bench_dir, iv["bhist_file"]),
                "accdist": os.path.join(bench_dir, iv["accdist_file"]),
            }
            ok = True
            for f in files.values():
                if not (os.path.exists(f) and os.path.getsize(f) > 0):
                    ok = False
                    break
            if not ok:
                continue
            refs[(bench, int(iv["interval_id"]))] = IntervalRef(
                benchmark=bench,
                interval_id=int(iv["interval_id"]),
                num_records=nrec,
                opcode_file=files["opcode"],
                regbm_file=files["regbm"],
                bhist_file=files["bhist"],
                accdist_file=files["accdist"],
            )
    return refs


def load_interval_features(
    ref: IntervalRef,
    opcode_lut: np.ndarray,
    seq_len: int,
    accdist_std: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Load + transform one interval into 8 windows x seq_len records.

    Returns dict of:
      opcode_idx : int64  [N_WINDOWS, seq_len]   (dense vocab idx, 0=pad/unk)
      bhist      : int64  [N_WINDOWS, seq_len]   (0..255 categorical)
      regbm      : float32[N_WINDOWS, seq_len, 256]  multi-hot
      accdist    : float32[N_WINDOWS, seq_len, 32]   signed-log (normalized by std)
      mask       : float32[N_WINDOWS, seq_len]   1 = real record, 0 = pad

    Windows are the contiguous record blocks (records[k*2048:(k+1)*2048]); each
    2048-record window is subsampled by stride to seq_len records.  Ragged tail
    intervals (num_records not a multiple of 2048) are padded.
    """
    n = ref.num_records

    op = np.memmap(ref.opcode_file, dtype="<u4", mode="r", shape=(n,))
    bh = np.memmap(ref.bhist_file, dtype="<u8", mode="r", shape=(n,))
    rb = np.memmap(ref.regbm_file, dtype="<u8", mode="r", shape=(n, REGBM_WORDS))
    ad = np.memmap(ref.accdist_file, dtype="<i8", mode="r", shape=(n, ACCDIST_NM))

    out_op = np.zeros((N_WINDOWS, seq_len), dtype=np.int64)
    out_bh = np.zeros((N_WINDOWS, seq_len), dtype=np.int64)
    out_rb = np.zeros((N_WINDOWS, seq_len, REGBM_BITS), dtype=np.float32)
    out_ad = np.zeros((N_WINDOWS, seq_len, ACCDIST_NM), dtype=np.float32)
    out_mask = np.zeros((N_WINDOWS, seq_len), dtype=np.float32)

    for w in range(N_WINDOWS):
        start = w * WINDOW_LEN
        if start >= n:
            break
        end = min(start + WINDOW_LEN, n)
        avail = end - start
        # Subsample this window's records by stride to seq_len.
        if avail >= seq_len:
            rel = np.linspace(0, avail - 1, seq_len).astype(np.int64)
        else:
            rel = np.arange(avail, dtype=np.int64)  # short window: take what exists
        idx = start + rel
        k = len(idx)

        # opcode: discriminant -> dense idx via LUT (clip out-of-range -> 0)
        disc = np.asarray(op[idx], dtype=np.int64)
        disc = np.where(disc < len(opcode_lut), disc, 0)
        out_op[w, :k] = opcode_lut[disc]

        # bhist: low nq(8) bits already; values 0..255 categorical
        out_bh[w, :k] = np.asarray(bh[idx] & 0xFF, dtype=np.int64)

        # regbm: 4 u64 words -> 256-bit multi-hot (word0 = ids 0..63, LE)
        words = np.asarray(rb[idx], dtype=np.uint64)  # [k, 4]
        bits = np.unpackbits(words.view(np.uint8).reshape(k, REGBM_WORDS, 8), axis=2,
                             bitorder="little")  # [k, 4, 64]
        out_rb[w, :k] = bits.reshape(k, REGBM_BITS).astype(np.float32)

        # accdist: signed-log, then divide by global std
        raw = np.asarray(ad[idx], dtype=np.float64)  # [k, 32]
        slog = np.sign(raw) * np.log1p(np.abs(raw))
        out_ad[w, :k] = (slog / max(accdist_std, 1e-6)).astype(np.float32)

        out_mask[w, :k] = 1.0

    return {
        "opcode_idx": out_op,
        "bhist": out_bh,
        "regbm": out_rb,
        "accdist": out_ad,
        "mask": out_mask,
    }


# ============================================================
# Labels / HW / split
# ============================================================

@dataclass
class Example:
    bench: str
    interval_id: int
    hw_id: int
    hw_cont_raw: np.ndarray   # [42] continuous raw (pre-z-score)
    hw_onehot: np.ndarray     # [5] one-hot categorical (never z-scored)
    label_raw: np.ndarray     # [10] raw (post log1p on MPKI, pre-z-score)


@dataclass
class Stats:
    hw_mean: np.ndarray
    hw_std: np.ndarray
    label_mean: np.ndarray
    label_std: np.ndarray
    accdist_std: float


def build_examples(
    pkl_path: str,
    refs: Dict[Tuple[str, int], IntervalRef],
) -> List[Example]:
    """Build endpoint examples for which trace features are available.

    Mirrors dataset.py: hw_id = int(hw['id']); labels via _extract_labels_raw;
    HW via _extract_hw_raw.
    """
    data = pickle.load(open(pkl_path, "rb"))
    examples: List[Example] = []
    for item in data:
        bench = item.get("benchmark", "unknown")
        iid = int(item.get("interval_id", -1))
        if (bench, iid) not in refs:
            continue
        hw = item["hardware"]
        lbl = item["label"]
        hw_id = int(hw.get("id", -1))
        hw_cont_raw = np.asarray(extract_hw_raw(hw), dtype=np.float32)
        hw_onehot = np.asarray(extract_hw_onehot(hw), dtype=np.float32)
        label_raw = np.asarray(extract_labels_raw(lbl), dtype=np.float32)
        examples.append(Example(bench, iid, hw_id, hw_cont_raw, hw_onehot, label_raw))
    return examples


def hw_holdout_split(
    examples: List[Example], seed: int = 42
) -> Tuple[List[int], List[int], set, set]:
    """Split examples by hardware configuration (80/20 by default).

    SWEEP_BALANCE and EXCLUDE_SWEEP select optional exploration protocols,
    not the default hardware-holdout protocol used for the paper.

    Returns (train_idx, val_idx, train_hw, val_hw) where idx index into examples.
    """
    hw_ids = np.array([e.hw_id for e in examples])
    unique_hw = np.unique(hw_ids)
    if os.getenv("SWEEP_BALANCE", "0").strip() == "1":
        # Optional exploration split with balanced sweep coverage:
        # 100 sweeps 50/50 (rng seed+100), 5 bases->train, 200 multi-param 80/20 (rng seed).
        _sp = os.getenv("SWEEP_CONFIG_JSON") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..",
            "data", "gem5_hardware_vectors_single_param.json")
        if not os.path.exists(_sp):
            raise RuntimeError(f"SWEEP_BALANCE=1 but sweep config json not found: {_sp}")
        _cfgs = json.load(open(_sp))
        if isinstance(_cfgs, dict):
            _cfgs = _cfgs.get("configs", list(_cfgs.values()))
        _base_ids, _sweep_ids = set(), set()
        for c in _cfgs:
            if not (isinstance(c, dict) and "id" in c):
                continue
            (_base_ids if str(c.get("name", "")).startswith("Base_") else _sweep_ids).add(int(c["id"]))
        if not _sweep_ids:
            raise RuntimeError(f"SWEEP_BALANCE=1 but no sweep ids loaded from {_sp}")
        _uniq = [int(h) for h in unique_hw]
        _multi = np.array(sorted(h for h in _uniq if h not in _sweep_ids and h not in _base_ids))
        _sweeps = np.array(sorted(h for h in _uniq if h in _sweep_ids))
        np.random.default_rng(seed).shuffle(_multi)
        np.random.default_rng(seed + 100).shuffle(_sweeps)
        _n_m = int(len(_multi) * 0.8)
        _n_s = len(_sweeps) // 2
        train_hw = set(_multi[:_n_m].tolist()) | set(_sweeps[:_n_s].tolist()) | {int(b) for b in _base_ids if b in _uniq}
        val_hw = set(_multi[_n_m:].tolist()) | set(_sweeps[_n_s:].tolist())
        train_idx = [i for i in range(len(examples)) if int(hw_ids[i]) in train_hw]
        val_idx = [i for i in range(len(examples)) if int(hw_ids[i]) in val_hw]
        return train_idx, val_idx, train_hw, val_hw
    if os.getenv("EXCLUDE_SWEEP", "0").strip() == "1":
        # Optional exploration split: exclude the sweep and base configurations
        # listed in the config JSON from both training and validation, then split
        # the remaining hardware configurations 80/20.
        _sp = os.getenv("SWEEP_CONFIG_JSON") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..",
            "data", "gem5_hardware_vectors_single_param.json")
        if not os.path.exists(_sp):
            raise RuntimeError(f"EXCLUDE_SWEEP=1 but sweep config json not found: {_sp}")
        _cfgs = json.load(open(_sp))
        if isinstance(_cfgs, dict):
            _cfgs = _cfgs.get("configs", list(_cfgs.values()))
        _sweep_ids = {int(c["id"]) for c in _cfgs if isinstance(c, dict) and "id" in c}
        if not _sweep_ids:
            raise RuntimeError(f"EXCLUDE_SWEEP=1 but no sweep ids loaded from {_sp}")
        unique_hw = np.array([h for h in unique_hw if int(h) not in _sweep_ids])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_hw)
    n_train_hw = int(len(unique_hw) * 0.8)
    train_hw = set(unique_hw[:n_train_hw].tolist())
    val_hw = set(unique_hw[n_train_hw:].tolist())
    train_idx = [i for i in range(len(examples)) if hw_ids[i] in train_hw]
    val_idx = [i for i in range(len(examples)) if hw_ids[i] in val_hw]
    return train_idx, val_idx, train_hw, val_hw


def compute_stats(
    examples: List[Example],
    train_idx: List[int],
    refs: Dict[Tuple[str, int], IntervalRef],
    opcode_lut: np.ndarray,
    seq_len: int,
    accdist_sample_intervals: int = 32,
) -> Stats:
    """Compute z-score stats on the TRAIN split only (HW + labels), plus a
    global accdist signed-log std estimated from a sample of train intervals.
    """
    hw_mat = np.stack([examples[i].hw_cont_raw for i in train_idx])  # [N, 42]
    lbl_mat = np.stack([examples[i].label_raw for i in train_idx])
    hw_mean = hw_mat.mean(axis=0).astype(np.float32)
    hw_std = (hw_mat.std(axis=0) + 1e-8).astype(np.float32)
    label_mean = lbl_mat.mean(axis=0).astype(np.float32)
    label_std = (lbl_mat.std(axis=0) + 1e-8).astype(np.float32)

    # accdist global std (signed-log space), sampled from train intervals.
    train_keys = list({(examples[i].bench, examples[i].interval_id) for i in train_idx})
    rng = np.random.default_rng(0)
    rng.shuffle(train_keys)
    vals = []
    for key in train_keys[:accdist_sample_intervals]:
        ref = refs[key]
        n = ref.num_records
        ad = np.memmap(ref.accdist_file, dtype="<i8", mode="r", shape=(n, ACCDIST_NM))
        take = min(n, 2048)
        sl = np.sign(ad[:take].astype(np.float64)) * np.log1p(np.abs(ad[:take].astype(np.float64)))
        vals.append(sl.reshape(-1))
    accdist_std = float(np.concatenate(vals).std()) if vals else 1.0
    if not math.isfinite(accdist_std) or accdist_std <= 0:
        accdist_std = 1.0

    return Stats(hw_mean, hw_std, label_mean, label_std, accdist_std)


def normalize_hw(example: "Example", stats: Stats) -> np.ndarray:
    """Build the full 47-dim normalized HW vector for one example."""
    return vectorize_hw(example.hw_cont_raw, example.hw_onehot,
                        stats.hw_mean, stats.hw_std)


def normalize_label(label_raw: np.ndarray, stats: Stats) -> np.ndarray:
    return (label_raw - stats.label_mean) / stats.label_std
