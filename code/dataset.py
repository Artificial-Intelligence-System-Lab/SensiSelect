import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
import math
import os
import random
from typing import Dict, Any, Optional, List
from collections import defaultdict


RFP_VECTOR_LEN = 8
RFP_GROUPS = ["int", "float", "all"]
RFP_AVG_INDEXES = [7, 15, 23]
FOOTPRINT_KEYS = ["half_cacheline", "cacheline", "dual_cacheline"]
FOOTPRINT_FIELDS = ["fp", "ifp", "gwfp", "gwifp"]


class MicroArchDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        stats: Optional[Dict[str, Any]] = None,
        force_process: bool = False,
    ):
        """
        Args:
            data_path: Path to the pickled training dataset file (list of dicts).
            stats: Optional pre-computed statistics for z-score normalization.
                   If None, statistics will be computed from the dataset.
                   Pass training set stats when loading validation/test sets.
            force_process: If True, ignore cache and re-process.
        """
        self.data_path = data_path

        # Cache filename: data_path + ".processed.pkl"
        cache_path = data_path + ".processed.pkl"
        loaded = False

        # Try to load from cache if stats is None (standard training flow)
        if stats is None and not force_process and os.path.exists(cache_path):
            print(f"Loading processed dataset from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                self.samples = cached["samples"]
                self.stats = cached["stats"]
                self.fp_len = cached["fp_len"]
                # bbv_raw stores (bbv_x, bbv_w) as lists (not tensors) to save memory
                self.bbv_raw = cached.get("bbv_raw", None)

                # Cache compatibility check: require group_id/hw_id/interval_id for response training
                if not self.samples or any(
                    k not in self.samples[0]
                    for k in [
                        "benchmark",
                        "interval_id",
                        "group_id",
                        "hw_id",
                        "rfp",
                        "gwfp",
                        "gwifp",
                    ]
                ):
                    raise ValueError(
                        "Processed cache missing group_id/hw_id/interval_id/rfp/gwfp/gwifp; will reprocess"
                    )
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        if not loaded:
            print(f"Loading raw data from {data_path}...")
            with open(data_path, "rb") as f:
                self.raw_data = pickle.load(f)

            # 自动获取 fp_len (数据集中所有样本的 fp 长度应一致)
            self.fp_len = self._infer_fp_len()

            # 计算或使用传入的归一化统计量 (z-score)
            if stats is None:
                self.stats = self._compute_statistics()
            else:
                self.stats = stats

            # 处理数据 (bbv_x/bbv_w stored separately to reduce cache size)
            print("Processing data (this may take a while)...")
            self.samples, self.bbv_raw = self._process_data()

            # Save cache if stats is None (meaning this is the 'source of truth' training set)
            if stats is None:
                print(f"Saving processed dataset to cache: {cache_path}")
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "samples": self.samples,
                            "stats": self.stats,
                            "fp_len": self.fp_len,
                            "bbv_raw": self.bbv_raw,  # Store as lists, not tensors
                        },
                        f,
                    )

            # Free memory
            del self.raw_data

    def _infer_fp_len(self) -> int:
        """从数据集推断 footprint 曲线长度"""
        if not self.raw_data:
            raise ValueError("Dataset is empty")

        # 取第一个样本的 fp 长度
        first_item = self.raw_data[0]
        fp_len = len(first_item["workload"]["fp"]["cacheline"])

        # 可选：验证所有样本的 fp 和 ifp 长度一致
        for i, item in enumerate(self.raw_data):
            wk = item["workload"]
            for field in FOOTPRINT_FIELDS:
                if field not in wk:
                    raise ValueError(
                        f"Missing workload[{field!r}] in sample {i}; rebuild raw dataset with gwfp/gwifp"
                    )
                curr_len = len(wk[field]["cacheline"])
                if curr_len != fp_len:
                    raise ValueError(
                        f"Inconsistent {field}_len: sample 0 fp has {fp_len}, "
                        f"sample {i} {field} has {curr_len}"
                    )

        return fp_len

    def _compute_statistics(self) -> Dict[str, Any]:
        """计算 z-score 归一化所需的统计量 (均值和标准差)"""
        # 收集所有原始数值用于统计
        # 注意：BBE 已经是模型输出并做过 L2 归一化，不需要再次归一化
        all_fp = []  # 展平的 fp 值 (data access footprint)
        all_ifp = []  # 展平的 ifp 值 (instruction access footprint)
        all_fp_ratio_half = []  # half/cache ratio for fp
        all_fp_ratio_cache2dual = []  # cache/dual ratio for fp
        all_ifp_ratio_half = []  # half/cache ratio for ifp
        all_ifp_ratio_cache2dual = []  # cache/dual ratio for ifp
        all_gwfp = []  # global-window data footprint
        all_gwifp = []  # global-window instruction footprint
        all_gwfp_ratio_half = []
        all_gwfp_ratio_cache2dual = []
        all_gwifp_ratio_half = []
        all_gwifp_ratio_cache2dual = []
        all_entropy = []
        all_intensity = []
        all_rfp_vecs = []
        all_hw_vecs = []
        all_labels = []

        for item in self.raw_data:
            wk = item["workload"]
            hw = item["hardware"]
            lbl = item["label"]

            # Footprint (data access) - 收集所有曲线的原始值，先做 log1p 转换
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["fp"][key]
                all_fp.extend([math.log1p(x) for x in curve])

            # Instruction Footprint (instruction access) - log1p transform
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["ifp"][key]
                all_ifp.extend([math.log1p(x) for x in curve])

            # Global-window data footprint - same representation as fp, but
            # using reuse intervals measured against prior global history.
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["gwfp"][key]
                all_gwfp.extend([math.log1p(x) for x in curve])

            # Global-window instruction footprint.
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["gwifp"][key]
                all_gwifp.extend([math.log1p(x) for x in curve])

            # FP ratio: half/cache and cache/dual (spatial locality features)
            fp_half = np.array(wk["fp"]["half_cacheline"])
            fp_cache = np.array(wk["fp"]["cacheline"])
            fp_dual = np.array(wk["fp"]["dual_cacheline"])
            all_fp_ratio_half.extend(fp_half / (fp_cache + 1e-8))
            all_fp_ratio_cache2dual.extend(fp_cache / (fp_dual + 1e-8))

            # Inst FP ratio: half/cache and cache/dual (spatial locality features)
            ifp_half = np.array(wk["ifp"]["half_cacheline"])
            ifp_cache = np.array(wk["ifp"]["cacheline"])
            ifp_dual = np.array(wk["ifp"]["dual_cacheline"])
            all_ifp_ratio_half.extend(ifp_half / (ifp_cache + 1e-8))
            all_ifp_ratio_cache2dual.extend(ifp_cache / (ifp_dual + 1e-8))

            # Global-window FP ratios
            gwfp_half = np.array(wk["gwfp"]["half_cacheline"])
            gwfp_cache = np.array(wk["gwfp"]["cacheline"])
            gwfp_dual = np.array(wk["gwfp"]["dual_cacheline"])
            all_gwfp_ratio_half.extend(gwfp_half / (gwfp_cache + 1e-8))
            all_gwfp_ratio_cache2dual.extend(gwfp_cache / (gwfp_dual + 1e-8))

            # Global-window IFP ratios
            gwifp_half = np.array(wk["gwifp"]["half_cacheline"])
            gwifp_cache = np.array(wk["gwifp"]["cacheline"])
            gwifp_dual = np.array(wk["gwifp"]["dual_cacheline"])
            all_gwifp_ratio_half.extend(gwifp_half / (gwifp_cache + 1e-8))
            all_gwifp_ratio_cache2dual.extend(gwifp_cache / (gwifp_dual + 1e-8))

            # Branch stats
            all_entropy.append(wk["ent"]["entropy"])
            all_intensity.append(wk["ent"]["intensity"])

            # Register dependency histogram
            all_rfp_vecs.append(self._extract_rfp_raw(wk))

            # Hardware (原始值，不做预处理)
            hw_vec = self._extract_hw_raw(hw)
            all_hw_vecs.append(hw_vec)

            # Labels
            label_vec = self._extract_labels_raw(lbl)
            all_labels.append(label_vec)

        # 转换为 numpy 数组计算统计量
        all_fp = np.array(all_fp, dtype=np.float32)
        all_ifp = np.array(all_ifp, dtype=np.float32)
        all_fp_ratio_half = np.array(all_fp_ratio_half, dtype=np.float32)
        all_fp_ratio_cache2dual = np.array(all_fp_ratio_cache2dual, dtype=np.float32)
        all_ifp_ratio_half = np.array(all_ifp_ratio_half, dtype=np.float32)
        all_ifp_ratio_cache2dual = np.array(all_ifp_ratio_cache2dual, dtype=np.float32)
        all_gwfp = np.array(all_gwfp, dtype=np.float32)
        all_gwifp = np.array(all_gwifp, dtype=np.float32)
        all_gwfp_ratio_half = np.array(all_gwfp_ratio_half, dtype=np.float32)
        all_gwfp_ratio_cache2dual = np.array(all_gwfp_ratio_cache2dual, dtype=np.float32)
        all_gwifp_ratio_half = np.array(all_gwifp_ratio_half, dtype=np.float32)
        all_gwifp_ratio_cache2dual = np.array(all_gwifp_ratio_cache2dual, dtype=np.float32)
        all_entropy = np.array(all_entropy, dtype=np.float32)
        all_intensity = np.array(all_intensity, dtype=np.float32)
        all_rfp_vecs = np.array(all_rfp_vecs, dtype=np.float32)
        all_hw_vecs = np.array(all_hw_vecs, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.float32)

        if all_rfp_vecs.size > 0:
            all_rfp_stats = all_rfp_vecs.copy()
            all_rfp_stats[:, RFP_AVG_INDEXES] = np.log1p(
                np.clip(all_rfp_stats[:, RFP_AVG_INDEXES], a_min=0.0, a_max=None)
            )
            rfp_mean = all_rfp_stats.mean(axis=0)
            rfp_std = all_rfp_stats.std(axis=0) + 1e-8
        else:
            rfp_mean = np.zeros(len(RFP_GROUPS) * RFP_VECTOR_LEN, dtype=np.float32)
            rfp_std = np.ones(len(RFP_GROUPS) * RFP_VECTOR_LEN, dtype=np.float32)

        # Calculate label stats
        label_mean = all_labels.mean(axis=0)
        label_std = all_labels.std(axis=0) + 1e-8

        # Empirical statistics are used for every label dimension;
        # all labels are subsequently z-score normalized.

        stats = {
            # 注意：BBE 已经 L2 归一化，不需要额外统计量
            # Footprint (data access): global stats (log1p values)
            "fp_mean": all_fp.mean().astype(np.float32),
            "fp_std": (all_fp.std() + 1e-8).astype(np.float32),
            # Instruction footprint: global statistics of log1p-transformed values.
            "ifp_mean": all_ifp.mean().astype(np.float32),
            "ifp_std": (all_ifp.std() + 1e-8).astype(np.float32),
            # Global-window footprint stats.
            "gwfp_mean": all_gwfp.mean().astype(np.float32),
            "gwfp_std": (all_gwfp.std() + 1e-8).astype(np.float32),
            "gwifp_mean": all_gwifp.mean().astype(np.float32),
            "gwifp_std": (all_gwifp.std() + 1e-8).astype(np.float32),
            # FP ratio statistics (half/cache and cache/dual)
            "fp_ratio_half_mean": all_fp_ratio_half.mean().astype(np.float32),
            "fp_ratio_half_std": (all_fp_ratio_half.std() + 1e-8).astype(np.float32),
            "fp_ratio_cache2dual_mean": all_fp_ratio_cache2dual.mean().astype(np.float32),
            "fp_ratio_cache2dual_std": (all_fp_ratio_cache2dual.std() + 1e-8).astype(np.float32),
            # Inst FP ratio statistics (half/cache and cache/dual)
            "ifp_ratio_half_mean": all_ifp_ratio_half.mean().astype(np.float32),
            "ifp_ratio_half_std": (all_ifp_ratio_half.std() + 1e-8).astype(np.float32),
            "ifp_ratio_cache2dual_mean": all_ifp_ratio_cache2dual.mean().astype(np.float32),
            "ifp_ratio_cache2dual_std": (all_ifp_ratio_cache2dual.std() + 1e-8).astype(np.float32),
            # Global-window FP ratio statistics.
            "gwfp_ratio_half_mean": all_gwfp_ratio_half.mean().astype(np.float32),
            "gwfp_ratio_half_std": (all_gwfp_ratio_half.std() + 1e-8).astype(np.float32),
            "gwfp_ratio_cache2dual_mean": all_gwfp_ratio_cache2dual.mean().astype(np.float32),
            "gwfp_ratio_cache2dual_std": (all_gwfp_ratio_cache2dual.std() + 1e-8).astype(np.float32),
            "gwifp_ratio_half_mean": all_gwifp_ratio_half.mean().astype(np.float32),
            "gwifp_ratio_half_std": (all_gwifp_ratio_half.std() + 1e-8).astype(np.float32),
            "gwifp_ratio_cache2dual_mean": all_gwifp_ratio_cache2dual.mean().astype(np.float32),
            "gwifp_ratio_cache2dual_std": (all_gwifp_ratio_cache2dual.std() + 1e-8).astype(np.float32),
            # Branch stats
            "entropy_mean": np.float32(all_entropy.mean()),
            "entropy_std": np.float32(all_entropy.std() + 1e-8),
            "intensity_mean": np.float32(all_intensity.mean()),
            "intensity_std": np.float32(all_intensity.std() + 1e-8),
            # Register dependency histogram: per-dimension stats,
            # with log1p applied to average-distance slots only.
            "rfp_mean": rfp_mean,
            "rfp_std": rfp_std,
            # Hardware: per-dimension stats
            "hw_mean": all_hw_vecs.mean(axis=0).astype(np.float32),
            "hw_std": (all_hw_vecs.std(axis=0) + 1e-8).astype(np.float32),
            # Labels: per-task stats
            "label_mean": label_mean,
            "label_std": label_std,
        }

        return stats

    def _extract_hw_raw(self, hw: Dict) -> list:
        """提取硬件配置的原始数值 (未归一化)"""
        vec = []

        # Frequency
        vec.append(hw["clock_frequency"])

        # Pipeline Widths
        vec.append(hw["fetchWidth"])
        vec.append(hw["decodeWidth"])
        vec.append(
            hw.get("renameWidth", hw["decodeWidth"])
        )  # renameWidth (may default to decodeWidth)
        vec.append(hw.get("dispatchWidth", 4))  # Defaults to 4 when absent.
        vec.append(hw["issueWidth"])
        vec.append(hw["commitWidth"])
        vec.append(hw.get("squashWidth", hw["commitWidth"]))  # squashWidth

        # Buffers (log2 transform for better distribution)
        vec.append(math.log2(hw["numROBEntries"]))
        vec.append(math.log2(hw["numIQEntries"]))
        vec.append(math.log2(hw["LQEntries"]))
        vec.append(math.log2(hw["SQEntries"]))

        # Fetch buffers
        vec.append(hw.get("fetchBufferSize", 16))  # fetchBufferSize
        vec.append(hw.get("fetchQueueSize", 16))  # fetchQueueSize

        # Branch Pred Resources
        vec.append(math.log2(hw["btb_entries"]))
        vec.append(hw["ras_entries"])

        # Caches (log2 Size in bytes)
        def parse_size_log2(s):
            if "kB" in s:
                return math.log2(int(s.replace("kB", "")) * 1024)
            if "MB" in s:
                return math.log2(int(s.replace("MB", "")) * 1024 * 1024)
            return 0.0

        vec.append(parse_size_log2(hw["l1d_size"]))
        vec.append(parse_size_log2(hw["l1i_size"]))  # L1I size
        vec.append(parse_size_log2(hw["l2_size"]))
        vec.append(hw["l1d_assoc"])
        vec.append(hw["l1i_assoc"])  # L1I associativity
        vec.append(hw["l2_assoc"])
        vec.append(hw["l1d_mshrs"])
        vec.append(hw.get("l1i_mshrs", 16))  # L1I MSHR
        vec.append(hw["l2_mshrs"])

        # FUs
        vec.append(hw["fu_int_alu"])
        vec.append(hw.get("fu_int_mult", 2))  # Integer multipliers
        vec.append(hw["fu_fp_alu"])
        vec.append(hw.get("fu_fp_mult", 2))  # FP multipliers

        # ============================================
        # Latency features (13 parameters)
        # Using log2 transform for better distribution
        # ============================================

        # Cache Latencies (cycles)
        vec.append(math.log2(hw.get("l1d_latency", 2)))  # Range [1-4]
        vec.append(math.log2(hw.get("l1i_latency", 2)))  # Range [1-4]
        vec.append(math.log2(hw.get("l2_latency", 14)))  # Range [10-20]

        # Pipeline Delays (cycles) - affects branch misprediction penalty
        vec.append(math.log2(hw.get("fetch_to_decode_delay", 1)))  # Range [1-4]
        vec.append(math.log2(hw.get("decode_to_rename_delay", 1)))  # Range [1-3]
        vec.append(math.log2(hw.get("rename_to_iew_delay", 1)))  # Range [1-3]
        vec.append(math.log2(hw.get("iew_to_commit_delay", 1)))  # Range [1-3]

        # Functional Unit Latencies (cycles)
        vec.append(math.log2(hw.get("int_alu_latency", 1)))  # Range [1-2]
        vec.append(math.log2(hw.get("int_mult_latency", 3)))  # Range [3-5]
        vec.append(math.log2(hw.get("int_div_latency", 20)))  # Range [10-30]
        vec.append(math.log2(hw.get("fp_alu_latency", 2)))  # Range [1-3]
        vec.append(math.log2(hw.get("fp_mult_latency", 4)))  # Range [4-6]
        vec.append(math.log2(hw.get("fp_div_latency", 12)))  # Range [10-30]

        return vec

    def _extract_labels_raw(self, lbl: Dict) -> list:
        """提取标签的原始数值 (对 MPKI 做 log1p 转换)
        NaN values (e.g. btb_miss_rate when no branches) are clamped to 0.0."""

        def _safe(v, default=0.0):
            """Return default if v is None or NaN."""
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return default
            return v

        cpi = _safe(lbl.get("cpi", 1.0), 1.0)
        br_mpki = math.log1p(_safe(lbl.get("branch_mpki", 0.0)))
        l1_mpki = math.log1p(_safe(lbl.get("l1d_mpki", 0.0)))
        l1i_mpki = math.log1p(_safe(lbl.get("l1i_mpki", 0.0)))
        l2_mpki = math.log1p(_safe(lbl.get("l2_mpki", 0.0)))
        rob_stall = _safe(lbl.get("rob_stall_rate", 0.0))
        lq_stall = _safe(lbl.get("lq_stall_rate", 0.0))
        sq_stall = _safe(lbl.get("sq_stall_rate", 0.0))
        l2_miss = _safe(lbl.get("l2_miss_rate", 0.0))
        btb_miss = _safe(lbl.get("btb_miss_rate", 0.0))

        return [
            cpi,
            br_mpki,
            l1_mpki,
            l1i_mpki,
            l2_mpki,
            rob_stall,
            lq_stall,
            sq_stall,
            l2_miss,
            btb_miss,
        ]

    def _extract_rfp_raw(self, wk: Dict) -> list:
        """提取寄存器依赖直方图特征，按 int/float/all 拼接为 24 维向量。"""
        rfp = wk.get("rfp") or {}
        vec = []

        for key in RFP_GROUPS:
            values = rfp.get(key)
            if values is None:
                values = [0.0] * RFP_VECTOR_LEN
            if len(values) != RFP_VECTOR_LEN:
                raise ValueError(
                    f"Invalid rfp[{key}] length: expected {RFP_VECTOR_LEN}, got {len(values)}"
                )
            vec.extend(float(x) for x in values)

        return vec

    def get_stats(self) -> Dict[str, Any]:
        """返回统计量，用于传递给验证/测试集"""
        return self.stats

    def _process_data(self):
        processed = []
        bbv_raw_list = []  # Store bbv_x/bbv_w as raw lists to save memory

        # Assign a stable integer group id per (benchmark, interval_id)
        group_map: Dict[Any, int] = {}
        next_group_id = 0

        for item in self.raw_data:
            wk = item["workload"]
            hw = item["hardware"]
            lbl = item["label"]

            benchmark = item.get("benchmark", "unknown")
            interval_id = int(item.get("interval_id", -1))
            group_key = (benchmark, interval_id)
            if group_key not in group_map:
                group_map[group_key] = next_group_id
                next_group_id += 1
            group_id = group_map[group_key]

            hw_id = int(hw.get("id", -1))

            # 1. Workload Features
            # BBE: [512] - 已经 L2 归一化，直接使用
            # BBE (Basic Block Attributes) 向量是 semanticbbv 特征，
            # 它们在生成时已经经过了正则化处理（通常是 L2 归一化），确保了数值稳定性。
            # 因此这里不需要再进行额外的标准化或归一化操作。
            bbe = torch.tensor(wk["semanticbbv"], dtype=torch.float32)

            # Footprint (data access): [3, fp_len] - log1p then z-score normalize
            fp_curves = []
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["fp"][key]
                # log1p transform
                curve = np.array([math.log1p(x) for x in curve], dtype=np.float32)
                # z-score normalize
                curve = (curve - self.stats["fp_mean"]) / self.stats["fp_std"]
                fp_curves.append(curve)
            fp_tensor = torch.tensor(np.stack(fp_curves), dtype=torch.float32)

            # Data FP: 添加 2 个 ratio 通道 (spatial locality features)
            fp_half = np.array(wk["fp"]["half_cacheline"], dtype=np.float32)
            fp_cache = np.array(wk["fp"]["cacheline"], dtype=np.float32)
            fp_dual = np.array(wk["fp"]["dual_cacheline"], dtype=np.float32)

            fp_ratio_half = fp_half / (fp_cache + 1e-8)
            fp_ratio_half = (
                fp_ratio_half - self.stats["fp_ratio_half_mean"]
            ) / self.stats["fp_ratio_half_std"]

            fp_ratio_cache2dual = fp_cache / (fp_dual + 1e-8)  # cache/dual
            fp_ratio_cache2dual = (
                fp_ratio_cache2dual - self.stats["fp_ratio_cache2dual_mean"]
            ) / self.stats["fp_ratio_cache2dual_std"]

            # HOTL-derived miss ratio approximation (temporal locality)
            # Δfp along window axis ≈ miss ratio at different cache sizes
            def _fp_slope(raw_curve):
                slope = np.diff(raw_curve)
                slope = np.append(slope, 0.0)
                slope = np.array([math.log1p(abs(x)) for x in slope], dtype=np.float32)
                return (slope - self.stats["fp_mean"]) / self.stats["fp_std"]

            fp_slope_half = _fp_slope(fp_half)       # sub-cacheline MR (sector cache)
            fp_slope_cache = _fp_slope(fp_cache)      # standard MR (L1/L2 miss rate)
            fp_slope_dual = _fp_slope(fp_dual)        # prefetch-granularity MR

            # 拼接: [3, fp_len] → [8, fp_len]
            fp_tensor = torch.cat(
                [
                    fp_tensor,
                    torch.tensor(fp_ratio_half).unsqueeze(0),
                    torch.tensor(fp_ratio_cache2dual).unsqueeze(0),
                    torch.tensor(fp_slope_half).unsqueeze(0),
                    torch.tensor(fp_slope_cache).unsqueeze(0),
                    torch.tensor(fp_slope_dual).unsqueeze(0),
                ],
                dim=0,
            )

            # Instruction Footprint (instruction access): [3, fp_len] - log1p then z-score normalize
            ifp_curves = []
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["ifp"][key]
                # log1p transform
                curve = np.array([math.log1p(x) for x in curve], dtype=np.float32)
                # z-score normalize
                curve = (curve - self.stats["ifp_mean"]) / self.stats["ifp_std"]
                ifp_curves.append(curve)
            ifp_tensor = torch.tensor(np.stack(ifp_curves), dtype=torch.float32)

            # Inst FP: 添加 2 个 ratio 通道 (spatial locality features)
            ifp_half = np.array(wk["ifp"]["half_cacheline"], dtype=np.float32)
            ifp_cache = np.array(wk["ifp"]["cacheline"], dtype=np.float32)
            ifp_dual = np.array(wk["ifp"]["dual_cacheline"], dtype=np.float32)

            ifp_ratio_half = ifp_half / (ifp_cache + 1e-8)
            ifp_ratio_half = (
                ifp_ratio_half - self.stats["ifp_ratio_half_mean"]
            ) / self.stats["ifp_ratio_half_std"]

            ifp_ratio_cache2dual = ifp_cache / (ifp_dual + 1e-8)  # cache/dual
            ifp_ratio_cache2dual = (
                ifp_ratio_cache2dual - self.stats["ifp_ratio_cache2dual_mean"]
            ) / self.stats["ifp_ratio_cache2dual_std"]

            # HOTL-derived miss ratio approximation for instruction cache
            def _ifp_slope(raw_curve):
                slope = np.diff(raw_curve)
                slope = np.append(slope, 0.0)
                slope = np.array([math.log1p(abs(x)) for x in slope], dtype=np.float32)
                return (slope - self.stats["ifp_mean"]) / self.stats["ifp_std"]

            ifp_slope_half = _ifp_slope(ifp_half)
            ifp_slope_cache = _ifp_slope(ifp_cache)
            ifp_slope_dual = _ifp_slope(ifp_dual)

            # 拼接: [3, fp_len] → [8, fp_len]
            ifp_tensor = torch.cat(
                [
                    ifp_tensor,
                    torch.tensor(ifp_ratio_half).unsqueeze(0),
                    torch.tensor(ifp_ratio_cache2dual).unsqueeze(0),
                    torch.tensor(ifp_slope_half).unsqueeze(0),
                    torch.tensor(ifp_slope_cache).unsqueeze(0),
                    torch.tensor(ifp_slope_dual).unsqueeze(0),
                ],
                dim=0,
            )

            # Global-window Data FP: same 8-channel transform as local fp,
            # but statistics are kept separate because the distribution differs.
            gwfp_curves = []
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["gwfp"][key]
                curve = np.array([math.log1p(x) for x in curve], dtype=np.float32)
                curve = (curve - self.stats["gwfp_mean"]) / self.stats["gwfp_std"]
                gwfp_curves.append(curve)
            gwfp_tensor = torch.tensor(np.stack(gwfp_curves), dtype=torch.float32)

            gwfp_half = np.array(wk["gwfp"]["half_cacheline"], dtype=np.float32)
            gwfp_cache = np.array(wk["gwfp"]["cacheline"], dtype=np.float32)
            gwfp_dual = np.array(wk["gwfp"]["dual_cacheline"], dtype=np.float32)

            gwfp_ratio_half = gwfp_half / (gwfp_cache + 1e-8)
            gwfp_ratio_half = (
                gwfp_ratio_half - self.stats["gwfp_ratio_half_mean"]
            ) / self.stats["gwfp_ratio_half_std"]

            gwfp_ratio_cache2dual = gwfp_cache / (gwfp_dual + 1e-8)
            gwfp_ratio_cache2dual = (
                gwfp_ratio_cache2dual - self.stats["gwfp_ratio_cache2dual_mean"]
            ) / self.stats["gwfp_ratio_cache2dual_std"]

            def _gwfp_slope(raw_curve):
                slope = np.diff(raw_curve)
                slope = np.append(slope, 0.0)
                slope = np.array([math.log1p(abs(x)) for x in slope], dtype=np.float32)
                return (slope - self.stats["gwfp_mean"]) / self.stats["gwfp_std"]

            gwfp_slope_half = _gwfp_slope(gwfp_half)
            gwfp_slope_cache = _gwfp_slope(gwfp_cache)
            gwfp_slope_dual = _gwfp_slope(gwfp_dual)

            gwfp_tensor = torch.cat(
                [
                    gwfp_tensor,
                    torch.tensor(gwfp_ratio_half).unsqueeze(0),
                    torch.tensor(gwfp_ratio_cache2dual).unsqueeze(0),
                    torch.tensor(gwfp_slope_half).unsqueeze(0),
                    torch.tensor(gwfp_slope_cache).unsqueeze(0),
                    torch.tensor(gwfp_slope_dual).unsqueeze(0),
                ],
                dim=0,
            )

            # Global-window Instruction FP.
            gwifp_curves = []
            for key in ["half_cacheline", "cacheline", "dual_cacheline"]:
                curve = wk["gwifp"][key]
                curve = np.array([math.log1p(x) for x in curve], dtype=np.float32)
                curve = (curve - self.stats["gwifp_mean"]) / self.stats["gwifp_std"]
                gwifp_curves.append(curve)
            gwifp_tensor = torch.tensor(np.stack(gwifp_curves), dtype=torch.float32)

            gwifp_half = np.array(wk["gwifp"]["half_cacheline"], dtype=np.float32)
            gwifp_cache = np.array(wk["gwifp"]["cacheline"], dtype=np.float32)
            gwifp_dual = np.array(wk["gwifp"]["dual_cacheline"], dtype=np.float32)

            gwifp_ratio_half = gwifp_half / (gwifp_cache + 1e-8)
            gwifp_ratio_half = (
                gwifp_ratio_half - self.stats["gwifp_ratio_half_mean"]
            ) / self.stats["gwifp_ratio_half_std"]

            gwifp_ratio_cache2dual = gwifp_cache / (gwifp_dual + 1e-8)
            gwifp_ratio_cache2dual = (
                gwifp_ratio_cache2dual - self.stats["gwifp_ratio_cache2dual_mean"]
            ) / self.stats["gwifp_ratio_cache2dual_std"]

            def _gwifp_slope(raw_curve):
                slope = np.diff(raw_curve)
                slope = np.append(slope, 0.0)
                slope = np.array([math.log1p(abs(x)) for x in slope], dtype=np.float32)
                return (slope - self.stats["gwifp_mean"]) / self.stats["gwifp_std"]

            gwifp_slope_half = _gwifp_slope(gwifp_half)
            gwifp_slope_cache = _gwifp_slope(gwifp_cache)
            gwifp_slope_dual = _gwifp_slope(gwifp_dual)

            gwifp_tensor = torch.cat(
                [
                    gwifp_tensor,
                    torch.tensor(gwifp_ratio_half).unsqueeze(0),
                    torch.tensor(gwifp_ratio_cache2dual).unsqueeze(0),
                    torch.tensor(gwifp_slope_half).unsqueeze(0),
                    torch.tensor(gwifp_slope_cache).unsqueeze(0),
                    torch.tensor(gwifp_slope_dual).unsqueeze(0),
                ],
                dim=0,
            )

            # Branch Stats: [2] -> [Entropy, Intensity] - z-score normalize
            entropy_norm = (
                wk["ent"]["entropy"] - self.stats["entropy_mean"]
            ) / self.stats["entropy_std"]
            intensity_norm = (
                wk["ent"]["intensity"] - self.stats["intensity_mean"]
            ) / self.stats["intensity_std"]
            br_stats = torch.tensor([entropy_norm, intensity_norm], dtype=torch.float32)

            # Register dependency histogram: 24-D
            rfp_tensor = self._vectorize_rfp(wk)

            # 2. Hardware Vectorization - z-score normalize
            hw_vec = self._vectorize_hardware(hw)

            # 3. Labels (Multi-task) - z-score normalize
            labels = self._vectorize_labels(lbl)

            # Store bbv_x/bbv_w as raw lists (not tensors) to save memory in cache
            # Will be converted to tensors in __getitem__
            bbv_x_raw = wk.get("bbv_x", None)
            bbv_w_raw = wk.get("bbv_w", None)
            bbv_raw_list.append((bbv_x_raw, bbv_w_raw))

            processed.append(
                {
                    "bbe": bbe,
                    "fp": fp_tensor,
                    "ifp": ifp_tensor,
                    "gwfp": gwfp_tensor,
                    "gwifp": gwifp_tensor,
                    "br": br_stats,
                    "rfp": rfp_tensor,
                    "hw": hw_vec,
                    "labels": labels,
                    "benchmark": benchmark,  # for Group Split
                    "interval_id": interval_id,
                    "group_id": group_id,
                    "hw_id": hw_id,
                }
            )

        print(f"Processed {len(processed)} samples.")
        return processed, bbv_raw_list

    def _vectorize_hardware(self, hw):
        """将硬件 Config 字典转换为固定长度向量，并使用 z-score 归一化"""
        vec = self._extract_hw_raw(hw)
        vec = np.array(vec, dtype=np.float32)

        # z-score normalize
        vec = (vec - self.stats["hw_mean"]) / self.stats["hw_std"]

        # One-hot categorical features (不做 z-score)
        onehot = []

        # BP Type: BiMode, Tournament, LTAGE
        bp_map = {"BiModeBP": [1, 0, 0], "TournamentBP": [0, 1, 0], "LTAGE": [0, 0, 1]}
        onehot.extend(bp_map.get(hw["bp_type"], [0, 0, 0]))

        # Prefetcher: Null, Stride
        pf_map = {"Null": [1, 0], "StridePrefetcher": [0, 1]}
        onehot.extend(pf_map.get(hw["l1d_prefetcher"], [0, 0]))

        # 合并连续特征和 one-hot 特征
        full_vec = np.concatenate([vec, np.array(onehot, dtype=np.float32)])

        return torch.tensor(full_vec, dtype=torch.float32)

    def _vectorize_rfp(self, wk: Dict) -> torch.Tensor:
        """将寄存器依赖直方图转换为固定长度向量，并做 per-dim 标准化。"""
        vec = np.array(self._extract_rfp_raw(wk), dtype=np.float32)
        vec_stats = vec.copy()
        vec_stats[RFP_AVG_INDEXES] = np.log1p(
            np.clip(vec_stats[RFP_AVG_INDEXES], a_min=0.0, a_max=None)
        )
        normalized = (vec_stats - self.stats["rfp_mean"]) / self.stats["rfp_std"]
        return torch.tensor(normalized, dtype=torch.float32)

    def _vectorize_labels(self, lbl):
        """提取并使用 z-score 标准化标签

        All labels (indices 0-9) are z-score normalized for consistent loss scales.
        """
        raw = np.array(self._extract_labels_raw(lbl), dtype=np.float32)

        # z-score normalize all labels
        normalized = (raw - self.stats["label_mean"]) / self.stats["label_std"]

        return torch.tensor(normalized, dtype=torch.float32)

    def denormalize_labels(self, normalized_labels: torch.Tensor) -> torch.Tensor:
        """将归一化的标签还原为原始值 (用于评估)

        All labels (indices 0-9) are z-score normalized.
        """
        mean = torch.tensor(self.stats["label_mean"], dtype=torch.float32)
        std = torch.tensor(self.stats["label_std"], dtype=torch.float32)
        return normalized_labels * std + mean

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()

        # Keep original dataset index for response_pair sampling (avoid pairing with itself)
        sample["sample_idx"] = int(idx)

        # Lazily convert bbv_x/bbv_w from raw lists to tensors
        if self.bbv_raw is not None and idx < len(self.bbv_raw):
            bbv_x_raw, bbv_w_raw = self.bbv_raw[idx]
            if bbv_x_raw is not None and len(bbv_x_raw) > 0:
                sample["bbv_x"] = torch.tensor(bbv_x_raw, dtype=torch.float32)
                sample["bbv_w"] = torch.tensor(
                    bbv_w_raw, dtype=torch.float32
                ).unsqueeze(-1)
            else:
                sample["bbv_x"] = None
                sample["bbv_w"] = None
        else:
            sample["bbv_x"] = None
            sample["bbv_w"] = None

        return sample

    def get_hw_dim(self):
        """返回硬件向量维度 (连续特征 + one-hot 特征)"""
        return self.samples[0]["hw"].shape[0]

    def get_fp_len(self):
        """返回 footprint 曲线长度"""
        return self.fp_len

    def get_rfp_dim(self):
        """返回寄存器依赖直方图向量维度"""
        return self.samples[0]["rfp"].shape[0]

    def get_label_dim(self):
        """返回标签维度"""
        return self.samples[0]["labels"].shape[0]

    def get_benchmarks(self) -> List[str]:
        """返回所有样本的benchmark列表（用于Group Split，避免遍历__getitem__）"""
        return [s["benchmark"] for s in self.samples]

    def get_group_ids(self) -> List[int]:
        """返回所有样本的group_id列表（用于同interval内的硬件响应训练）"""
        return [int(s.get("group_id", -1)) for s in self.samples]

    def get_hw_ids(self) -> List[int]:
        """返回所有样本的hw_id列表（用于选择anchor/baseline硬件配置）"""
        return [int(s.get("hw_id", -1)) for s in self.samples]


def collate_fn(batch):
    """
    Custom collate function to handle variable length BBV inputs.
    Pads bbv_x and bbv_w to the maximum length in the batch.
    Optimized using torch.nn.utils.rnn.pad_sequence.

    Returns:
        bbe: [Batch, 512] - Pre-computed semantic embedding
        fp: [Batch, 8, fp_len] - Data footprint
        ifp: [Batch, 8, fp_len] - Instruction footprint
        gwfp: [Batch, 8, fp_len] - Global-window data footprint
        gwifp: [Batch, 8, fp_len] - Global-window instruction footprint
        br: [Batch, 2] - Branch stats
        rfp: available in sample dict but omitted here for backward compatibility
        hw: [Batch, hw_dim] - Hardware config
        labels: [Batch, label_dim] - Target labels
        bbv_x: [Batch, max_n_bbs, 128] or None - Raw BB embeddings (padded)
        bbv_w: [Batch, max_n_bbs, 1] or None - BB weights (padded)
        bbv_mask: [Batch, max_n_bbs] or None - Padding mask (True = valid, False = padded)
        group_id: [Batch] - Group id for same (benchmark, interval_id)
        hw_id: [Batch] - Hardware id
        sample_idx: [Batch] - Original dataset index
    """
    # Extract fixed-size fields
    bbe_list = [item["bbe"] for item in batch]
    fp_list = [item["fp"] for item in batch]
    ifp_list = [item["ifp"] for item in batch]
    gwfp_list = [item["gwfp"] for item in batch]
    gwifp_list = [item["gwifp"] for item in batch]
    br_list = [item["br"] for item in batch]
    hw_list = [item["hw"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    group_id_list = [int(item.get("group_id", -1)) for item in batch]
    hw_id_list = [int(item.get("hw_id", -1)) for item in batch]
    sample_idx_list = [int(item.get("sample_idx", -1)) for item in batch]

    # Stack fixed-size fields
    bbe = torch.stack(bbe_list)
    fp = torch.stack(fp_list)
    ifp = torch.stack(ifp_list)
    gwfp = torch.stack(gwfp_list)
    gwifp = torch.stack(gwifp_list)
    br = torch.stack(br_list)
    hw = torch.stack(hw_list)
    labels = torch.stack(labels_list)

    group_id = torch.tensor(group_id_list, dtype=torch.long)
    hw_id = torch.tensor(hw_id_list, dtype=torch.long)
    sample_idx = torch.tensor(sample_idx_list, dtype=torch.long)

    # Handle variable-length BBV inputs (for trainable SetTransformer)
    bbv_x = None
    bbv_w = None
    bbv_mask = None

    # Check if bbv_x is present in the first sample and is not None/empty
    first_bbv_x = batch[0].get("bbv_x")
    has_bbv = first_bbv_x is not None and len(first_bbv_x) > 0

    if has_bbv:
        # Extract variable-length BBV data
        bbv_x_list = [item["bbv_x"] for item in batch]
        bbv_w_list = [item["bbv_w"] for item in batch]

        # Get sequence lengths for mask creation
        lengths = [x.shape[0] for x in bbv_x_list]
        max_len = max(lengths)

        # Pad sequences using pad_sequence (batch_first=True)
        # pad_sequence expects [seq_len, ...] tensors, so we need to handle this
        bbv_x = pad_sequence(
            bbv_x_list, batch_first=True, padding_value=0.0
        )  # [Batch, max_len, 128]
        bbv_w = pad_sequence(
            bbv_w_list, batch_first=True, padding_value=0.0
        )  # [Batch, max_len, 1]

        # Create attention mask: True for valid positions, False for padded
        batch_size = len(batch)
        bbv_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            bbv_mask[i, :length] = True

    return (
        bbe,
        fp,
        ifp,
        gwfp,
        gwifp,
        br,
        hw,
        labels,
        bbv_x,
        bbv_w,
        bbv_mask,
        group_id,
        hw_id,
        sample_idx,
    )


def collate_fn_with_rfp(batch):
    """Backward-compatible extension of collate_fn that also returns batched rfp."""
    base = collate_fn(batch)
    rfp = torch.stack([item["rfp"] for item in batch])
    return base[:6] + (rfp,) + base[6:]


class GroupWiseSampler(Sampler):
    """
    Sampler that ensures each batch contains K samples from the same group (interval).

    This allows the model to learn:
    1. How the same workload performs across different hardware configurations
    2. The hardware sensitivity of each workload

    Args:
        dataset: MicroArchDataset instance
        batch_size: Total batch size
        k_per_group: Number of samples from the same group per batch (default: 8)
        shuffle: Whether to shuffle groups and samples within groups (default: True)
    """

    def __init__(
        self, dataset, batch_size: int = 64, k_per_group: int = 8, shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k_per_group = min(k_per_group, batch_size // 2)
        self.shuffle = shuffle

        # Build group -> indices mapping
        self.group_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            group_id = sample.get("group_id", -1)
            self.group_indices[group_id].append(idx)

        # Filter groups with enough samples
        self.valid_groups = [
            g
            for g, indices in self.group_indices.items()
            if len(indices) >= self.k_per_group
        ]

        # Build hw_id -> indices mapping (for hw-level sampling)
        self.hw_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            hw_id = sample.get("hw_id", -1)
            self.hw_indices[hw_id].append(idx)

        self.valid_hws = list(self.hw_indices.keys())

        print(
            f"GroupWiseSampler: {len(self.valid_groups)} groups with >= {self.k_per_group} samples"
        )
        print(f"GroupWiseSampler: {len(self.valid_hws)} unique hardware configs")

    def __iter__(self):
        # Get all valid indices
        all_indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(all_indices)

        batches = []
        used_indices = set()

        # Strategy: Mix of group-wise batches and random batches
        # 50% group-wise (same interval, different hw)
        # 30% hw-wise (same hw, different intervals)
        # 20% random (for diversity)

        n_group_batches = len(all_indices) // self.batch_size // 2
        n_hw_batches = len(all_indices) // self.batch_size * 3 // 10
        n_random_batches = (
            len(all_indices) // self.batch_size - n_group_batches - n_hw_batches
        )

        # Generate group-wise batches
        group_list = self.valid_groups.copy()
        if self.shuffle:
            random.shuffle(group_list)

        for _ in range(n_group_batches):
            if not group_list:
                break
            group_id = group_list.pop()
            group_samples = self.group_indices[group_id].copy()
            if self.shuffle:
                random.shuffle(group_samples)

            # Take k_per_group samples from this group
            batch = group_samples[: self.k_per_group]

            # Fill remaining with random samples
            remaining = self.batch_size - len(batch)
            fill_count = 0
            for idx in all_indices:
                if idx not in batch and idx not in used_indices:
                    batch.append(idx)
                    fill_count += 1
                    if fill_count >= remaining:
                        break

            if len(batch) == self.batch_size:
                batches.append(batch)
                used_indices.update(batch)

        # Generate hw-wise batches
        hw_list = self.valid_hws.copy()
        if self.shuffle:
            random.shuffle(hw_list)

        for _ in range(n_hw_batches):
            if not hw_list:
                break
            hw_id = hw_list.pop()
            hw_samples = self.hw_indices[hw_id].copy()
            if self.shuffle:
                random.shuffle(hw_samples)

            # Take samples from this hw config
            batch = hw_samples[: self.batch_size]

            if len(batch) == self.batch_size:
                batches.append(batch)
                used_indices.update(batch)

        # Generate random batches for remaining
        remaining_indices = [i for i in all_indices if i not in used_indices]
        if self.shuffle:
            random.shuffle(remaining_indices)

        for i in range(0, len(remaining_indices), self.batch_size):
            batch = remaining_indices[i : i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
