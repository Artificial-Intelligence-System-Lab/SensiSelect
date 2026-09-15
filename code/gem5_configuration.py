import random
import json
import math
import copy

# ==========================================
# 1. 设计空间定义 (Design Space Definition)
# ==========================================
DESIGN_SPACE = {
    # 主频范围 (GHz)
    # 频率越高，DRAM 的物理延迟（ns）转换成的 Cycle 数就越多，内存墙效应越明显
    "frequencies": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    # 流水线宽度 (前端 -> 后端)
    "widths": [2, 3, 4, 6, 8, 12],
    # 乱序执行资源
    "rob_range": (32, 512),  # Reorder Buffer
    "lq_range": (10, 128),  # Load Queue
    "sq_range": (10, 128),  # Store Queue
    "iq_range": (16, 128),  # Instruction Queue
    # 执行单元 (Functional Units)
    "fu_int_range": (2, 8),
    "fu_fp_range": (1, 6),
    # 乘法器数量范围
    "fu_int_mult_range": (1, 4),
    "fu_fp_mult_range": (1, 4),
    # 内存端口数量
    "fu_read_port_range": (1, 4),
    "fu_write_port_range": (1, 4),
    # 分支预测 (Branch Prediction)
    "bp_types": ["BiModeBP", "TournamentBP", "LTAGE"],
    "btb_entries": [256, 512, 1024, 2048, 4096, 8192],
    "ras_entries": [8, 16, 32, 48],
    # 缓存层级 (Memory Hierarchy)
    "l1_sizes": ["16kB", "32kB", "64kB", "128kB"],
    "l1_assocs": [4, 8],
    "l2_sizes": ["256kB", "512kB", "1MB", "2MB", "4MB"],
    "l2_assocs": [8, 16],
    # L1I/L1D/L2 关联度
    "l1i_sizes": ["16kB", "32kB", "64kB", "128kB"],  # L1I 独立大小
    "l1i_assocs": [4, 8, 16],
    "l1d_assocs": [4, 8, 16],
    "l2_assocs_extended": [4, 8, 16, 32],  # L2 关联度扩展
    # 取指缓冲区
    # 注意: 必须是 cache block size (64) 的约数
    "fetch_buffer_range": (8, 64),
    "fetch_queue_range": (8, 64),
    "fetch_buffer_sizes": [8, 16, 32, 64],  # 64 的约数
    "fetch_queue_sizes": [8, 16, 32, 64],  # 64 的约数
    # 预取器与并发度 (MLP)
    "mshrs_l1": [4, 8, 16, 32],
    "mshrs_l2": [8, 16, 32, 48],
    "mshrs_l1i": [8, 16, 32, 64],  # L1I MSHR
    "prefetchers": ["Null", "StridePrefetcher"],
    # Latencies
    "l1_latencies": [1, 2, 3, 4],
    "l2_latencies": [10, 12, 14, 16, 20],
    # Pipeline Depths / Delays (影响分支预测惩罚)
    "fetch_to_decode_delays": [1, 2, 3, 4],
    "decode_to_rename_delays": [1, 2, 3],
    "rename_to_iew_delays": [1, 2, 3],
    "iew_to_commit_delays": [1, 2, 3],
    # Functional Unit Latencies
    "int_alu_latencies": [1, 2],  # Simple Int Ops
    "int_mult_latencies": [3, 4, 5],  # Integer Multiplication
    "int_div_latencies": [10, 20, 30],  # Integer Division
    "fp_alu_latencies": [1, 2, 3],  # FP Add/Sub
    "fp_mult_latencies": [4, 5, 6],  # FP Mult
    "fp_div_latencies": [10, 15, 20, 30],  # FP Div
}


# ==========================================
# 2. 辅助工具函数
# ==========================================
def log_uniform_sample(min_val, max_val):
    """在对数尺度上采样"""
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    return int(math.exp(random.uniform(log_min, log_max)))


def get_aligned_queues(rob_size):
    """生成符合逻辑的队列大小"""
    iq = int(rob_size * random.uniform(0.4, 0.7))
    lq = int(rob_size * random.uniform(0.25, 0.4))
    sq = int(rob_size * random.uniform(0.25, 0.4))
    return max(16, iq), max(8, lq), max(8, sq)


def validate_and_fix_config(cfg):
    """
    验证配置的合理性，修复可能导致 gem5 崩溃的极端配置。

    主要检查：
    1. 流水线宽度不平衡（前端 >> 后端）
    2. LSQ 太小导致频繁阻塞
    3. 内存端口与 LSQ 的匹配
    """
    # 获取流水线宽度
    frontend_width = max(
        cfg.get("fetchWidth", 4),
        cfg.get("decodeWidth", 4),
        cfg.get("renameWidth", 4),
    )
    backend_width = min(
        cfg.get("issueWidth", 4),
        cfg.get("commitWidth", 4),
    )

    # 规则 1: 后端宽度至少是前端的 1/3
    # 这防止极端的 decodeWidth=12, commitWidth=2 配置
    min_backend = max(2, (frontend_width + 2) // 3)
    if backend_width < min_backend:
        cfg["issueWidth"] = max(cfg.get("issueWidth", 4), min_backend)
        cfg["commitWidth"] = max(cfg.get("commitWidth", 4), min_backend)
        cfg["squashWidth"] = cfg["commitWidth"]

    # 规则 2: dispatchWidth 应该与 renameWidth 匹配
    cfg["dispatchWidth"] = cfg.get("renameWidth", cfg.get("dispatchWidth", 4))

    # 规则 3: LSQ 大小应该与流水线宽度匹配
    # 最小 LSQ = 前端宽度 * 4 (至少能容纳 4 周期的 load/store)
    min_lsq = max(8, frontend_width * 4)
    if cfg.get("LQEntries", 32) < min_lsq:
        cfg["LQEntries"] = min_lsq
    if cfg.get("SQEntries", 32) < min_lsq:
        cfg["SQEntries"] = min_lsq

    # 规则 4: 内存端口数量应该合理
    # 读端口至少 1，最多不超过 issueWidth
    issue_width = cfg.get("issueWidth", 4)
    cfg["fu_read_port"] = max(1, min(cfg.get("fu_read_port", 2), issue_width))
    cfg["fu_write_port"] = max(1, min(cfg.get("fu_write_port", 2), issue_width))

    # 规则 5: ROB 大小应该与流水线宽度匹配
    # 最小 ROB = 前端宽度 * 16 (至少能容纳 16 周期的指令)
    min_rob = max(32, frontend_width * 16)
    if cfg.get("numROBEntries", 128) < min_rob:
        cfg["numROBEntries"] = min_rob

    return cfg


def create_template(name):
    """创建一个默认配置模板"""
    return {
        "name": name,
        "description": "Default Template",
        "clock_frequency": 2.0,  # Default 2GHz
        # Pipeline
        "fetchWidth": 4,
        "decodeWidth": 4,
        "renameWidth": 4,
        "dispatchWidth": 4,
        "issueWidth": 4,
        "commitWidth": 4,
        "squashWidth": 4,
        # Buffers
        "numROBEntries": 128,
        "numIQEntries": 64,
        "LQEntries": 32,
        "SQEntries": 32,
        "fetchBufferSize": 16,
        "fetchQueueSize": 16,
        # Execution
        "fu_int_alu": 4,
        "fu_int_mult": 2,
        "fu_fp_alu": 2,
        "fu_fp_mult": 2,
        "fu_read_port": 2,  # 内存读端口
        "fu_write_port": 2,  # 内存写端口
        # Branch Prediction
        "bp_type": "TournamentBP",
        "btb_entries": 4096,
        "ras_entries": 16,
        # Memory
        "l1d_size": "32kB",
        "l1d_assoc": 8,
        "l1d_mshrs": 16,
        "l1d_latency": 2,
        "l1i_size": "32kB",
        "l1i_assoc": 8,
        "l1i_mshrs": 16,
        "l1i_latency": 2,
        "l2_size": "1MB",
        "l2_assoc": 16,
        "l2_mshrs": 20,
        "l2_latency": 14,
        "l1d_prefetcher": "StridePrefetcher",
    }


# ==========================================
# 3. 策略 A: 平衡设计 (Balanced Tiers) - 48个
# ==========================================
def generate_balanced(count):
    configs = []
    # 定义4个层级：(Freq, Width, ROB, BTB, L1, L2, L1Lat, L2Lat)
    levels = [
        (1.0, 2, 64, 512, "16kB", "512kB", 1, 10),  # Low-end / IoT
        (2.5, 4, 128, 2048, "32kB", "1MB", 2, 12),  # Mid-range / Desktop
        (3.2, 6, 192, 4096, "48kB", "2MB", 3, 14),  # High-end / Laptop
        (4.0, 8, 256, 8192, "64kB", "4MB", 4, 16),  # Server / HPC
    ]

    for i in range(count):
        lvl = random.uniform(0, 3)
        base_idx = int(round(lvl))
        f_base, w_base, rob_base, btb_base, l1_base, l2_base, l1_lat, l2_lat = levels[
            base_idx
        ]

        cfg = create_template(f"Balanced_{i}")
        cfg["description"] = f"Balanced Tier {base_idx} Design"

        # 频率浮动
        freq_options = [
            f for f in DESIGN_SPACE["frequencies"] if abs(f - f_base) <= 1.0
        ]
        cfg["clock_frequency"] = random.choice(freq_options) if freq_options else f_base

        # 宽度微扰
        width = max(2, min(8, w_base + random.randint(-1, 1)))
        for k in [
            "fetchWidth",
            "decodeWidth",
            "renameWidth",
            "dispatchWidth",
            "issueWidth",
            "commitWidth",
            "squashWidth",
        ]:
            cfg[k] = width

        # 资源微扰
        cfg["numROBEntries"] = int(rob_base * random.uniform(0.8, 1.2))
        cfg["numIQEntries"], cfg["LQEntries"], cfg["SQEntries"] = get_aligned_queues(
            cfg["numROBEntries"]
        )

        # BTB微扰
        try:
            btb_idx = DESIGN_SPACE["btb_entries"].index(btb_base)
            new_idx = max(
                0,
                min(
                    len(DESIGN_SPACE["btb_entries"]) - 1,
                    btb_idx + random.randint(-1, 1),
                ),
            )
            cfg["btb_entries"] = DESIGN_SPACE["btb_entries"][new_idx]
        except:
            cfg["btb_entries"] = btb_base

        cfg["bp_type"] = "BiModeBP" if width <= 4 else "LTAGE"
        cfg["ras_entries"] = 16 if base_idx < 2 else 32

        cfg["l1d_size"] = l1_base
        cfg["l2_size"] = l2_base
        # 添加 Cache Latency
        cfg["l1d_latency"] = l1_lat
        cfg["l1i_latency"] = l1_lat
        cfg["l2_latency"] = l2_lat

        # Pipeline Delays (scale with tier: higher tier = deeper pipeline)
        cfg["fetch_to_decode_delay"] = 1 + base_idx  # 1-4
        cfg["decode_to_rename_delay"] = 1 + (base_idx // 2)  # 1-2
        cfg["rename_to_iew_delay"] = 1 + (base_idx // 2)  # 1-2
        cfg["iew_to_commit_delay"] = 1 + (base_idx // 2)  # 1-2

        # FU Latencies (default values, slight variation by tier)
        cfg["int_alu_latency"] = 1
        cfg["int_mult_latency"] = 3 + (base_idx // 2)  # 3-4
        cfg["int_div_latency"] = 20
        cfg["fp_alu_latency"] = 1 + (base_idx // 2)  # 1-2
        cfg["fp_mult_latency"] = 4 + (base_idx // 2)  # 4-5
        cfg["fp_div_latency"] = 12

        cfg["fu_int_alu"] = width
        cfg["fu_fp_alu"] = max(2, width // 2)

        # 内存端口 (与宽度相关)
        cfg["fu_read_port"] = max(1, min(width // 2, 4))
        cfg["fu_write_port"] = max(1, min(width // 2, 4))

        # 验证并修复配置
        cfg = validate_and_fix_config(cfg)

        configs.append(cfg)
    return configs


# ==========================================
# 4. 策略 B: 随机探索 (Random LHS) - 40个
# ==========================================
def generate_random(count):
    configs = []
    for i in range(count):
        cfg = create_template(f"Random_{i}")
        cfg["description"] = "Randomized Exploration"

        # 随机频率
        cfg["clock_frequency"] = random.choice(DESIGN_SPACE["frequencies"])

        # 随机宽度
        fetch = random.choice(DESIGN_SPACE["widths"])
        cfg["fetchWidth"] = fetch
        cfg["decodeWidth"] = fetch
        cfg["renameWidth"] = fetch
        cfg["issueWidth"] = random.choice(
            [w for w in DESIGN_SPACE["widths"] if w <= fetch]
        )
        cfg["commitWidth"] = cfg["issueWidth"]
        cfg["squashWidth"] = cfg["commitWidth"]

        # 随机资源
        cfg["numROBEntries"] = log_uniform_sample(40, 512)
        cfg["numIQEntries"], cfg["LQEntries"], cfg["SQEntries"] = get_aligned_queues(
            cfg["numROBEntries"]
        )

        # 随机 FU
        cfg["fu_int_alu"] = random.randint(2, 6)
        cfg["fu_fp_alu"] = random.randint(1, 4)

        # 随机分支 & 内存
        cfg["bp_type"] = random.choice(DESIGN_SPACE["bp_types"])
        cfg["btb_entries"] = random.choice(DESIGN_SPACE["btb_entries"])
        cfg["ras_entries"] = random.choice(DESIGN_SPACE["ras_entries"])

        cfg["l1d_size"] = random.choice(DESIGN_SPACE["l1_sizes"])
        cfg["l2_size"] = random.choice(DESIGN_SPACE["l2_sizes"])
        cfg["l1d_prefetcher"] = random.choice(DESIGN_SPACE["prefetchers"])
        cfg["l1d_mshrs"] = random.choice(DESIGN_SPACE["mshrs_l1"])
        cfg["l2_mshrs"] = random.choice(DESIGN_SPACE["mshrs_l2"])

        # 随机 Latency
        cfg["l1d_latency"] = random.choice(DESIGN_SPACE["l1_latencies"])
        cfg["l1i_latency"] = random.choice(DESIGN_SPACE["l1_latencies"])
        cfg["l2_latency"] = random.choice(DESIGN_SPACE["l2_latencies"])

        # 随机 Pipeline Delays
        cfg["fetch_to_decode_delay"] = random.choice(
            DESIGN_SPACE["fetch_to_decode_delays"]
        )
        cfg["decode_to_rename_delay"] = random.choice(
            DESIGN_SPACE["decode_to_rename_delays"]
        )
        cfg["rename_to_iew_delay"] = random.choice(DESIGN_SPACE["rename_to_iew_delays"])
        cfg["iew_to_commit_delay"] = random.choice(DESIGN_SPACE["iew_to_commit_delays"])

        # 随机 FU Latencies
        cfg["int_alu_latency"] = random.choice(DESIGN_SPACE["int_alu_latencies"])
        cfg["int_mult_latency"] = random.choice(DESIGN_SPACE["int_mult_latencies"])
        cfg["int_div_latency"] = random.choice(DESIGN_SPACE["int_div_latencies"])
        cfg["fp_alu_latency"] = random.choice(DESIGN_SPACE["fp_alu_latencies"])
        cfg["fp_mult_latency"] = random.choice(DESIGN_SPACE["fp_mult_latencies"])
        cfg["fp_div_latency"] = random.choice(DESIGN_SPACE["fp_div_latencies"])

        # 随机内存端口
        cfg["fu_read_port"] = random.randint(*DESIGN_SPACE["fu_read_port_range"])
        cfg["fu_write_port"] = random.randint(*DESIGN_SPACE["fu_write_port_range"])

        # 验证并修复配置
        cfg = validate_and_fix_config(cfg)

        configs.append(cfg)
    return configs


# ==========================================
# 5. 策略 C: 瓶颈注入 (Stress Testing) - 32个
# ==========================================
def generate_stress(count):
    configs = []
    scenarios = ["Frontend_Bound", "ROB_Bound", "Memory_Bound", "Compute_Bound"]

    for i in range(count):
        scenario = scenarios[i % len(scenarios)]
        cfg = create_template(f"Stress_{scenario}_{i}")

        width = 8
        for k in [
            "fetchWidth",
            "decodeWidth",
            "renameWidth",
            "issueWidth",
            "commitWidth",
            "squashWidth",
        ]:
            cfg[k] = width

        if scenario == "Frontend_Bound":
            cfg["fetchWidth"] = 2
            cfg["decodeWidth"] = 2
            cfg["bp_type"] = "BiModeBP"
            cfg["btb_entries"] = 256
            cfg["ras_entries"] = 8
            cfg["clock_frequency"] = 2.0
            cfg["description"] = "Weak Frontend, Tiny BTB"

        elif scenario == "ROB_Bound":
            cfg["numROBEntries"] = 40
            cfg["numIQEntries"] = 16
            cfg["LQEntries"] = 10
            cfg["SQEntries"] = 10
            cfg["l2_size"] = "4MB"
            cfg["clock_frequency"] = 3.0
            cfg["description"] = "Tiny Reorder Window"

        elif scenario == "Memory_Bound":
            # 高频 + 小缓存 = 极端内存敏感
            cfg["clock_frequency"] = 4.0
            cfg["numROBEntries"] = 256
            cfg["l1d_size"] = "16kB"
            cfg["l2_size"] = "256kB"
            cfg["l1d_mshrs"] = 4
            cfg["l2_mshrs"] = 8
            cfg["l1d_prefetcher"] = "Null"
            # High Latency to stress memory system
            cfg["l1d_latency"] = 4
            cfg["l1i_latency"] = 4
            cfg["l2_latency"] = 20
            # Pipeline delays (Standard)
            cfg["fetch_to_decode_delay"] = 1
            cfg["decode_to_rename_delay"] = 1
            cfg["rename_to_iew_delay"] = 1
            cfg["iew_to_commit_delay"] = 1
            # FU Latencies (Standard)
            cfg["int_alu_latency"] = 1
            cfg["int_mult_latency"] = 3
            cfg["int_div_latency"] = 20
            cfg["fp_alu_latency"] = 2
            cfg["fp_mult_latency"] = 4
            cfg["fp_div_latency"] = 12
            cfg["description"] = "Tiny Cache, Low Bandwidth, High Freq"

        elif scenario == "Compute_Bound":
            cfg["numROBEntries"] = 192
            cfg["fu_int_alu"] = 1
            cfg["fu_fp_alu"] = 1
            cfg["l1d_size"] = "128kB"
            cfg["clock_frequency"] = 3.5
            cfg["description"] = "Starved Execution Units"

        # Ensure all scenarios have latency parameters (if not already set)
        if "l1d_latency" not in cfg:
            cfg["l1d_latency"] = 2
        if "l1i_latency" not in cfg:
            cfg["l1i_latency"] = 2
        if "l2_latency" not in cfg:
            cfg["l2_latency"] = 14
        if "fetch_to_decode_delay" not in cfg:
            cfg["fetch_to_decode_delay"] = 1
        if "decode_to_rename_delay" not in cfg:
            cfg["decode_to_rename_delay"] = 1
        if "rename_to_iew_delay" not in cfg:
            cfg["rename_to_iew_delay"] = 1
        if "iew_to_commit_delay" not in cfg:
            cfg["iew_to_commit_delay"] = 1
        if "int_alu_latency" not in cfg:
            cfg["int_alu_latency"] = 1
        if "int_mult_latency" not in cfg:
            cfg["int_mult_latency"] = 3
        if "int_div_latency" not in cfg:
            cfg["int_div_latency"] = 20
        if "fp_alu_latency" not in cfg:
            cfg["fp_alu_latency"] = 2
        if "fp_mult_latency" not in cfg:
            cfg["fp_mult_latency"] = 4
        if "fp_div_latency" not in cfg:
            cfg["fp_div_latency"] = 12

        # 添加内存端口参数 (与 width 相关)
        if "fu_read_port" not in cfg:
            cfg["fu_read_port"] = max(1, min(width // 2, 4))
        if "fu_write_port" not in cfg:
            cfg["fu_write_port"] = max(1, min(width // 2, 4))

        # 验证并修复配置
        cfg = validate_and_fix_config(cfg)

        configs.append(cfg)
    return configs


# ==========================================
# 6. 策略 D: 常数参数变体探索 (Constant Parameter Variants) - 80个
# ==========================================
def generate_constant_variants(count):
    """
    专门针对 9 个常数参数生成变体配置，扩展这些参数的训练覆盖范围。

    变体生成使用的基准参数值:
    - fetchBufferSize = 16
    - fetchQueueSize = 16
    - fu_fp_mult = 2
    - fu_int_mult = 2
    - l1d_assoc = 8
    - l1i_assoc = 8
    - l1i_mshrs = 16
    - l1i_size = "32kB"
    - l2_assoc = 16
    """
    configs = []

    # 参数变体矩阵 - 每个参数的备选值
    param_variants = [
        # L1I size 变化 (影响 L1I MPKI)
        {"l1i_size": "16kB"},
        {"l1i_size": "64kB"},
        {"l1i_size": "128kB"},
        # L1I assoc 变化
        {"l1i_assoc": 4},
        {"l1i_assoc": 16},
        # L1D assoc 变化
        {"l1d_assoc": 4},
        {"l1d_assoc": 16},
        # L2 assoc 变化
        {"l2_assoc": 4},
        {"l2_assoc": 8},
        {"l2_assoc": 32},
        # L1I MSHR 变化
        {"l1i_mshrs": 8},
        {"l1i_mshrs": 32},
        {"l1i_mshrs": 64},
        # 整数乘法器变化
        {"fu_int_mult": 1},
        {"fu_int_mult": 4},
        # 浮点乘法器变化
        {"fu_fp_mult": 1},
        {"fu_fp_mult": 4},
        # 取指缓冲区变化 (必须是 64 的约数)
        {"fetchBufferSize": 8},
        {"fetchBufferSize": 16},
        {"fetchBufferSize": 32},
        {"fetchBufferSize": 64},
        # 取指队列变化 (必须是 64 的约数)
        {"fetchQueueSize": 8},
        {"fetchQueueSize": 16},
        {"fetchQueueSize": 32},
        {"fetchQueueSize": 64},
    ]

    # 重点配置组合 - 确保覆盖关键参数变化
    focused_combinations = []

    # L1I size + L1I assoc 组合 (高优先级)
    for l1i_size in ["16kB", "64kB", "128kB"]:
        for l1i_assoc in [4, 16]:
            focused_combinations.append({"l1i_size": l1i_size, "l1i_assoc": l1i_assoc})

    # L1D assoc + L2 assoc 组合
    for l1d_assoc in [4, 16]:
        for l2_assoc in [4, 32]:
            focused_combinations.append({"l1d_assoc": l1d_assoc, "l2_assoc": l2_assoc})

    # FU mult 组合
    for int_mult in [1, 4]:
        for fp_mult in [1, 4]:
            focused_combinations.append(
                {"fu_int_mult": int_mult, "fu_fp_mult": fp_mult}
            )

    # 生成配置
    for i in range(count):
        cfg = create_template(f"ConstantVariant_{i}")

        # 随机选择基准配置模板
        base_template = random.choice(
            [
                create_template("Balanced_Base"),
                create_template("Random_Base"),
                create_template("Stress_Base"),
            ]
        )
        # 设置合理的基准参数 (全部从 DESIGN_SPACE 选择)
        base_template["clock_frequency"] = random.choice(DESIGN_SPACE["frequencies"])
        width = random.choice(DESIGN_SPACE["widths"])
        for k in [
            "fetchWidth",
            "decodeWidth",
            "renameWidth",
            "issueWidth",
            "commitWidth",
            "squashWidth",
        ]:
            base_template[k] = width
        base_template["numROBEntries"] = log_uniform_sample(*DESIGN_SPACE["rob_range"])
        iq, lq, sq = get_aligned_queues(base_template["numROBEntries"])
        base_template["numIQEntries"] = iq
        base_template["LQEntries"] = lq
        base_template["SQEntries"] = sq
        base_template["fu_int_alu"] = random.randint(*DESIGN_SPACE["fu_int_range"])
        base_template["fu_fp_alu"] = random.randint(*DESIGN_SPACE["fu_fp_range"])
        base_template["fu_int_mult"] = random.randint(
            *DESIGN_SPACE["fu_int_mult_range"]
        )
        base_template["fu_fp_mult"] = random.randint(*DESIGN_SPACE["fu_fp_mult_range"])
        base_template["fetchBufferSize"] = random.choice(
            DESIGN_SPACE["fetch_buffer_sizes"]
        )
        base_template["fetchQueueSize"] = random.choice(
            DESIGN_SPACE["fetch_queue_sizes"]
        )
        base_template["l1d_size"] = random.choice(DESIGN_SPACE["l1_sizes"])
        base_template["l1d_assoc"] = random.choice(DESIGN_SPACE["l1d_assocs"])
        base_template["l1d_mshrs"] = random.choice(DESIGN_SPACE["mshrs_l1"])
        base_template["l1i_size"] = random.choice(DESIGN_SPACE["l1i_sizes"])
        base_template["l1i_assoc"] = random.choice(DESIGN_SPACE["l1i_assocs"])
        base_template["l1i_mshrs"] = random.choice(DESIGN_SPACE["mshrs_l1i"])
        base_template["l2_size"] = random.choice(DESIGN_SPACE["l2_sizes"])
        base_template["l2_assoc"] = random.choice(DESIGN_SPACE["l2_assocs_extended"])
        base_template["l2_mshrs"] = random.choice(DESIGN_SPACE["mshrs_l2"])
        base_template["bp_type"] = random.choice(DESIGN_SPACE["bp_types"])
        base_template["btb_entries"] = random.choice(DESIGN_SPACE["btb_entries"])
        base_template["ras_entries"] = random.choice(DESIGN_SPACE["ras_entries"])
        base_template["l1d_prefetcher"] = random.choice(DESIGN_SPACE["prefetchers"])

        # 补充 Latency
        base_template["l1d_latency"] = random.choice(DESIGN_SPACE["l1_latencies"])
        base_template["l1i_latency"] = random.choice(DESIGN_SPACE["l1_latencies"])
        base_template["l2_latency"] = random.choice(DESIGN_SPACE["l2_latencies"])

        # 补充 Pipeline Delays
        base_template["fetch_to_decode_delay"] = random.choice(
            DESIGN_SPACE["fetch_to_decode_delays"]
        )
        base_template["decode_to_rename_delay"] = random.choice(
            DESIGN_SPACE["decode_to_rename_delays"]
        )
        base_template["rename_to_iew_delay"] = random.choice(
            DESIGN_SPACE["rename_to_iew_delays"]
        )
        base_template["iew_to_commit_delay"] = random.choice(
            DESIGN_SPACE["iew_to_commit_delays"]
        )

        # 补充 FU Latencies
        base_template["int_alu_latency"] = random.choice(
            DESIGN_SPACE["int_alu_latencies"]
        )
        base_template["int_mult_latency"] = random.choice(
            DESIGN_SPACE["int_mult_latencies"]
        )
        base_template["int_div_latency"] = random.choice(
            DESIGN_SPACE["int_div_latencies"]
        )
        base_template["fp_alu_latency"] = random.choice(
            DESIGN_SPACE["fp_alu_latencies"]
        )
        base_template["fp_mult_latency"] = random.choice(
            DESIGN_SPACE["fp_mult_latencies"]
        )
        base_template["fp_div_latency"] = random.choice(
            DESIGN_SPACE["fp_div_latencies"]
        )

        # 复制基准配置
        for k, v in base_template.items():
            if k not in ["name", "description"]:
                cfg[k] = v

        cfg["description"] = "Constant Parameter Variant"

        # 决定使用重点组合还是随机变体
        if i < len(focused_combinations):
            # 使用重点组合
            variants_to_apply = [focused_combinations[i]]
            # 额外添加 1 个随机变体
            variants_to_apply.append(random.choice(param_variants))
        else:
            # 随机选择 2-3 个参数变体
            num_variants = random.randint(2, 3)
            variants_to_apply = random.sample(param_variants, num_variants)

        # 应用变体
        for variant in variants_to_apply:
            cfg.update(variant)

        # 添加内存端口参数 (与 width 相关)
        if "fu_read_port" not in cfg:
            cfg["fu_read_port"] = random.randint(*DESIGN_SPACE["fu_read_port_range"])
        if "fu_write_port" not in cfg:
            cfg["fu_write_port"] = random.randint(*DESIGN_SPACE["fu_write_port_range"])

        # 验证并修复配置
        cfg = validate_and_fix_config(cfg)

        configs.append(cfg)

    return configs


# ==========================================
# 7. 主程序
# ==========================================
if __name__ == "__main__":
    import os

    # 检查是否存在原有配置文件
    original_file = "gem5_hardware_vectors_120.json"
    use_original = os.path.exists(original_file)

    if use_original:
        print(f"Found existing {original_file}, loading original 120 configs...")
        with open(original_file, "r") as f:
            original_configs = json.load(f)
        print(f"Loaded {len(original_configs)} configs from {original_file}")

        # 补充缺失的 Latency 信息 (Backfill)
        for cfg in original_configs:
            # Cache Latencies
            if "l1d_latency" not in cfg:
                cfg["l1d_latency"] = 2
            if "l1i_latency" not in cfg:
                cfg["l1i_latency"] = 2
            if "l2_latency" not in cfg:
                cfg["l2_latency"] = 14
            # Pipeline Delays
            if "fetch_to_decode_delay" not in cfg:
                cfg["fetch_to_decode_delay"] = 1
            if "decode_to_rename_delay" not in cfg:
                cfg["decode_to_rename_delay"] = 1
            if "rename_to_iew_delay" not in cfg:
                cfg["rename_to_iew_delay"] = 1
            if "iew_to_commit_delay" not in cfg:
                cfg["iew_to_commit_delay"] = 1
            # FU Latencies
            if "int_alu_latency" not in cfg:
                cfg["int_alu_latency"] = 1
            if "int_mult_latency" not in cfg:
                cfg["int_mult_latency"] = 3
            if "int_div_latency" not in cfg:
                cfg["int_div_latency"] = 20
            if "fp_alu_latency" not in cfg:
                cfg["fp_alu_latency"] = 2
            if "fp_mult_latency" not in cfg:
                cfg["fp_mult_latency"] = 4
            if "fp_div_latency" not in cfg:
                cfg["fp_div_latency"] = 12
            # Memory Ports
            if "fu_read_port" not in cfg:
                # 基于 pipeline width 推断合理的端口数
                width = cfg.get("fetchWidth", 4)
                cfg["fu_read_port"] = max(1, min(width // 2, 4))
            if "fu_write_port" not in cfg:
                width = cfg.get("fetchWidth", 4)
                cfg["fu_write_port"] = max(1, min(width // 2, 4))
            # 验证并修复旧配置
            cfg = validate_and_fix_config(cfg)
        num_original = len(original_configs)
    else:
        print(f"{original_file} not found, generating 120 configs from scratch...")
        original_configs = []

        # 基础配置样本：120 个
        original_configs.extend(generate_balanced(48))  # 40%
        original_configs.extend(generate_random(40))  # 33%
        original_configs.extend(generate_stress(32))  # 27%
        num_original = 120

    # 参数变体样本：80 个
    print("Generating 80 constant variant configs...")
    new_configs = generate_constant_variants(80)

    # 合并配置：原有配置直接保留，新配置需要去重
    final_configs = []
    seen_hashes = set()

    # 先添加原有配置（不做去重修改）
    for cfg in original_configs:
        sig = json.dumps(
            {k: v for k, v in cfg.items() if k not in ["id", "name", "description"]},
            sort_keys=True,
        )
        seen_hashes.add(sig)
        final_configs.append(cfg)

    # 再添加新配置（去重）
    for cfg in new_configs:
        sig = json.dumps(
            {k: v for k, v in cfg.items() if k not in ["id", "name", "description"]},
            sort_keys=True,
        )

        if sig not in seen_hashes:
            seen_hashes.add(sig)
            final_configs.append(cfg)
        else:
            # 微扰去重
            cfg["numROBEntries"] += random.randint(1, 3)
            final_configs.append(cfg)

    # 分配 id
    for idx, cfg in enumerate(final_configs):
        cfg["id"] = idx

    print(f"Generated {len(final_configs)} unique configurations.")
    print(f"  Original: {num_original}, New: {len(final_configs) - num_original}")

    filename = "gem5_hardware_vectors_200.json"
    with open(filename, "w") as f:
        json.dump(final_configs, f, indent=2)

    print(f"Successfully saved to {filename}")
