#!/usr/bin/env python3
"""
Generate up to 100 sweep configurations for sensitivity evaluation.

Strategy: Pick representative base configs, then sweep one target parameter.
Dependent resources and configuration constraints may change additional fields.
This enables ground-truth evaluation of "which intervals are most sensitive
to a specific HW change" — the core DSE use case.

Output: gem5_hardware_vectors_single_param.json, with five base configurations
(IDs 200-204) and up to 100 sweep configurations (IDs starting at 205).

Sweep parameters (10 key microarchitectural knobs):
  1. L1D size:       16kB → 32kB → 64kB → 128kB
  2. L2 size:        256kB → 512kB → 1MB → 2MB → 4MB
  3. ROB entries:    64 → 128 → 256 → 512
  4. Pipeline width: 2 → 4 → 6 → 8
  5. BP type:        BiMode → Tournament → LTAGE
  6. L1D latency:    1 → 2 → 3 → 4
  7. L2 latency:     10 → 14 → 20
  8. BTB entries:    512 → 2048 → 8192
  9. L1D MSHR:       4 → 16 → 32
 10. IQ entries:     16 → 64 → 128

Base configs (5): Low-end, Mid, High, Memory-heavy, Compute-heavy
Candidates are generated across all bases and sweep values. Unchanged candidates
are skipped, duplicates are removed, and the remaining candidates are sorted by
the number of changed fields. At most 100 are retained; there is no per-base quota.
"""
import argparse
import json
import copy
import sys
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"

# Import from the bundled config generator
sys.path.insert(0, str(CODE_DIR))
from gem5_configuration import (
    create_template, validate_and_fix_config, get_aligned_queues, DESIGN_SPACE
)


# ==========================================
# 5 Representative Base Configs
# ==========================================
def make_base_configs():
    """5 base configs spanning the design space."""
    bases = {}

    # Base A: Low-end / IoT-like
    a = create_template("Base_LowEnd")
    a["clock_frequency"] = 1.5
    for k in ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
              "issueWidth", "commitWidth", "squashWidth"]:
        a[k] = 2
    a["numROBEntries"] = 64
    a["numIQEntries"] = 32
    a["LQEntries"] = 16
    a["SQEntries"] = 16
    a["fetchBufferSize"] = 16
    a["fetchQueueSize"] = 16
    a["fu_int_alu"] = 2
    a["fu_int_mult"] = 1
    a["fu_fp_alu"] = 1
    a["fu_fp_mult"] = 1
    a["fu_read_port"] = 1
    a["fu_write_port"] = 1
    a["bp_type"] = "BiModeBP"
    a["btb_entries"] = 512
    a["ras_entries"] = 16
    a["l1d_size"] = "32kB"
    a["l1d_assoc"] = 8
    a["l1d_mshrs"] = 8
    a["l1d_latency"] = 2
    a["l1i_size"] = "32kB"
    a["l1i_assoc"] = 8
    a["l1i_mshrs"] = 16
    a["l1i_latency"] = 2
    a["l2_size"] = "512kB"
    a["l2_assoc"] = 8
    a["l2_mshrs"] = 16
    a["l2_latency"] = 12
    a["l1d_prefetcher"] = "Null"
    a["fetch_to_decode_delay"] = 1
    a["decode_to_rename_delay"] = 1
    a["rename_to_iew_delay"] = 1
    a["iew_to_commit_delay"] = 1
    a["int_alu_latency"] = 1
    a["int_mult_latency"] = 3
    a["int_div_latency"] = 20
    a["fp_alu_latency"] = 2
    a["fp_mult_latency"] = 4
    a["fp_div_latency"] = 10
    bases["LowEnd"] = validate_and_fix_config(a)

    # Base B: Mid-range / Desktop
    b = create_template("Base_Mid")
    b["clock_frequency"] = 2.5
    for k in ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
              "issueWidth", "commitWidth", "squashWidth"]:
        b[k] = 4
    b["numROBEntries"] = 128
    b["numIQEntries"] = 64
    b["LQEntries"] = 32
    b["SQEntries"] = 32
    b["fetchBufferSize"] = 16
    b["fetchQueueSize"] = 16
    b["fu_int_alu"] = 4
    b["fu_int_mult"] = 2
    b["fu_fp_alu"] = 2
    b["fu_fp_mult"] = 2
    b["fu_read_port"] = 2
    b["fu_write_port"] = 2
    b["bp_type"] = "TournamentBP"
    b["btb_entries"] = 2048
    b["ras_entries"] = 16
    b["l1d_size"] = "32kB"
    b["l1d_assoc"] = 8
    b["l1d_mshrs"] = 16
    b["l1d_latency"] = 2
    b["l1i_size"] = "32kB"
    b["l1i_assoc"] = 8
    b["l1i_mshrs"] = 16
    b["l1i_latency"] = 2
    b["l2_size"] = "1MB"
    b["l2_assoc"] = 16
    b["l2_mshrs"] = 16
    b["l2_latency"] = 14
    b["l1d_prefetcher"] = "StridePrefetcher"
    b["fetch_to_decode_delay"] = 2
    b["decode_to_rename_delay"] = 1
    b["rename_to_iew_delay"] = 1
    b["iew_to_commit_delay"] = 1
    b["int_alu_latency"] = 1
    b["int_mult_latency"] = 3
    b["int_div_latency"] = 20
    b["fp_alu_latency"] = 1
    b["fp_mult_latency"] = 4
    b["fp_div_latency"] = 10
    bases["Mid"] = validate_and_fix_config(b)

    # Base C: High-end / Server
    c = create_template("Base_HighEnd")
    c["clock_frequency"] = 3.5
    for k in ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
              "issueWidth", "commitWidth", "squashWidth"]:
        c[k] = 8
    c["numROBEntries"] = 256
    c["numIQEntries"] = 128
    c["LQEntries"] = 64
    c["SQEntries"] = 64
    c["fetchBufferSize"] = 32
    c["fetchQueueSize"] = 32
    c["fu_int_alu"] = 6
    c["fu_int_mult"] = 2
    c["fu_fp_alu"] = 4
    c["fu_fp_mult"] = 2
    c["fu_read_port"] = 4
    c["fu_write_port"] = 4
    c["bp_type"] = "LTAGE"
    c["btb_entries"] = 8192
    c["ras_entries"] = 32
    c["l1d_size"] = "64kB"
    c["l1d_assoc"] = 8
    c["l1d_mshrs"] = 16
    c["l1d_latency"] = 2
    c["l1i_size"] = "64kB"
    c["l1i_assoc"] = 8
    c["l1i_mshrs"] = 32
    c["l1i_latency"] = 2
    c["l2_size"] = "2MB"
    c["l2_assoc"] = 16
    c["l2_mshrs"] = 32
    c["l2_latency"] = 14
    c["l1d_prefetcher"] = "StridePrefetcher"
    c["fetch_to_decode_delay"] = 3
    c["decode_to_rename_delay"] = 2
    c["rename_to_iew_delay"] = 2
    c["iew_to_commit_delay"] = 2
    c["int_alu_latency"] = 1
    c["int_mult_latency"] = 4
    c["int_div_latency"] = 20
    c["fp_alu_latency"] = 2
    c["fp_mult_latency"] = 5
    c["fp_div_latency"] = 15
    bases["HighEnd"] = validate_and_fix_config(c)

    # Base D: Memory-heavy (large caches, high latency)
    d = create_template("Base_MemHeavy")
    d["clock_frequency"] = 3.0
    for k in ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
              "issueWidth", "commitWidth", "squashWidth"]:
        d[k] = 4
    d["numROBEntries"] = 192
    d["numIQEntries"] = 80
    d["LQEntries"] = 48
    d["SQEntries"] = 48
    d["fetchBufferSize"] = 16
    d["fetchQueueSize"] = 16
    d["fu_int_alu"] = 4
    d["fu_int_mult"] = 2
    d["fu_fp_alu"] = 2
    d["fu_fp_mult"] = 2
    d["fu_read_port"] = 2
    d["fu_write_port"] = 2
    d["bp_type"] = "TournamentBP"
    d["btb_entries"] = 4096
    d["ras_entries"] = 16
    d["l1d_size"] = "64kB"
    d["l1d_assoc"] = 8
    d["l1d_mshrs"] = 16
    d["l1d_latency"] = 3
    d["l1i_size"] = "32kB"
    d["l1i_assoc"] = 8
    d["l1i_mshrs"] = 16
    d["l1i_latency"] = 3
    d["l2_size"] = "4MB"
    d["l2_assoc"] = 16
    d["l2_mshrs"] = 32
    d["l2_latency"] = 16
    d["l1d_prefetcher"] = "StridePrefetcher"
    d["fetch_to_decode_delay"] = 2
    d["decode_to_rename_delay"] = 1
    d["rename_to_iew_delay"] = 1
    d["iew_to_commit_delay"] = 1
    d["int_alu_latency"] = 1
    d["int_mult_latency"] = 3
    d["int_div_latency"] = 20
    d["fp_alu_latency"] = 1
    d["fp_mult_latency"] = 4
    d["fp_div_latency"] = 10
    bases["MemHeavy"] = validate_and_fix_config(d)

    # Base E: Compute-heavy (wide execution, small caches)
    e = create_template("Base_CompHeavy")
    e["clock_frequency"] = 4.0
    for k in ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
              "issueWidth", "commitWidth", "squashWidth"]:
        e[k] = 6
    e["numROBEntries"] = 192
    e["numIQEntries"] = 96
    e["LQEntries"] = 48
    e["SQEntries"] = 48
    e["fetchBufferSize"] = 32
    e["fetchQueueSize"] = 32
    e["fu_int_alu"] = 6
    e["fu_int_mult"] = 3
    e["fu_fp_alu"] = 4
    e["fu_fp_mult"] = 3
    e["fu_read_port"] = 3
    e["fu_write_port"] = 3
    e["bp_type"] = "LTAGE"
    e["btb_entries"] = 4096
    e["ras_entries"] = 32
    e["l1d_size"] = "32kB"
    e["l1d_assoc"] = 8
    e["l1d_mshrs"] = 8
    e["l1d_latency"] = 1
    e["l1i_size"] = "32kB"
    e["l1i_assoc"] = 8
    e["l1i_mshrs"] = 16
    e["l1i_latency"] = 1
    e["l2_size"] = "512kB"
    e["l2_assoc"] = 8
    e["l2_mshrs"] = 16
    e["l2_latency"] = 10
    e["l1d_prefetcher"] = "Null"
    e["fetch_to_decode_delay"] = 2
    e["decode_to_rename_delay"] = 1
    e["rename_to_iew_delay"] = 1
    e["iew_to_commit_delay"] = 1
    e["int_alu_latency"] = 1
    e["int_mult_latency"] = 3
    e["int_div_latency"] = 10
    e["fp_alu_latency"] = 1
    e["fp_mult_latency"] = 4
    e["fp_div_latency"] = 10
    bases["CompHeavy"] = validate_and_fix_config(e)

    return bases


# ==========================================
# Single-Parameter Sweep Definitions
# ==========================================
# Each sweep: (param_name, values_list, description)
# For pipeline width, we set all width params together.
SWEEPS = [
    # Cache hierarchy (most relevant for DSE)
    ("l1d_size",     ["16kB", "32kB", "64kB", "128kB"],  "L1D Cache Size"),
    ("l2_size",      ["256kB", "512kB", "1MB", "2MB", "4MB"], "L2 Cache Size"),
    ("l1d_latency",  [1, 2, 3, 4],                        "L1D Latency"),
    ("l2_latency",   [10, 14, 20],                         "L2 Latency"),
    ("l1d_mshrs",    [4, 16, 32],                          "L1D MSHRs"),
    # Execution resources
    ("numROBEntries", [64, 128, 256, 512],                 "ROB Entries"),
    ("_width",        [2, 4, 6, 8],                        "Pipeline Width"),
    ("numIQEntries",  [16, 64, 128],                       "IQ Entries"),
    # Branch prediction
    ("bp_type",       ["BiModeBP", "TournamentBP", "LTAGE"], "BP Type"),
    ("btb_entries",   [512, 2048, 8192],                   "BTB Entries"),
]


def apply_sweep(base_cfg, param_name, value):
    """Apply a target-parameter sweep and adjust dependent configuration fields."""
    cfg = copy.deepcopy(base_cfg)

    if param_name == "_width":
        # Pipeline width affects all width params
        for k in ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
                   "issueWidth", "commitWidth", "squashWidth"]:
            cfg[k] = value
        # Adjust FU counts to match width
        cfg["fu_int_alu"] = max(2, value)
        cfg["fu_fp_alu"] = max(1, value // 2)
        cfg["fu_read_port"] = max(1, min(value // 2, 4))
        cfg["fu_write_port"] = max(1, min(value // 2, 4))
    elif param_name == "numROBEntries":
        cfg["numROBEntries"] = value
        # Adjust dependent queues
        cfg["numIQEntries"], cfg["LQEntries"], cfg["SQEntries"] = get_aligned_queues(value)
    else:
        cfg[param_name] = value

    return validate_and_fix_config(cfg)


def configs_are_equal(c1, c2):
    """Check if two configs are functionally identical (ignoring name/id/description)."""
    skip = {"name", "id", "description"}
    for k in set(list(c1.keys()) + list(c2.keys())):
        if k in skip:
            continue
        if c1.get(k) != c2.get(k):
            return False
    return True


def count_param_diffs(c1, c2):
    """Count how many parameters differ between two configs."""
    skip = {"name", "id", "description"}
    diffs = []
    for k in set(list(c1.keys()) + list(c2.keys())):
        if k in skip:
            continue
        if c1.get(k) != c2.get(k):
            diffs.append(k)
    return diffs


# ==========================================
# Main Generation
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "gem5_hardware_vectors_single_param.json"),
        help="Path for the generated single-parameter sweep hardware-config JSON.",
    )
    parser.add_argument(
        "--meta-output",
        default=str(DATA_DIR / "single_param_sweep_metadata.json"),
        help="Path for the generated single-parameter sweep metadata JSON.",
    )
    args = parser.parse_args()

    bases = make_base_configs()
    configs = []
    metadata = []  # Track (base_name, param_name, value) for each config

    print("Generating single-parameter sweep configs...")
    print(f"  Bases: {list(bases.keys())}")
    print(f"  Sweeps: {len(SWEEPS)} parameters")

    for base_name, base_cfg in bases.items():
        for param_name, values, desc in SWEEPS:
            for value in values:
                cfg = apply_sweep(base_cfg, param_name, value)
                cfg["name"] = f"Sweep_{base_name}_{param_name}_{value}"
                cfg["description"] = f"{desc} sweep: {value} (base={base_name})"

                # Skip if identical to base (value unchanged)
                if configs_are_equal(cfg, base_cfg):
                    continue

        # Count changed fields, including dependent adjustments and validation fixes.
                diffs = count_param_diffs(cfg, base_cfg)

                configs.append(cfg)
                metadata.append({
                    "base": base_name,
                    "swept_param": param_name,
                    "swept_value": value,
                    "actual_diffs": diffs,
                    "n_diffs": len(diffs),
                })

    # Deduplicate
    unique_configs = []
    seen = set()
    unique_metadata = []
    for cfg, meta in zip(configs, metadata):
        sig = json.dumps(
            {k: v for k, v in cfg.items() if k not in ["id", "name", "description"]},
            sort_keys=True,
        )
        if sig not in seen:
            seen.add(sig)
            unique_configs.append(cfg)
            unique_metadata.append(meta)

    # Trim to 100 if needed (prioritize single-param diffs)
    unique_configs.sort(key=lambda c: 0)  # stable sort
    # Sort metadata by n_diffs to prioritize cleanest sweeps
    paired = list(zip(unique_configs, unique_metadata))
    paired.sort(key=lambda x: x[1]["n_diffs"])

    if len(paired) > 100:
        paired = paired[:100]

    # Assign sweep IDs starting from 205.
    final_configs = []
    for idx, (cfg, meta) in enumerate(paired):
        cfg["id"] = 205 + idx
        final_configs.append(cfg)

    # Save base configurations with IDs 200-204.
    base_configs_list = []
    for i, (name, cfg) in enumerate(bases.items()):
        cfg = copy.deepcopy(cfg)
        cfg["id"] = 200 + i
        cfg["name"] = f"Base_{name}"
        cfg["description"] = f"Base config: {name}"
        base_configs_list.append(cfg)

    all_configs = base_configs_list + final_configs

    # Print summary
    print(f"\nGenerated {len(final_configs)} sweep configs + {len(base_configs_list)} base configs")
    print(f"Total: {len(all_configs)} configs (id={all_configs[0]['id']}-{all_configs[-1]['id']})")

    # Stats on parameter diffs
    meta_final = [m for _, m in paired[:len(final_configs)]]
    diff_counts = {}
    for m in meta_final:
        n = m["n_diffs"]
        diff_counts[n] = diff_counts.get(n, 0) + 1
    print(f"\nParameter diff distribution:")
    for n in sorted(diff_counts.keys()):
        print(f"  {n} params changed: {diff_counts[n]} configs")

    # Stats by swept parameter
    param_counts = {}
    for m in meta_final:
        p = m["swept_param"]
        param_counts[p] = param_counts.get(p, 0) + 1
    print(f"\nConfigs per swept parameter:")
    for p, c in sorted(param_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")

    # Stats by base
    base_counts = {}
    for m in meta_final:
        b = m["base"]
        base_counts[b] = base_counts.get(b, 0) + 1
    print(f"\nConfigs per base:")
    for b, c in sorted(base_counts.items()):
        print(f"  {b}: {c}")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(all_configs, f, indent=2)
    print(f"\nSaved to {output_file}")

    # Save metadata for evaluation
    meta_file = Path(args.meta_output)
    meta_output = {
        "bases": {name: cfg["id"] for name, cfg in zip(bases.keys(), base_configs_list)},
        "sweeps": meta_final,
    }
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with meta_file.open("w") as f:
        json.dump(meta_output, f, indent=2)
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    main()
