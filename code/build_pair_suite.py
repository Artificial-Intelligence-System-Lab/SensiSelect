"""Hardware constants used to construct structured evaluation groups.

Parameter order and size conversions match the original pair-suite builder.
Only the constants required by the released evaluator are included here.
"""

NUMERIC_PARAMS = [
    'clock_frequency', 'fetchWidth', 'decodeWidth', 'renameWidth',
    'dispatchWidth', 'issueWidth', 'commitWidth', 'squashWidth',
    'numROBEntries', 'numIQEntries', 'LQEntries', 'SQEntries',
    'fetchBufferSize', 'fetchQueueSize',
    'fu_int_alu', 'fu_int_mult', 'fu_fp_alu', 'fu_fp_mult',
    'fu_read_port', 'fu_write_port',
    'btb_entries', 'ras_entries',
    'l1d_size', 'l1d_assoc', 'l1d_mshrs', 'l1d_latency',
    'l1i_size', 'l1i_assoc', 'l1i_mshrs', 'l1i_latency',
    'l2_size', 'l2_assoc', 'l2_mshrs', 'l2_latency',
    'fetch_to_decode_delay', 'decode_to_rename_delay',
    'rename_to_iew_delay', 'iew_to_commit_delay',
    'int_alu_latency', 'int_mult_latency', 'int_div_latency',
    'fp_alu_latency', 'fp_mult_latency', 'fp_div_latency',
]

SIZE_TO_KB = {
    '16kB': 16, '32kB': 32, '48kB': 48, '64kB': 64, '128kB': 128,
    '256kB': 256, '512kB': 512, '1MB': 1024, '2MB': 2048, '4MB': 4096,
}
