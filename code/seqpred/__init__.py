"""SeqPredictor: TAO-style instruction-trace self-attention baseline for SensiSelect.

Evaluation settings shared with the FiLM model in train_unified_v2.py:
  - same 10 endpoints (METRIC_KEYS), same HW-holdout split (seed 42, 80/20 by hw_id),
  - same 47-dim HW encoding (dataset.py:_extract_hw_raw) + label normalization (log1p MPKI -> z-score),
  - same evaluation metrics (sens_rho / abs_rho / topk_recall@5%).

Backbone is disjoint from the FiLM four-stream encoder: it consumes raw
instruction-trace per-record features (opcode / regbm / bhist / accdist).
"""
