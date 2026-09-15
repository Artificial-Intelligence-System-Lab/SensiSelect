"""Shared evaluation infrastructure for the current unified FiLM model.

Uses DataLoader-based batching to avoid pre-loading entire dataset onto GPU.
Defers GPU loading to inference time.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau

CODE_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CODE_DIR.parent
DATA_DIR = RELEASE_ROOT / "data"
CHECKPOINT_DIR = RELEASE_ROOT / "checkpoints"
EVAL_RESULTS_DIR = RELEASE_ROOT / "eval_results"
DEFAULT_DATASET_PATH = str(DATA_DIR / "training_dataset_ALL.pkl")
DEFAULT_SWEEP_META_PATH = str(DATA_DIR / "single_param_sweep_metadata.json")
DEFAULT_SWEEP_CONFIG_PATH = str(DATA_DIR / "gem5_hardware_vectors_single_param.json")
DEFAULT_HW_JSON_PATH = str(DATA_DIR / "gem5_hardware_vectors_0_304.json")

sys.path.insert(0, str(CODE_DIR))

from dataset import MicroArchDataset
from model_unified_film_v2 import FiLMUnifiedModel, METRIC_KEYS

SEED = 42

TASK_KEYS = list(METRIC_KEYS)
CACHE_TASKS = ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate")
TASK_INDEX = {name: idx for idx, name in enumerate(TASK_KEYS)}
DEFAULT_ABS_REGRESSOR_CHECKPOINT = str(CHECKPOINT_DIR / "abs_only_regressor_hw_best_model.pth")
MODEL_CONFIG_ENV_KEYS = (
    "HIDDEN",
    "GAM_HIDDEN",
    "MODEL_TYPE",
    "HW_ENCODER",
    "USE_RESIDUAL",
    "C2A_ALPHA",
    "C3A_ENABLED",
    "C3A_LAMBDA",
    "A1_ENABLED",
    "A1_BASIS_HIDDEN",
    "A1_KNOB_MODE",
    "H1A_ENABLED",
    "H2A_ENABLED",
    "H2A_CONCAVE",
    "H2A_ABOVE_MODE",
    "H3A_ENABLED",
    "H3A_ETA",
    "H3A_DETACH_BASE",
    "H3A_MODE",
    "CACHEX_ENABLED",
    "N1A_ENABLED",
    "DIRECT_DELTA_ENABLED",
    "DROPOUT",
    "NOISE_STD",
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _resolve_existing_path(candidates, label):
    seen = set()
    rendered = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        rendered.append(str(resolved))
        if resolved.exists():
            return str(resolved)
    raise FileNotFoundError(f"Could not find {label}. Searched: {rendered}")


def resolve_data_path(raw_path):
    raw = Path(raw_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([
            Path.cwd() / raw,
            CODE_DIR / raw,
            RELEASE_ROOT / raw,
            DATA_DIR / raw.name,
        ])
    return _resolve_existing_path(candidates, f"data path '{raw_path}'")


def resolve_checkpoint_path(raw_path):
    raw = Path(raw_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([
            Path.cwd() / raw,
            CODE_DIR / raw,
            RELEASE_ROOT / raw,
            CHECKPOINT_DIR / raw.name,
        ])
    return _resolve_existing_path(candidates, f"checkpoint path '{raw_path}'")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_spearman(y_true, y_pred):
    if len(y_true) < 3:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else 0.0


def topk_recall(true_sens, pred_score, k):
    k = max(1, min(int(k), len(true_sens)))
    true_topk = set(np.argsort(true_sens)[-k:])
    pred_topk = set(np.argsort(pred_score)[-k:])
    return len(true_topk & pred_topk) / len(true_topk)


def ndcg_at_k(true_sens, pred_score, k):
    k = max(1, min(int(k), len(true_sens)))
    pred_order = np.argsort(pred_score)[-k:][::-1]
    ideal_order = np.argsort(true_sens)[-k:][::-1]
    gains = true_sens[pred_order]
    ideal_gains = true_sens[ideal_order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    idcg = float(np.sum(ideal_gains * discounts))
    return 0.0 if idcg <= 1e-12 else dcg / idcg


def set_ndcg_at_k(true_sens, selected_idx, k):
    """Tie-neutral NDCG for methods that output an unordered top-K set."""
    k = max(1, min(int(k), len(true_sens)))
    selected_idx = np.asarray(sorted(set(int(i) for i in selected_idx)), dtype=np.int64)
    if selected_idx.size == 0:
        return 0.0
    selected_idx = selected_idx[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    selected_gains = true_sens[selected_idx]
    # The set has no internal ranking, so assign each selected item the expected
    # discount of a uniformly random slot among the K available positions.
    # Missing slots (|S| < K) are treated as zero-gain positions.
    dcg = float(selected_gains.sum() * discounts.mean())
    ideal_order = np.argsort(true_sens)[-k:][::-1]
    ideal_gains = true_sens[ideal_order]
    idcg = float(np.sum(ideal_gains * discounts))
    return 0.0 if idcg <= 1e-12 else dcg / idcg


def positive_degradation(delta):
    """Positive part of an interval-wise delta vector."""
    delta = np.asarray(delta, dtype=np.float64)
    return np.maximum(delta, 0.0)


def bottleneck_top_indices(severity, tail_frac):
    """Indices of the top tail_frac intervals by positive degradation severity."""
    severity = np.asarray(severity, dtype=np.float64)
    if severity.size == 0:
        return np.array([], dtype=np.int64)
    tail_k = max(1, min(int(len(severity) * float(tail_frac)), len(severity)))
    return np.argsort(severity)[-tail_k:]


def valid_bottleneck_query(severity, tail_frac=0.05, min_tail_mass=1e-6):
    """Return True when a query-metric pair has a meaningful non-zero bottleneck tail."""
    severity = np.asarray(severity, dtype=np.float64)
    if severity.size == 0:
        return False
    tail_idx = bottleneck_top_indices(severity, tail_frac)
    if tail_idx.size == 0:
        return False
    tail_vals = severity[tail_idx]
    return (
        int((severity > 1e-12).sum()) >= int(tail_idx.size)
        and float(tail_vals.sum()) > float(min_tail_mass)
        and float(tail_vals.max()) > 1e-8
    )


def bottleneck_hit_at_k(severity, pred_score, k, tail_frac=0.05):
    k = max(1, min(int(k), len(severity)))
    true_top = set(int(i) for i in bottleneck_top_indices(severity, tail_frac))
    pred_top = set(int(i) for i in np.argsort(pred_score)[-k:])
    return 1.0 if true_top & pred_top else 0.0


def bottleneck_recall_at_k(severity, pred_score, k, tail_frac=0.05):
    k = max(1, min(int(k), len(severity)))
    true_top = set(int(i) for i in bottleneck_top_indices(severity, tail_frac))
    pred_top = set(int(i) for i in np.argsort(pred_score)[-k:])
    return 0.0 if not true_top else len(true_top & pred_top) / len(true_top)


def tail_mass_capture_at_k(severity, pred_score, k, tail_frac=0.05):
    k = max(1, min(int(k), len(severity)))
    true_idx = bottleneck_top_indices(severity, tail_frac)
    if true_idx.size == 0:
        return 0.0
    pred_top = set(int(i) for i in np.argsort(pred_score)[-k:])
    true_top = set(int(i) for i in true_idx)
    inter = sorted(true_top & pred_top)
    denom = float(np.sum(severity[true_idx]))
    if denom <= 1e-12:
        return 0.0
    numer = float(np.sum(severity[np.array(inter, dtype=np.int64)])) if inter else 0.0
    return numer / denom


def max_severity_capture_at_k(severity, pred_score, k):
    k = max(1, min(int(k), len(severity)))
    severity = np.asarray(severity, dtype=np.float64)
    denom = float(np.max(severity))
    if denom <= 1e-12:
        return 0.0
    pred_idx = np.argsort(pred_score)[-k:]
    numer = float(np.max(severity[pred_idx])) if len(pred_idx) > 0 else 0.0
    return numer / denom


def stack_metric_dict(metric_dict):
    return torch.cat([metric_dict[key] for key in METRIC_KEYS], dim=-1)


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _parse_tristate(raw):
    if raw is None:
        return None
    raw = str(raw).strip().lower()
    if raw in {"", "auto"}:
        return None
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid tri-state value: {raw}")


def add_model_override_args(parser):
    """Add explicit model-config override args to an argparse parser."""
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--noise-std', type=float, default=None)
    parser.add_argument('--interaction-type', choices=['film', 'swiglu'], default=None)
    parser.add_argument('--hw-encoder', choices=['attention', 'gam'], default=None)
    parser.add_argument('--use-residual', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--c2a-alpha', type=float, default=None)
    parser.add_argument('--c3a-enabled', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--a1-enabled', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--a1-basis-hidden', type=int, default=None)
    parser.add_argument('--a1-knob-mode', choices=['full47', 'min'], default=None)
    parser.add_argument('--h1a-enabled', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--h2a-enabled', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--h2a-concave', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--h2a-above-mode', choices=['boundary', 'zero'], default=None)
    parser.add_argument('--h3a-enabled', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--h3a-eta', type=float, default=None)
    parser.add_argument('--h3a-detach-base', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--h3a-mode', choices=['full', 'prior_only', 'residual_only'], default=None)
    parser.add_argument('--cachex-enabled', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--direct-delta-enabled', choices=['auto', '0', '1'], default='auto')
    return parser


def add_dataset_path_arg(parser):
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to the unified interval-level dataset pickle.",
    )
    return parser


def add_sweep_data_args(parser):
    add_dataset_path_arg(parser)
    parser.add_argument(
        "--sweep-meta-path",
        default=DEFAULT_SWEEP_META_PATH,
        help="Path to single-parameter sweep metadata JSON.",
    )
    parser.add_argument(
        "--sweep-config-path",
        default=DEFAULT_SWEEP_CONFIG_PATH,
        help="Path to single-parameter sweep hardware-config JSON.",
    )
    return parser


def add_hw_json_arg(parser):
    parser.add_argument(
        "--hw-json-path",
        default=DEFAULT_HW_JSON_PATH,
        help="Path to the held-out-HW hardware-vector JSON.",
    )
    return parser


def collect_model_overrides(args):
    """Collect explicit override args from argparse Namespace."""
    overrides = {}
    scalar_map = {
        'hidden': 'embed_dim',
        'dropout': 'dropout',
        'noise_std': 'noise_std',
        'interaction_type': 'interaction_type',
        'hw_encoder': 'hw_encoder',
        'c2a_alpha': 'c2a_alpha',
        'a1_basis_hidden': 'a1_basis_hidden',
        'a1_knob_mode': 'a1_knob_mode',
        'h2a_above_mode': 'h2a_above_mode',
        'h3a_eta': 'h3a_eta',
        'h3a_mode': 'h3a_mode',
    }
    for arg_name, cfg_name in scalar_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_name] = value

    tristate_map = {
        'use_residual': 'use_residual',
        'c3a_enabled': 'c3a_enabled',
        'a1_enabled': 'a1_enabled',
        'h1a_enabled': 'h1a_enabled',
        'h2a_enabled': 'h2a_enabled',
        'h2a_concave': 'h2a_concave',
        'h3a_enabled': 'h3a_enabled',
        'h3a_detach_base': 'h3a_detach_base',
        'cachex_enabled': 'cachex_enabled',
        'direct_delta_enabled': 'direct_delta_enabled',
    }
    for arg_name, cfg_name in tristate_map.items():
        if hasattr(args, arg_name):
            parsed = _parse_tristate(getattr(args, arg_name))
            if parsed is not None:
                overrides[cfg_name] = parsed
    return overrides


def add_regressor_override_mode_arg(parser):
    parser.add_argument(
        "--regressor-override-mode",
        choices=["auto", "none", "match-main"],
        default="auto",
        help=(
            "How to apply explicit model override args to the abs regressor baseline. "
            "'auto' uses no regressor overrides unless no explicit overrides were passed; "
            "if explicit main-model overrides are present, auto requires an explicit choice."
        ),
    )


def resolve_regressor_model_overrides(main_overrides, mode):
    """Resolve regressor override policy relative to explicit main-model overrides.

    We do not silently reuse selector-specific override flags for the regressor.
    If the caller passed explicit main-model overrides and also enabled the
    regressor baseline, they must state whether the regressor should ignore
    them (`none`) or match them (`match-main`).
    """
    mode = (mode or "auto").strip().lower()
    has_explicit = bool(main_overrides)
    if mode == "match-main":
        return dict(main_overrides)
    if mode == "none":
        return None
    if mode != "auto":
        raise ValueError(f"Unknown regressor override mode: {mode}")
    if has_explicit:
        raise ValueError(
            "Explicit model overrides are active with an abs regressor baseline. "
            "Choose --regressor-override-mode=none or --regressor-override-mode=match-main."
        )
    return None


def _unwrap_state(raw_state):
    if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        return raw_state["state_dict"]
    return raw_state


def _load_checkpoint_state(checkpoint_path, device):
    try:
        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        raw_state = torch.load(checkpoint_path, map_location=device)
    state = _unwrap_state(raw_state)
    remapped = {}
    for k, v in state.items():
        if k.startswith("rfp_enc."):
            continue
        if k.startswith("film."):
            remapped[k.replace("film.", "interaction.", 1)] = v
        else:
            remapped[k] = v
    return remapped


def _infer_hidden_dim(state):
    if "br_proj.weight" in state:
        return int(state["br_proj.weight"].shape[0])
    if "cache_heads.l1_mpki.1.weight" in state:
        return int(state["cache_heads.l1_mpki.1.weight"].shape[0])
    return _env_int("HIDDEN", 256)


def _infer_gam_hidden(state):
    """Infer GAM_HIDDEN from hw_encoder.param_fn.0.0.weight shape[0]."""
    weight = state.get("hw_encoder.param_fn.0.0.weight")
    if weight is not None and len(weight.shape) == 2:
        return int(weight.shape[0])
    return _env_int("GAM_HIDDEN", 32)


def _infer_interaction_type(state, checkpoint_path):
    raw = os.getenv("MODEL_TYPE", "").strip().lower()
    if raw in {"film", "swiglu"}:
        return raw
    if any(k.startswith("interaction.proj_w_gate.") for k in state):
        return "swiglu"
    if any(k.startswith("interaction.to_gamma_beta.") for k in state):
        return "film"
    basename = os.path.basename(checkpoint_path).lower()
    if "swiglu" in basename:
        return "swiglu"
    return "film"


def _infer_hw_encoder(state):
    raw = os.getenv("HW_ENCODER", "").strip().lower()
    if raw in {"attention", "gam"}:
        return raw
    if any(k.startswith("hw_encoder.") for k in state):
        return "gam"
    return "attention"


def _infer_a1_knob_mode(state, checkpoint_path):
    raw = os.getenv("A1_KNOB_MODE", "").strip().lower()
    if raw in {"full47", "min"}:
        return raw
    for metric in ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate"):
        key = f"a1_cache_heads.{metric}.coeff.4.weight"
        if key not in state:
            continue
        n_knobs = int(state[key].shape[0])
        if n_knobs == 47:
            return "full47"
        return "min"
    basename = os.path.basename(checkpoint_path).lower()
    if "a1_min" in basename or basename.endswith("_min.pth") or "_min_" in basename:
        return "min"
    return "full47"


def _infer_a1_basis_hidden(state):
    raw = os.getenv("A1_BASIS_HIDDEN", "").strip()
    if raw:
        return int(raw)
    for metric in ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate"):
        key = f"a1_cache_heads.{metric}.response.0.0.weight"
        if key in state:
            return int(state[key].shape[0])
    return 16


def _infer_h2a_concave(checkpoint_path):
    basename = os.path.basename(checkpoint_path).lower()
    if any(tok in basename for tok in ("_c0", "concave0", "concave=0", "cummax_only")):
        return False
    if any(tok in basename for tok in ("_c1", "concave1", "concave=1", "cummin")):
        return True
    return _env_flag("H2A_CONCAVE", True)


def _infer_h2a_above_mode():
    raw = os.getenv("H2A_ABOVE_MODE", "").strip().lower()
    if raw in {"boundary", "zero"}:
        return raw
    return "boundary"


def _infer_h3a_mode(checkpoint_path):
    raw = os.getenv("H3A_MODE", "").strip().lower()
    if raw in {"full", "prior_only", "residual_only"}:
        return raw
    basename = os.path.basename(checkpoint_path).lower()
    if "h3d" in basename or "residual_only" in basename:
        return "residual_only"
    if "h3c" in basename or "prior_only" in basename:
        return "prior_only"
    return "full"


def _infer_h3a_eta(checkpoint_path):
    raw = os.getenv("H3A_ETA", "").strip()
    if raw:
        return float(raw)
    basename = os.path.basename(checkpoint_path).lower()
    match = re.search(r"eta[_=]?([0-9]+(?:\.[0-9]+)?)", basename)
    if match:
        return float(match.group(1))
    if "h3" in basename:
        return 0.01
    return 0.02


def _infer_c3a_enabled(c3a_present):
    raw_flag = os.getenv("C3A_ENABLED")
    if raw_flag is not None:
        return raw_flag.strip().lower() in {"1", "true", "yes", "on"}
    raw_lambda = os.getenv("C3A_LAMBDA")
    if raw_lambda is not None and raw_lambda.strip() != "":
        return float(raw_lambda) > 0.0
    return c3a_present


def _build_hotl_stats(dataset):
    return {
        "gwfp_mean": float(dataset.stats["gwfp_mean"]),
        "gwfp_std": float(dataset.stats["gwfp_std"]),
        "gwifp_mean": float(dataset.stats["gwifp_mean"]),
        "gwifp_std": float(dataset.stats["gwifp_std"]),
        "cache_hw_mean": dataset.stats["hw_mean"][16:19].tolist(),
        "cache_hw_std": dataset.stats["hw_std"][16:19].tolist(),
    }


def _infer_model_config(state, checkpoint_path):
    keys = set(state.keys())
    c2a_present = "delta_scale_raw" in keys
    c3a_present = any(k.startswith("score_heads.") for k in keys)
    a1_present = any(k.startswith("a1_cache_heads.") for k in keys)
    h1a_present = any(k.startswith("h1a_proj.") for k in keys)
    h2a_present = any(k.startswith("h2a_proj.") or k.startswith("h2a_pair_proj.") for k in keys)
    h3a_present = any(k.startswith("h3a_pair_proj.") or k.startswith("h3a_alpha_heads.") for k in keys)
    cachex_present = any(k.startswith("cachex_proj.") for k in keys)
    n1a_present = any(k.startswith("stall_readouts.") for k in keys)
    direct_delta_present = any(k.startswith("direct_pair_trunk.") or k.startswith("direct_delta_heads.") for k in keys)

    # Infer GAM_HIDDEN from hw_encoder param shape
    gam_hidden = _infer_gam_hidden(state)

    c2a_alpha = _env_float("C2A_ALPHA", None)
    if c2a_present:
        if c2a_alpha is None:
            raise ValueError(
                f"{checkpoint_path} contains C2a parameters; set C2A_ALPHA explicitly for evaluation."
            )
    else:
        c2a_alpha = 0.0

    return {
        "embed_dim": _env_int("HIDDEN", _infer_hidden_dim(state)),
        "gam_hidden": gam_hidden,
        "dropout": _env_float("DROPOUT", 0.1),
        "noise_std": _env_float("NOISE_STD", 0.01),
        "interaction_type": _infer_interaction_type(state, checkpoint_path),
        "hw_encoder": _infer_hw_encoder(state),
        "use_residual": _env_flag(
            "USE_RESIDUAL", any(k.startswith("linear_baseline.") for k in keys)
        ),
        "c2a_alpha": c2a_alpha,
        "c3a_enabled": _infer_c3a_enabled(c3a_present),
        "a1_enabled": _env_flag("A1_ENABLED", a1_present),
        "a1_basis_hidden": _infer_a1_basis_hidden(state),
        "a1_knob_mode": _infer_a1_knob_mode(state, checkpoint_path),
        "h1a_enabled": _env_flag("H1A_ENABLED", h1a_present),
        "h2a_enabled": _env_flag("H2A_ENABLED", h2a_present),
        "h2a_concave": _infer_h2a_concave(checkpoint_path),
        "h2a_above_mode": _infer_h2a_above_mode(),
        "h3a_enabled": _env_flag("H3A_ENABLED", h3a_present),
        "h3a_eta": _infer_h3a_eta(checkpoint_path),
        "h3a_detach_base": _env_flag("H3A_DETACH_BASE", True),
        "h3a_mode": _infer_h3a_mode(checkpoint_path),
        "cachex_enabled": _env_flag("CACHEX_ENABLED", cachex_present),
        "n1a_enabled": _env_flag("N1A_ENABLED", n1a_present),
        "direct_delta_enabled": _env_flag("DIRECT_DELTA_ENABLED", direct_delta_present),
    }


def _apply_model_overrides(config, model_overrides):
    if not model_overrides:
        return config
    merged = dict(config)
    merged.update(model_overrides)
    return merged


def _run_with_env_mask(keys, fn):
    """Temporarily unset selected env vars while calling fn()."""
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_film_model(checkpoint_path, device, dataset, model_overrides=None, allow_ambient_env=True):
    """Load the current v2 FiLMUnifiedModel from checkpoint.

    The loader reconstructs the branch config from checkpoint keys plus
    environment overrides, then loads strictly to avoid silently evaluating a
    mismatched architecture.
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)

    def _build():
        state = _load_checkpoint_state(checkpoint_path, device)
        config = _apply_model_overrides(
            _infer_model_config(state, checkpoint_path),
            model_overrides,
        )

        hotl_stats = None
        if config["h1a_enabled"] or config["h2a_enabled"]:
            hotl_stats = _build_hotl_stats(dataset)
        prev_gam_hidden = os.environ.get("GAM_HIDDEN")
        try:
            if config.get("gam_hidden") is not None:
                os.environ["GAM_HIDDEN"] = str(config["gam_hidden"])
            model = FiLMUnifiedModel(
                embed_dim=config["embed_dim"],
                dropout=config["dropout"],
                noise_std=config["noise_std"],
                interaction_type=config["interaction_type"],
                hw_encoder=config["hw_encoder"],
                use_residual=config["use_residual"],
                c2a_alpha=config["c2a_alpha"],
                c3a_enabled=config["c3a_enabled"],
                a1_enabled=config["a1_enabled"],
                a1_basis_hidden=config["a1_basis_hidden"],
                a1_knob_mode=config["a1_knob_mode"],
                h1a_enabled=config["h1a_enabled"],
                h1a_stats=hotl_stats if config["h1a_enabled"] else None,
                h2a_enabled=config["h2a_enabled"],
                h2a_stats=hotl_stats if config["h2a_enabled"] else None,
                h2a_concave=config["h2a_concave"],
                h2a_above_mode=config["h2a_above_mode"],
                h3a_enabled=config["h3a_enabled"],
                h3a_eta=config["h3a_eta"],
                h3a_detach_base=config["h3a_detach_base"],
                h3a_mode=config["h3a_mode"],
                cachex_enabled=config["cachex_enabled"],
                n1a_enabled=config.get("n1a_enabled", False),
                direct_delta_enabled=config.get("direct_delta_enabled", False),
            ).to(device)
        finally:
            if prev_gam_hidden is None:
                os.environ.pop("GAM_HIDDEN", None)
            else:
                os.environ["GAM_HIDDEN"] = prev_gam_hidden

        # Strip JEPA auxiliary keys when evaluating non-JEPA checkpoints
        jepa_keys = [k for k in state if k.startswith("jepa_")]
        for k in jepa_keys:
            del state[k]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load {checkpoint_path} into v2 model with inferred config {config}"
            ) from exc
        model.eval()
        return model

    if allow_ambient_env:
        return _build()
    return _run_with_env_mask(MODEL_CONFIG_ENV_KEYS, _build)


def validate_abs_regressor_checkpoint(checkpoint_path, device):
    """Reject selector-style checkpoints passed as the abs regressor baseline.

    The abs-only regressor baseline should not carry learned ranking heads such
    as C3a/NC3/J1a. We validate the checkpoint structure before evaluation so a
    miswired experiment does not silently relabel another learned selector as
    "Abs. Regressor".
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    state = _load_checkpoint_state(checkpoint_path, device)
    forbidden_prefixes = (
        "score_heads.",
        "nc3_score_heads.",
        "j1a_score_head.",
    )
    present = [prefix for prefix in forbidden_prefixes if any(k.startswith(prefix) for k in state)]
    if present:
        raise ValueError(
            f"{checkpoint_path} is not a valid abs regressor checkpoint; "
            f"found selector-specific weights for {present}"
        )
    if not any(k.startswith("cache_heads.") for k in state):
        raise ValueError(
            f"{checkpoint_path} does not look like a valid performance regressor checkpoint; "
            "missing cache_heads.* weights"
        )
    return state


def hw_holdout_split(dataset, seed=SEED):
    """80/20 hardware-holdout split, aligned with train_unified_v2 SPLIT_MODE=hw."""
    hw_ids = np.array([int(sample.get("hw_id", i)) for i, sample in enumerate(dataset.samples)])
    unique_hw = np.unique(hw_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_hw)
    n_train_hw = int(len(unique_hw) * 0.8)
    train_hw = set(unique_hw[:n_train_hw].tolist())
    val_hw = set(unique_hw[n_train_hw:].tolist())
    train_idx = np.array([i for i in range(len(dataset)) if int(hw_ids[i]) in train_hw], dtype=np.int64)
    val_idx = np.array([i for i in range(len(dataset)) if int(hw_ids[i]) in val_hw], dtype=np.int64)
    return train_idx, val_idx, train_hw, val_hw


def collate_film_batch(batch):
    """Collate list of sample dicts into batched tensors for FiLM model."""
    fp = torch.stack([s["fp"] for s in batch])
    ifp = torch.stack([s["ifp"] for s in batch])
    gwfp = torch.stack([s["gwfp"] for s in batch])
    gwifp = torch.stack([s["gwifp"] for s in batch])
    hw = torch.stack([s["hw"] for s in batch])
    rfp = torch.stack([s["rfp"] for s in batch])
    bbe = torch.stack([s["bbe"] for s in batch])
    br = torch.stack([s["br"] for s in batch])
    labels = torch.stack([s["labels"] for s in batch])
    group_ids = torch.tensor([s.get("group_id", 0) for s in batch], dtype=torch.long)
    hw_ids = torch.tensor([s.get("hw_id", 0) for s in batch], dtype=torch.long)
    return fp, ifp, gwfp, gwifp, hw, rfp, bbe, br, labels, group_ids, hw_ids


def collect_predictions_from_indices(model, dataset, indices, device, batch_size=128):
    """Run inference on specific dataset indices, return per-sample dicts."""
    from torch.utils.data import DataLoader

    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_film_batch, num_workers=0)

    all_data = []
    with torch.no_grad():
        for batch in loader:
            fp, ifp, gwfp, gwifp, hw, rfp, bbe, br, labels, group_ids, hw_ids = batch
            fp, ifp, gwfp, gwifp, hw, rfp, bbe, br = [
                t.to(device) for t in [fp, ifp, gwfp, gwifp, hw, rfp, bbe, br]]

            output = model.forward_absolute(
                fp, ifp, hw, rfp, bbe, br, gwfp, gwifp)
            pred_abs = stack_metric_dict(output["absolute"]).cpu().numpy()
            labels_np = labels[:, :10].cpu().numpy()
            feat_np = output.get("_feat")
            H_np = output.get("_H")
            h6_feat_np = output.get("_h6_feat")
            cachex_feat_np = output.get("_cachex_feat")
            h2_desc = output.get("_h2_desc")

            for i in range(len(pred_abs)):
                item = {
                    'group_id': int(group_ids[i]),
                    'hw_id': int(hw_ids[i]),
                    'pred': pred_abs[i],
                    'label': labels_np[i],
                }
                if feat_np is not None:
                    item['feat'] = feat_np[i].detach().cpu().numpy().astype(np.float32, copy=False)
                if H_np is not None:
                    item['H'] = H_np[i].detach().cpu().numpy().astype(np.float32, copy=False)
                if h6_feat_np is not None:
                    item['h6_feat'] = h6_feat_np[i].detach().cpu().numpy().astype(np.float32, copy=False)
                if cachex_feat_np is not None:
                    item['cachex_feat'] = cachex_feat_np[i].detach().cpu().numpy().astype(np.float32, copy=False)
                if h2_desc is not None:
                    item['h2_desc'] = {
                        name: desc[i].detach().cpu().numpy().astype(np.float32, copy=False)
                        for name, desc in h2_desc.items()
                    }
                all_data.append(item)
    return all_data


def build_group_hw_map(all_data):
    groups = defaultdict(dict)
    for item in all_data:
        groups[item['group_id']][item['hw_id']] = item
    return groups


def collect_pair_bundle(groups, h1, h2):
    rows = []
    for gid, hw_map in groups.items():
        if h1 in hw_map and h2 in hw_map:
            rows.append((gid, hw_map[h1], hw_map[h2]))
    if not rows:
        return [], [], [], None, None
    rows.sort(key=lambda x: x[0])
    gids = [gid for gid, _, _ in rows]
    items_a = [a for _, a, _ in rows]
    items_b = [b for _, _, b in rows]
    gt = np.stack([a['label'] - b['label'] for a, b in zip(items_a, items_b)], axis=0)
    pred = np.stack([a['pred'] - b['pred'] for a, b in zip(items_a, items_b)], axis=0)
    return gids, items_a, items_b, gt, pred


def compute_pair_scores(model, items_a, items_b, device, batch_size=4096):
    if model is None or not items_a:
        return {}

    def _has_field(items, field):
        return all(field in item and item[field] is not None for item in items)

    pred_delta = np.stack([a['pred'] - b['pred'] for a, b in zip(items_a, items_b)], axis=0)
    scores = {}

    with torch.no_grad():
        if (
            getattr(model, 'c3a_enabled', False)
            and _has_field(items_a, 'feat') and _has_field(items_b, 'feat')
            and _has_field(items_a, 'H') and _has_field(items_b, 'H')
        ):
            score_buffers = {name: [] for name in getattr(model, 'CACHE_METRICS', CACHE_TASKS)}
            for start in range(0, len(items_a), batch_size):
                end = min(start + batch_size, len(items_a))
                feat_a = torch.as_tensor(
                    np.stack([item['feat'] for item in items_a[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                feat_b = torch.as_tensor(
                    np.stack([item['feat'] for item in items_b[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                H_a = torch.as_tensor(
                    np.stack([item['H'] for item in items_a[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                H_b = torch.as_tensor(
                    np.stack([item['H'] for item in items_b[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                feat_pair = 0.5 * (feat_a + feat_b)
                H_pair = 0.5 * (H_a + H_b)
                h2_proj = {}
                if getattr(model, 'h2a_enabled', False):
                    if not (_has_field(items_a[start:end], 'h2_desc') and _has_field(items_b[start:end], 'h2_desc')):
                        continue
                    for name in getattr(model, 'CACHE_METRICS', CACHE_TASKS):
                        desc_a = torch.as_tensor(
                            np.stack([item['h2_desc'][name] for item in items_a[start:end]], axis=0),
                            dtype=torch.float32, device=device,
                        )
                        desc_b = torch.as_tensor(
                            np.stack([item['h2_desc'][name] for item in items_b[start:end]], axis=0),
                            dtype=torch.float32, device=device,
                        )
                        raw = model._build_h2a_pair_features(desc_a, desc_b)
                        h2_proj[name] = model.h2a_pair_proj(raw)

                for name in getattr(model, 'CACHE_METRICS', CACHE_TASKS):
                    metric_idx = TASK_INDEX[name]
                    parts = [feat_pair, H_pair]
                    if name in h2_proj:
                        parts.append(h2_proj[name])
                    sens = torch.as_tensor(
                        np.abs(pred_delta[start:end, metric_idx:metric_idx + 1]),
                        dtype=torch.float32, device=device,
                    )
                    parts.append(torch.log1p(sens))
                    score_buffers[name].append(
                        model.score_heads[name](torch.cat(parts, dim=-1)).squeeze(-1).cpu().numpy()
                    )
            for name, values in score_buffers.items():
                if values:
                    scores[name] = np.concatenate(values, axis=0)

        if (
            getattr(model, 'nc3_enabled', False)
            and _has_field(items_a, 'feat') and _has_field(items_b, 'feat')
            and _has_field(items_a, 'H') and _has_field(items_b, 'H')
        ):
            target_metrics = tuple(getattr(model, 'nc3_target_metrics', ()))
            effective_h6_targets = set(getattr(model, 'h6_target_metrics_effective', ()))
            score_buffers = {name: [] for name in target_metrics}
            for start in range(0, len(items_a), batch_size):
                end = min(start + batch_size, len(items_a))
                feat_a = torch.as_tensor(
                    np.stack([item['feat'] for item in items_a[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                feat_b = torch.as_tensor(
                    np.stack([item['feat'] for item in items_b[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                H_a = torch.as_tensor(
                    np.stack([item['H'] for item in items_a[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                H_b = torch.as_tensor(
                    np.stack([item['H'] for item in items_b[start:end]], axis=0),
                    dtype=torch.float32, device=device,
                )
                feat_pair = 0.5 * (feat_a + feat_b)
                H_pair = 0.5 * (H_a + H_b)
                h6_pair = None
                if _has_field(items_a[start:end], 'h6_feat') and _has_field(items_b[start:end], 'h6_feat'):
                    h6_a = torch.as_tensor(
                        np.stack([item['h6_feat'] for item in items_a[start:end]], axis=0),
                        dtype=torch.float32, device=device,
                    )
                    h6_b = torch.as_tensor(
                        np.stack([item['h6_feat'] for item in items_b[start:end]], axis=0),
                        dtype=torch.float32, device=device,
                    )
                    h6_pair = 0.5 * (h6_a + h6_b)
                for name in target_metrics:
                    metric_idx = TASK_INDEX[name]
                    parts = [feat_pair, H_pair]
                    sens = torch.as_tensor(
                        np.abs(pred_delta[start:end, metric_idx:metric_idx + 1]),
                        dtype=torch.float32, device=device,
                    )
                    parts.append(torch.log1p(sens))
                    if h6_pair is not None and name in effective_h6_targets:
                        parts.append(h6_pair)
                    score_buffers[name].append(
                        model.nc3_score_heads[name](torch.cat(parts, dim=-1)).squeeze(-1).cpu().numpy()
                    )
            for name, values in score_buffers.items():
                if values:
                    scores[name] = np.concatenate(values, axis=0)
    return scores


def _mean_metric(metric_map, tasks, key):
    vals = [metric_map[t][key] for t in tasks if t in metric_map]
    return round(float(np.mean(vals)), 4) if vals else 0.0


def compute_pairwise_metrics(all_data, model=None, device=None, seed=SEED):
    """Compute raw-delta and sensitivity metrics from collected predictions."""
    groups = defaultdict(list)
    for item in all_data:
        groups[item['group_id']].append(item)

    all_gt, all_pred = [], []
    for gid, items in groups.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                all_gt.append(items[i]['label'] - items[j]['label'])
                all_pred.append(items[i]['pred'] - items[j]['pred'])

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)

    ranking = {}
    sensitivity = {}
    for t, task in enumerate(TASK_KEYS):
        gt, pred = all_gt[:, t], all_pred[:, t]
        nonzero = np.sign(gt) != 0
        dir_acc = (np.sign(gt[nonzero]) == np.sign(pred[nonzero])).mean() if nonzero.sum() > 0 else 0.0
        ranking[task] = {
            'spearman': round(float(safe_spearman(gt, pred)), 4),
            'direction_acc': round(float(dir_acc), 4),
        }
        sensitivity[task] = {
            'spearman': round(float(safe_spearman(np.abs(gt), np.abs(pred))), 4),
        }
    ranking['mean'] = {
        'spearman': round(np.mean([v['spearman'] for k, v in ranking.items() if k != 'mean']), 4),
        'direction_acc': round(np.mean([v['direction_acc'] for k, v in ranking.items() if k != 'mean']), 4),
    }
    sensitivity['mean'] = {
        'spearman': round(np.mean([v['spearman'] for k, v in sensitivity.items() if k != 'mean']), 4),
    }

    # Recall@K via random HW pairs
    hw_groups = build_group_hw_map(all_data)

    all_hw_ids = sorted(set(item['hw_id'] for item in all_data))
    np.random.seed(seed)
    hw_pairs = [(all_hw_ids[i], all_hw_ids[j])
                for i in range(len(all_hw_ids)) for j in range(i + 1, len(all_hw_ids))]
    if len(hw_pairs) > 500:
        idx = np.random.choice(len(hw_pairs), 500, replace=False)
        hw_pairs = [hw_pairs[i] for i in idx]

    k_fractions = [0.01, 0.05, 0.10]
    recall_accum = {task: {f'recall@{int(k*100)}%': [] for k in k_fractions} for task in TASK_KEYS}
    ndcg_accum = {task: {f'ndcg@{int(k*100)}%': [] for k in k_fractions} for task in TASK_KEYS}

    for h1, h2 in hw_pairs:
        gids, items_a, items_b, gt_mat, pred_mat = collect_pair_bundle(hw_groups, h1, h2)
        if gt_mat is None or len(gids) < 10:
            continue
        pair_scores = compute_pair_scores(model, items_a, items_b, device) if model is not None else {}
        for t, task in enumerate(TASK_KEYS):
            gt_abs = np.abs(gt_mat[:, t])
            pred_abs = np.abs(pred_mat[:, t])
            rank_signal = pair_scores.get(task, pred_abs)
            for k_frac in k_fractions:
                k = max(1, int(len(gids) * k_frac))
                recall_accum[task][f'recall@{int(k_frac*100)}%'].append(topk_recall(gt_abs, rank_signal, k))
                ndcg_accum[task][f'ndcg@{int(k_frac*100)}%'].append(ndcg_at_k(gt_abs, rank_signal, k))

    recall = {}
    ndcg = {}
    for task in TASK_KEYS:
        recall[task] = {k: round(float(np.mean(v)), 4) if v else 0.0
                        for k, v in recall_accum[task].items()}
        ndcg[task] = {k: round(float(np.mean(v)), 4) if v else 0.0
                      for k, v in ndcg_accum[task].items()}
    recall['mean'] = {}
    ndcg['mean'] = {}
    for k_frac in k_fractions:
        key = f'recall@{int(k_frac*100)}%'
        recall['mean'][key] = round(np.mean([recall[t][key] for t in TASK_KEYS]), 4)
        ndcg_key = f'ndcg@{int(k_frac*100)}%'
        ndcg['mean'][ndcg_key] = round(np.mean([ndcg[t][ndcg_key] for t in TASK_KEYS]), 4)

    summary = {
        'cache_min_sens': round(min(sensitivity[t]['spearman'] for t in CACHE_TASKS), 4),
        'cache_mean_sens': _mean_metric(sensitivity, CACHE_TASKS, 'spearman'),
        'cache_mean_recall@5%': _mean_metric(recall, CACHE_TASKS, 'recall@5%'),
        'cache_mean_ndcg@5%': _mean_metric(ndcg, CACHE_TASKS, 'ndcg@5%'),
        'noncache_mean_sens': _mean_metric(
            sensitivity, [t for t in TASK_KEYS if t not in CACHE_TASKS], 'spearman'),
        'all_mean_sens': sensitivity['mean']['spearman'],
        'all_mean_recall@5%': recall['mean']['recall@5%'],
        'all_mean_ndcg@5%': ndcg['mean']['ndcg@5%'],
    }

    return {'ranking': ranking, 'sensitivity': sensitivity, 'recall': recall, 'ndcg': ndcg, 'summary': summary}
