"""Hardware-conditioned endpoint prediction and sensitivity scoring.

The released SensiSelect checkpoint uses:
  - Semantic, global-window data/instruction footprint, and branch encoders.
  - An additive hardware encoder and FiLM-based workload/hardware interaction.
  - Hardware-conditioned cache readouts and per-metric endpoint prediction heads.
  - A locality-proxy cache path (H1A_ENABLED) and cache ranking score heads.
  - Paired endpoint predictions for hardware-induced deltas and ranking signals.

Additional flags retain optional branches used for architecture exploration.
These include additive response heads (A1_ENABLED), branch readouts (B1_ENABLED),
raw HOTL features/corrections (H2A_ENABLED/H3A_ENABLED), and non-cache extensions.
They are not enabled in checkpoints/best_model.pth. Their presence in this module
does not make them part of the paper's main-model configuration.

Local fp/ifp and RFP inputs are retained for interface compatibility with
UnifiedLoopModel; the main model uses global-window footprints instead.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unified import (
    SemanticEncoder, FootprintEncoder, BranchEncoder,
    METRIC_KEYS, count_params,
)

# B1: Branch/BTB structured readout — [br_feat, H, branch_hw] MLP shortcut.
# Adds branch-aware features to the head input alongside the global [feat, H].
# No footprint tokens; branch-specific HW slice shared across both metrics.
_B1_CANONICAL_BRANCH_HW = (
    14, 15,                       # btb_entries_log2, ras_entries
    42, 43, 44,                  # bp_BiModeBP, bp_TournamentBP, bp_LTAGE
    1, 2,                         # fetchWidth, decodeWidth
    12, 13,                       # fetchBufferSize, fetchQueueSize
)

B1_BRANCH_HW_IDXS = {
    "branch_mpki": _B1_CANONICAL_BRANCH_HW,
    "btb_miss_rate": _B1_CANONICAL_BRANCH_HW,
}


class BranchReadout(nn.Module):
    """Branch/front-end shortcut: branch statistics + branch-specific HW.

    Takes the raw BranchEncoder output (br_feat, 64-dim branch statistics),
    the global HW encoding H, and a branch-specific HW slice (BTB, RAS,
    predictor type, front-end width/buffering). No footprint tokens.

    The output is a branch-aware feature appended to the head input — the
    head still receives the global [feat, H] interaction as context.
    """

    def __init__(self, hw_cond_dim, dim=256, dropout=0.1):
        super().__init__()
        self.br_proj_in = nn.Linear(64, dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * dim + hw_cond_dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, br_feat, H, hw_cond):
        return self.cond_proj(torch.cat([self.br_proj_in(br_feat), H, hw_cond], dim=-1))

# Stall-relevant HW parameter indices (in the 47-dim HW vector)
N1A_STALL_HW_IDXS = {
    "rob_stall": (8, 9, 5, 6, 4, 3, 10, 11, 24, 32, 25, 33),
    # numROBEntries, numIQEntries, issueWidth, commitWidth, dispatchWidth,
    # renameWidth, LQEntries, SQEntries, l1d_mshrs, l2_mshrs,
    # l1d_latency, l2_latency
    "lq_stall": (10, 24, 32, 25, 33, 5, 4),
    # LQEntries, l1d_mshrs, l2_mshrs, l1d_latency, l2_latency,
    # issueWidth, dispatchWidth
    "sq_stall": (11, 24, 32, 25, 33, 6, 5),
    # SQEntries, l1d_mshrs, l2_mshrs, l1d_latency, l2_latency,
    # commitWidth, issueWidth
}


class FiLMInteraction(nn.Module):
    """Feature-wise Linear Modulation: hardware conditions workload."""

    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim))
        self.fuse = nn.Sequential(
            nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 2 * dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.GELU())
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, w, h):
        gamma, beta = self.to_gamma_beta(h).chunk(2, dim=-1)
        modulated = w * (1.0 + torch.tanh(gamma)) + beta
        crossed = w * h
        residual = self.fuse(torch.cat([modulated, w, h, crossed], dim=-1))
        return self.out_norm(modulated + residual)


class AttentionPool(nn.Module):
    """Learned query attends over tokens -> single pooled vector."""

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        B = tokens.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(out.squeeze(1))


class CrossSwiGLUInteraction(nn.Module):
    """Bidirectional W-H gating."""

    def __init__(self, w_dim=256, h_dim=256, interact_dim=None,
                 expansion_factor=1.5, dropout=0.1):
        super().__init__()
        interact_dim = interact_dim or h_dim
        hidden = int(interact_dim * expansion_factor)
        self.proj_w_gate = nn.Linear(w_dim, hidden, bias=False)
        self.proj_h_gate = nn.Linear(h_dim, hidden, bias=True)
        self.proj_w_val = nn.Linear(w_dim, hidden, bias=False)
        self.proj_h_val = nn.Linear(h_dim, hidden, bias=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, interact_dim),
            nn.LayerNorm(interact_dim), nn.Dropout(dropout))
        self.res_w = nn.Linear(w_dim, interact_dim) if w_dim != interact_dim else nn.Identity()
        self.res_h = nn.Linear(h_dim, interact_dim) if h_dim != interact_dim else nn.Identity()

    def forward(self, W, H):
        gate = F.silu(self.proj_w_gate(W) + self.proj_h_gate(H))
        value = self.proj_w_val(W) + self.proj_h_val(H)
        fused = gate * value
        return self.out_proj(fused) + self.res_w(W) + self.res_h(H)


class HWConditionedReadout(nn.Module):
    """HW-conditioned attention over footprint windows + local change signal.

    Takes arbitrary HW params (e.g. cache size, buffer entries) and conditions
    the attention over footprint tokens on those params. The depthwise conv
    extracts a local change-rate signal from encoded tokens (NOT raw footprint
    derivative). This signal mixes temporal reuse, working-set growth, and
    some spatial information but should NOT be interpreted as a direct physical
    slope or spatial locality proxy.

    True spatial locality is captured by the ratio channels (half/cache,
    cache/dual) in the 8-channel footprint input, which FootprintEncoder
    processes via dilated conv layers.
    """

    def __init__(self, hw_cond_dim, dim=256, dropout=0.1):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(hw_cond_dim, 64), nn.GELU(), nn.Linear(64, dim))
        self.hw_to_query = nn.Linear(dim, dim)
        self.window_attn = nn.MultiheadAttention(
            dim, num_heads=4, dropout=dropout, batch_first=True)
        self.slope_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, fp_tokens, H, hw_cond):
        """fp_tokens: [B, 19, D], H: [B, D], hw_cond: [B, K]"""
        B, N, D = fp_tokens.shape

        cond_emb = self.cond_proj(hw_cond)                              # [B, D]
        query = self.hw_to_query(H + cond_emb).unsqueeze(1)             # [B, 1, D]
        attn_out, attn_weights = self.window_attn(query, fp_tokens, fp_tokens)
        fp_local = attn_out.squeeze(1)                                  # [B, D]

        fp_slope = self.slope_conv(fp_tokens.transpose(1, 2)).transpose(1, 2)
        slope_weighted = (attn_weights @ fp_slope).squeeze(1)           # [B, D]

        return self.out_proj(torch.cat([fp_local, slope_weighted, cond_emb], dim=-1))


class WorkloadConditionedAdditiveHead(nn.Module):
    """Cache-only workload-conditioned additive endpoint response (A1_ENABLED).

    This optional branch is distinct from the paper's grouped-ranking ablation.

    Implements:
      y_m(x, h) = b_m(x) + sum_i a_{m,i}(x) * g_{m,i}(h_i)

    where:
      - b_m(x): workload-only baseline
      - a_{m,i}(x): workload-conditioned coefficient for knob i
      - g_{m,i}(h_i): per-knob 1-d response basis

    The head remains endpoint-based; pair deltas are obtained later by
    subtracting endpoint predictions.
    """

    def __init__(self, workload_dim, knob_indices, hidden=256, basis_hidden=16, dropout=0.1):
        super().__init__()
        self.knob_indices = tuple(int(i) for i in knob_indices)
        self.n_knobs = len(self.knob_indices)
        self.coeff_scale = 0.1
        self.additive_scale = self.n_knobs ** -0.5
        self.baseline = nn.Sequential(
            nn.LayerNorm(workload_dim),
            nn.Linear(workload_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.coeff = nn.Sequential(
            nn.LayerNorm(workload_dim),
            nn.Linear(workload_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_knobs),
        )
        self.response = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, basis_hidden),
                nn.GELU(),
                nn.Linear(basis_hidden, 1),
            )
            for _ in range(self.n_knobs)
        ])
        for fn in self.response:
            nn.init.zeros_(fn[-1].weight)
            nn.init.zeros_(fn[-1].bias)

    def compute_workload_terms(self, workload_ctx):
        baseline = self.baseline(workload_ctx)
        coeff = self.coeff_scale * torch.tanh(self.coeff(workload_ctx))
        return baseline, coeff

    def compute_basis(self, hw):
        hw_sub = hw[:, self.knob_indices]
        return torch.cat(
            [fn(hw_sub[:, i:i + 1]) for i, fn in enumerate(self.response)],
            dim=1,
        )

    def forward(self, workload_ctx, hw, workload_terms=None):
        if workload_terms is None:
            if workload_ctx is None:
                raise ValueError("workload_ctx or workload_terms must be provided")
            baseline, coeff = self.compute_workload_terms(workload_ctx)
        else:
            baseline, coeff = workload_terms
        basis = self.compute_basis(hw)                                # [B, K]
        add = (coeff * basis).sum(dim=1, keepdim=True) * self.additive_scale
        return baseline + add


class GAMHWEncoder(nn.Module):
    """Additive hardware encoder with a residual interaction network.

    Each hardware feature is encoded by a separate univariate MLP. Their sum
    models additive responses; a bottleneck network over the concatenated
    embeddings models cross-parameter interactions. This structure is intended
    to separate individual parameter effects from interactions when hardware
    training configurations are sparse. Generalization is evaluated empirically.

    Reference: Hastie & Tibshirani, "Generalized Additive Models" (1990).
    """

    def __init__(self, n_params=47, hidden=32, out_dim=256, dropout=0.1):
        super().__init__()
        self.n_params = n_params
        # Per-parameter tiny MLPs: 1 → hidden → out_dim
        self.param_fn = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
            for _ in range(n_params)
        ])
        # Residual interaction: concat → bottleneck → out_dim
        self.interaction = nn.Sequential(
            nn.Linear(n_params * out_dim, out_dim * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, hw):
        contributions = [fn(hw[:, i:i + 1]) for i, fn in enumerate(self.param_fn)]
        additive = torch.stack(contributions, dim=1).sum(dim=1)         # [B, D]
        concat = torch.cat(contributions, dim=-1)                        # [B, 47*D]
        residual = self.interaction(concat)                               # [B, D]
        tokens = (additive + residual).unsqueeze(1)                       # [B, 1, D]
        return self.norm(tokens)


class FiLMUnifiedModel(nn.Module):
    """FiLM model with per-metric HW-conditioned footprint readout.

    Cache metrics use HWConditionedReadout. Non-cache metrics use global FiLM
    features unless an optional architecture-exploration branch is enabled.
    N1A_ENABLED adds hardware-conditioned footprint readouts for stall metrics;
    B1_ENABLED adds BranchReadout for branch metrics using front-end hardware
    features without footprint inputs. Neither is enabled in the main checkpoint.
    """

    METRIC_KEYS = METRIC_KEYS

    # Metrics with structured HW-conditioned readout
    CACHE_METRICS = ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate")
    CACHE_SIZE_IDX = {"l1_mpki": 16, "l1i_mpki": 17, "l2_mpki": 18, "l2_miss_rate": 18}

    HW_DIM = 47  # total HW parameter dimensions
    A1_MIN_KNOBS = {
        "l1_mpki": (16, 19),          # l1d_size, l1d_assoc
        "l1i_mpki": (17, 20),         # l1i_size, l1i_assoc
        "l2_mpki": (16, 17, 18, 21),  # l1d_size, l1i_size, l2_size, l2_assoc
        "l2_miss_rate": (18, 21),     # l2_size, l2_assoc
    }

    def __init__(self, embed_dim=256, dropout=0.1, noise_std=0.01, r2_enabled=False,
                 interaction_type="film", hw_encoder="attention",
                 use_residual=False, c2a_alpha=0.0, c3a_enabled=False,
                 nc3_enabled=False, nc3_target_metrics=None, nc3_detach_base=True,
                 j1a_enabled=False, j1a_target_metrics=None, j1a_detach_base=True,
                 a1_enabled=False, a1_basis_hidden=16, a1_knob_mode="full47",
                 h1a_enabled=False, h1a_stats=None,
                 h2a_enabled=False, h2a_stats=None,
                 h2a_concave=True, h2a_above_mode="boundary",
                 h6a_enabled=False, h6a_stats=None, h6_mode="h6a", h6a_proxy_scale=1.0,
                 h6a_target_metrics=None, h6a_slq_scale=None,
                 cachex_enabled=False, cachex_target_metrics=None,
                 h3a_enabled=False, h3a_eta=0.02,
                 h3a_detach_base=False, h3a_mode="full",
                 n1a_enabled=False, b1_enabled=False,
                 s2_enabled=False, s2_alpha=0.10,
                 jepa_enabled=False,
                 direct_delta_enabled=False):
        super().__init__()
        self.n1a_enabled = n1a_enabled
        self.r2_enabled = r2_enabled
        self.b1_enabled = b1_enabled
        self.s2_enabled = s2_enabled
        self.jepa_enabled = jepa_enabled
        self.direct_delta_enabled = direct_delta_enabled
        self.noise_std = noise_std
        self.hw_encoder_type = hw_encoder
        self.use_residual = use_residual
        self.c2a_alpha = c2a_alpha
        self.c3a_enabled = c3a_enabled
        self.nc3_enabled = nc3_enabled
        self.nc3_detach_base = nc3_detach_base
        self.j1a_enabled = j1a_enabled
        self.j1a_detach_base = j1a_detach_base
        self.a1_enabled = a1_enabled
        self.a1_knob_mode = a1_knob_mode
        self.h1a_enabled = h1a_enabled
        self.h2a_enabled = h2a_enabled
        self.h2a_concave = h2a_concave
        self.h2a_above_mode = h2a_above_mode
        self.h6a_enabled = h6a_enabled
        self.h6_mode = h6_mode
        if h6a_slq_scale is not None:
            h6a_proxy_scale = h6a_slq_scale
        self.h6a_proxy_scale = h6a_proxy_scale
        self.h3a_enabled = h3a_enabled
        self.h3a_eta = h3a_eta
        self.cachex_enabled = cachex_enabled
        self.h3a_detach_base = h3a_detach_base
        self.h3a_mode = h3a_mode
        if self.h2a_above_mode not in ("boundary", "zero"):
            raise ValueError("h2a_above_mode must be 'boundary' or 'zero'")
        if self.a1_knob_mode not in ("full47", "min"):
            raise ValueError("a1_knob_mode must be 'full47' or 'min'")
        if self.a1_enabled and (self.h1a_enabled or self.h2a_enabled or self.h3a_enabled):
            raise ValueError("A1_ENABLED cannot be combined with H1A_ENABLED, H2A_ENABLED, or H3A_ENABLED")
        if self.h3a_enabled and not self.h2a_enabled:
            raise ValueError("H3a requires H2a raw HOTL analytic features")
        if self.h3a_mode not in ("full", "prior_only", "residual_only"):
            raise ValueError("h3a_mode must be one of: full, prior_only, residual_only")
        if self.h6a_enabled and self.h6_mode not in ("h6a", "h6c", "h6d"):
            raise ValueError("h6_mode must be one of: h6a, h6c, h6d")
        default_h6a_targets = tuple(k for k in self.METRIC_KEYS if k not in self.CACHE_METRICS)
        if h6a_target_metrics is None:
            h6a_target_metrics = default_h6a_targets
        self.h6a_target_metrics = tuple(h6a_target_metrics)
        invalid_h6a = [k for k in self.h6a_target_metrics if k not in default_h6a_targets]
        if invalid_h6a:
            raise ValueError(
                "H6 target metrics must be non-cache metrics only; got "
                + ", ".join(invalid_h6a)
            )
        self.h6_target_metrics_effective = self.h6a_target_metrics if self.h6a_enabled else ()
        default_nc3_targets = ("cpi", "rob_stall", "lq_stall", "sq_stall")
        if nc3_target_metrics is None:
            nc3_target_metrics = default_nc3_targets
        self.nc3_target_metrics = tuple(nc3_target_metrics)
        invalid_nc3 = [k for k in self.nc3_target_metrics if k not in default_h6a_targets]
        if invalid_nc3:
            raise ValueError(
                "NC3 target metrics must be non-cache metrics only; got "
                + ", ".join(invalid_nc3)
            )
        default_j1a_targets = ("branch_mpki", "btb_miss_rate", "cpi")
        if j1a_target_metrics is None:
            j1a_target_metrics = default_j1a_targets
        self.j1a_target_metrics = tuple(j1a_target_metrics)
        invalid_j1a = [k for k in self.j1a_target_metrics if k not in default_h6a_targets]
        if invalid_j1a:
            raise ValueError(
                "J1A target metrics must be non-cache metrics only; got "
                + ", ".join(invalid_j1a)
            )
        D = embed_dim

        # --- Encoders ---
        self.sem_enc = SemanticEncoder(512, D, dropout)
        self.gwfp_enc = FootprintEncoder(8, 19, D, dropout)
        self.gwifp_enc = FootprintEncoder(8, 19, D, dropout)
        self.br_enc = BranchEncoder(2, 64, dropout)

        # --- Hardware encoder: attention or additive model ---
        if hw_encoder == "gam":
            gam_hidden = int(os.getenv("GAM_HIDDEN", "32"))
            self.hw_encoder = GAMHWEncoder(self.HW_DIM, hidden=gam_hidden, out_dim=D, dropout=dropout)
            self.hw_pool = lambda x: x.squeeze(1)  # [B, 1, D] -> [B, D]
        else:
            self.hw_param_emb = nn.Parameter(torch.randn(self.HW_DIM, D) * 0.02)
            self.hw_val_proj = nn.Sequential(nn.Linear(1, D), nn.GELU())
            self.hw_param_attn = nn.MultiheadAttention(
                D, num_heads=4, dropout=dropout, batch_first=True)
            self.hw_param_norm = nn.LayerNorm(D)

        # --- Linear residual baseline (P2) ---
        if use_residual:
            self.linear_baseline = nn.ModuleDict({
                name: nn.Linear(self.HW_DIM, 1)
                for name in METRIC_KEYS
            })

        # --- Branch projection ---
        self.br_proj = nn.Linear(64, D)

        # --- Attention pooling ---
        self.fp_pool = AttentionPool(D, dropout=dropout)
        if hw_encoder != "gam":
            self.hw_pool = AttentionPool(D, dropout=dropout)

        # --- Global FiLM ---
        if interaction_type == "swiglu":
            self.interaction = CrossSwiGLUInteraction(D, D, D,
                expansion_factor=1.5, dropout=dropout)
        else:
            self.interaction = FiLMInteraction(D, dropout)

        # --- HOTL cache-conditioned readouts ---
        self.cache_readout = nn.ModuleDict({
            name: HWConditionedReadout(1, D, dropout)
            for name in self.CACHE_METRICS
        })
        # B1: Branch/BTB structured readout — front-end-HW-conditioned MLP (no footprint)
        if self.b1_enabled:
            self.branch_readout = nn.ModuleDict({
                name: BranchReadout(len(B1_BRANCH_HW_IDXS[name]), D, dropout)
                for name in ("branch_mpki", "btb_miss_rate")
            })
        if self.a1_enabled:
            if self.a1_knob_mode == "full47":
                a1_knob_indices = {name: tuple(range(self.HW_DIM)) for name in self.CACHE_METRICS}
            else:
                a1_knob_indices = self.A1_MIN_KNOBS
            self.a1_workload_pool = AttentionPool(D, dropout=dropout)
            self.a1_cache_heads = nn.ModuleDict({
                name: WorkloadConditionedAdditiveHead(
                    workload_dim=2 * D,
                    knob_indices=a1_knob_indices[name],
                    hidden=D,
                    basis_hidden=a1_basis_hidden,
                    dropout=dropout,
                )
                for name in self.CACHE_METRICS
            })
        # L2 metrics additionally read instruction footprint (gwifp) for dual-path
        self.use_l2_dual = os.getenv("L2_DUAL_PATH", "0") == "1"
        if self.use_l2_dual:
            self.l2_inst_readout = nn.ModuleDict({
                name: HWConditionedReadout(1, D, dropout)
                for name in ("l2_mpki", "l2_miss_rate")
            })

        # --- C2a: Per-metric delta scale calibration ---
        # c_m = exp(alpha * tanh(u_m)), range [e^-alpha, e^alpha]
        if c2a_alpha > 0:
            self.delta_scale_raw = nn.Parameter(torch.zeros(len(METRIC_KEYS)))

        # --- S2: Pair-dependent bounded sensitivity calibration ---
        # scale_m = exp(alpha * tanh(MLP(pair_features)))
        # Applied per-pair, per-metric. Initialized to identity (scale≈1).
        if s2_enabled:
            self.s2_alpha = s2_alpha
            self.s2_head = nn.Sequential(
                nn.Linear(3 * D + self.HW_DIM, D), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(D, D // 2), nn.GELU(),
                nn.Linear(D // 2, len(METRIC_KEYS)),
            )
            # Zero-init last layer → scale ≈ 1 at start
            nn.init.zeros_(self.s2_head[-1].weight)
            nn.init.zeros_(self.s2_head[-1].bias)

        # --- JEPA: Joint Embedding Predictive Architecture auxiliary ---
        # Predicts representation transition s(x,h_a)+Δh→s(x,h_b) in latent space.
        D = embed_dim
        if jepa_enabled:
            self.jepa_proj = nn.Sequential(
                nn.Linear(2 * D, D), nn.LayerNorm(D), nn.GELU(),
                nn.Linear(D, D),
            )
            # Input: delta(47)+abs_delta(47)+changed_mask(47)+l0(1)+bucket(3) = 145
            self.jepa_pair_enc = nn.Sequential(
                nn.Linear(47 * 2 + 47 + 1 + 3, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 128),
            )
            self.jepa_predictor = nn.Sequential(
                nn.Linear(D + 128, D), nn.LayerNorm(D), nn.GELU(),
                nn.Linear(D, D),
            )

        # --- C3a: Sensitivity score head (cache metrics only, for ranking) ---
        if c3a_enabled:
            # Input per cache metric:
            #   base: [feat, H, log1p(|delta_m|)] -> 2D + 1
            #   H2a additionally contributes pair-aware raw HOTL analytic features.
            self.score_feat_dim = 2 * D + 1 + (D if h2a_enabled else 0)
            self.score_heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(self.score_feat_dim),
                    nn.Linear(self.score_feat_dim, D // 2), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D // 2, D // 4), nn.GELU(),
                    nn.Linear(D // 4, 1),
                    nn.Softplus())
                for name in self.CACHE_METRICS
            })

        if h1a_enabled and h2a_enabled:
            raise ValueError("H1a and H2a are alternative Step-D branches; enable only one at a time")

        # --- H1a/H2a locality statistics for raw/proxy feature decoding ---
        need_hotl_stats = h1a_enabled or h2a_enabled
        hotl_stats = h1a_stats if h1a_enabled else h2a_stats
        if need_hotl_stats:
            if hotl_stats is None:
                raise ValueError("H1a/H2a requires footprint/cache-size statistics for decoding")
            self.register_buffer(
                "h1a_gwfp_stats",
                torch.tensor([
                    float(hotl_stats["gwfp_mean"]),
                    float(hotl_stats["gwfp_std"]),
                ], dtype=torch.float32),
            )
            self.register_buffer(
                "h1a_gwifp_stats",
                torch.tensor([
                    float(hotl_stats["gwifp_mean"]),
                    float(hotl_stats["gwifp_std"]),
                ], dtype=torch.float32),
            )
            self.register_buffer(
                "h1a_cache_hw_mean",
                torch.tensor(hotl_stats["cache_hw_mean"], dtype=torch.float32),
            )
            self.register_buffer(
                "h1a_cache_hw_std",
                torch.tensor(hotl_stats["cache_hw_std"], dtype=torch.float32),
            )

        # --- H1a: HOTL proxy features (feature-only, no auxiliary loss) ---
        if h1a_enabled:
            self.h1a_n_features = 5  # slope, mass, curvature, knee, ws_ratio
            self.h1a_proj = nn.Sequential(
                nn.LayerNorm(self.h1a_n_features),
                nn.Linear(self.h1a_n_features, D),
            )

        # --- H2a: Raw HOTL analytic features (cache heads only) ---
        if h2a_enabled:
            # Endpoint-local analytic features that still preserve an absolute/cache-head path.
            self.h2a_n_features = 8  # mr_hotl(df/dx), mass_per_octave(df/dlog2x), slope_drop(d²fp/dx²), knee_distance, log_xc, ws/cache, below, above
            self.h2a_proj = nn.Sequential(
                nn.LayerNorm(self.h2a_n_features),
                nn.Linear(self.h2a_n_features, D),
            )
        # Pair-aware analytic features for (h_a, h_b).
            self.h2a_pair_n_features = 18
            self.h2a_pair_proj = nn.Sequential(
                nn.LayerNorm(self.h2a_pair_n_features),
                nn.Linear(self.h2a_pair_n_features, D),
            )

        # --- H6: spatial-compression proxy features for non-cache heads ---
        if h6a_enabled:
            if h6a_stats is None:
                raise ValueError("H6 requires gwfp/gwifp ratio statistics for proxy decoding")
            self.register_buffer(
                "h6a_gwfp_ratio_half_stats",
                torch.tensor([
                    float(h6a_stats["gwfp_ratio_half_mean"]),
                    float(h6a_stats["gwfp_ratio_half_std"]),
                ], dtype=torch.float32),
            )
            self.register_buffer(
                "h6a_gwfp_ratio_cache2dual_stats",
                torch.tensor([
                    float(h6a_stats["gwfp_ratio_cache2dual_mean"]),
                    float(h6a_stats["gwfp_ratio_cache2dual_std"]),
                ], dtype=torch.float32),
            )
            self.register_buffer(
                "h6a_gwifp_ratio_half_stats",
                torch.tensor([
                    float(h6a_stats["gwifp_ratio_half_mean"]),
                    float(h6a_stats["gwifp_ratio_half_std"]),
                ], dtype=torch.float32),
            )
            self.register_buffer(
                "h6a_gwifp_ratio_cache2dual_stats",
                torch.tensor([
                    float(h6a_stats["gwifp_ratio_cache2dual_mean"]),
                    float(h6a_stats["gwifp_ratio_cache2dual_std"]),
                ], dtype=torch.float32),
            )
            if self.h6_mode == "h6a":
                self.h6a_n_features = 12
                self.h6_raw_feature_names = (
                    "fp_half_mean",
                    "fp_half_tail_mean",
                    "fp_half_max",
                    "fp_cache2dual_mean",
                    "fp_cache2dual_tail_mean",
                    "fp_cache2dual_max",
                    "ifp_half_mean",
                    "ifp_half_tail_mean",
                    "ifp_half_max",
                    "ifp_cache2dual_mean",
                    "ifp_cache2dual_tail_mean",
                    "ifp_cache2dual_max",
                )
            elif self.h6_mode == "h6c":
                self.h6a_n_features = 16
                self.h6_raw_feature_names = tuple(
                    f"{prefix}_bin{bin_idx}"
                    for prefix in ("fp_half", "fp_cache2dual", "ifp_half", "ifp_cache2dual")
                    for bin_idx in range(4)
                )
            else:
                self.h6a_n_features = 8
                self.h6_raw_feature_names = (
                    "fp_cache2dual_mean",
                    "fp_cache2dual_tail6_mean",
                    "fp_cache2dual_tail3_mean",
                    "fp_cache2dual_max",
                    "ifp_cache2dual_mean",
                    "ifp_cache2dual_tail6_mean",
                    "ifp_cache2dual_tail3_mean",
                    "ifp_cache2dual_max",
                )
            self.h6a_proj = nn.Sequential(
                nn.LayerNorm(self.h6a_n_features),
                nn.Linear(self.h6a_n_features, D),
            )

        # --- H3a: Physics-residual cache delta correction ---
        if h3a_enabled:
            # Symmetric pair context for an antisymmetric signed-delta correction.
            self.h3a_pair_n_features = 16
            self.h3a_pair_proj = nn.Sequential(
                nn.LayerNorm(self.h3a_pair_n_features),
                nn.Linear(self.h3a_pair_n_features, D),
            )
            h3_in_dim = 4 * D  # feat_pair, H_pair, |H_b-H_a|, symmetric HOTL pair embedding
            self.h3a_prior_scale_raw = nn.Parameter(torch.zeros(len(self.CACHE_METRICS)))
            self.h3a_alpha_heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(h3_in_dim),
                    nn.Linear(h3_in_dim, D // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D // 2, 1),
                )
                for name in self.CACHE_METRICS
            })
            self.h3a_residual_heads = nn.ModuleDict()
            for idx, name in enumerate(self.CACHE_METRICS):
                head = nn.Sequential(
                    nn.LayerNorm(h3_in_dim),
                    nn.Linear(h3_in_dim, D // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D // 2, 1),
                )
                alpha_last = self.h3a_alpha_heads[name][-1]
                # Identity init is already guaranteed by prior_scale=0 and
                # residual=0, so alpha does not need an additional hard
                # suppressing bias. Keeping alpha near 0.5 avoids starving the
                # prior-scale branch of gradient at initialization.
                nn.init.zeros_(alpha_last.weight)
                nn.init.constant_(alpha_last.bias, 0.0)
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                self.h3a_residual_heads[name] = head

        # --- NC3: non-cache sensitivity score heads (ranking) ---
        if nc3_enabled:
            self.nc3_score_heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(2 * D + 1 + (D if name in self.h6_target_metrics_effective else 0)),
                    nn.Linear(2 * D + 1 + (D if name in self.h6_target_metrics_effective else 0), D // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(D // 2, D // 4),
                    nn.GELU(),
                    nn.Linear(D // 4, 1),
                    nn.Softplus(),
                )
                for name in self.nc3_target_metrics
            })
        self.j1a_uses_h6 = self.h6a_enabled and any(
            name in self.h6_target_metrics_effective for name in self.j1a_target_metrics
        )
        if j1a_enabled:
            j1_in_dim = 2 * D + 1 + (D if self.j1a_uses_h6 else 0)
            self.j1a_score_head = nn.Sequential(
                nn.LayerNorm(j1_in_dim),
                nn.Linear(j1_in_dim, D // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(D // 2, D // 4),
                nn.GELU(),
                nn.Linear(D // 4, 1),
                nn.Softplus(),
            )

        # --- CacheX: detached endpoint cache-expert signal for non-cache heads (R3) ---
        if cachex_enabled:
            self.cachex_signal_metrics = ("l1_mpki", "l1i_mpki", "l2_mpki", "l2_miss_rate")
            self.cachex_n_signals = len(self.cachex_signal_metrics)  # 4 endpoint scalars
            self.cachex_proj = nn.Sequential(
                nn.LayerNorm(self.cachex_n_signals),
                nn.Linear(self.cachex_n_signals, D),
            )
        self.cachex_target_metrics = set(cachex_target_metrics) if cachex_enabled else set()

        # --- Prediction heads ---
        head_factory = lambda in_dim: nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D // 2, 1))

        if self.direct_delta_enabled:
            direct_pair_dim = 4 * D + 2 * self.HW_DIM
            self.direct_pair_trunk = nn.Sequential(
                nn.LayerNorm(direct_pair_dim),
                nn.Linear(direct_pair_dim, D),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(D, D // 2),
                nn.GELU(),
            )
            self.direct_delta_heads = nn.ModuleDict({
                name: nn.Linear(D // 2, 1)
                for name in METRIC_KEYS
            })

        # Cache heads: 4D for L2 dual-path, 3D for others, +D for H1a/H2a feature branch
        locality_dim = D if (h1a_enabled or h2a_enabled) else 0
        l2_dim = (4 * D if self.use_l2_dual else 3 * D) + locality_dim
        self.cache_heads = nn.ModuleDict({
            name: head_factory(l2_dim if name in ("l2_mpki", "l2_miss_rate") else 3 * D + locality_dim)
            for name in self.CACHE_METRICS
        })

        # Stall-structured readouts: HW-conditioned attention over fp tokens for stall metrics
        self.n1a_stall_metrics = ("rob_stall", "lq_stall", "sq_stall") if n1a_enabled else ()
        if n1a_enabled:
            self.stall_readouts = nn.ModuleDict({
                name: HWConditionedReadout(len(N1A_STALL_HW_IDXS[name]), D, dropout)
                for name in self.n1a_stall_metrics
            })
            self.stall_heads = nn.ModuleDict({
                name: head_factory(3 * D)   # [feat, stall_readout, H]
                for name in self.n1a_stall_metrics
            })

        # Non-cache metrics: [feat, H] plus optional H6/CacheX signals (per-target).
        # Stall metrics excluded when n1a_enabled (handled by stall_heads above).
        n1a_set = set(self.n1a_stall_metrics)
        self.heads = nn.ModuleDict({
            name: head_factory(
                2 * D
                + (D if h6a_enabled and name in self.h6_target_metrics_effective else 0)
                + (D if name in self.cachex_target_metrics else 0)
                + (D if b1_enabled and name in B1_BRANCH_HW_IDXS else 0)
            )
            for name in METRIC_KEYS if name not in self.CACHE_METRICS and name not in n1a_set})

        # R2: rob_adapter — dedicated HW-conditioned path for rob_stall
        if r2_enabled:
            r2_hw_idxs = N1A_STALL_HW_IDXS["rob_stall"]
            self.r2_hw_idxs = r2_hw_idxs
            self.rob_adapter = nn.Sequential(
                nn.Linear(2 * D + len(r2_hw_idxs), D), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(D, D // 2), nn.GELU(),
                nn.Linear(D // 2, D))
            # Replace rob_stall in heads with adapter-aware version
            if "rob_stall" in self.heads:
                del self.heads["rob_stall"]
            self.heads["rob_stall"] = head_factory(3 * D)  # [feat, H, adapter]

    def _build_hw_tokens(self, hw):
        """Build hardware tokens using attention or the additive encoder."""
        if self.hw_encoder_type == "gam":
            return self.hw_encoder(hw)                               # [B, 1, D]
        B = hw.size(0)
        param_vals = self.hw_val_proj(hw.unsqueeze(-1))              # [B, 47, D]
        param_emb = self.hw_param_emb.unsqueeze(0).expand(B, -1, -1)
        tokens = param_emb + param_vals
        tokens, _ = self.hw_param_attn(tokens, tokens, tokens)
        return self.hw_param_norm(tokens)                            # [B, 47, D]

    def _decode_cacheline_curve(self, curve_z, stats):
        """Approximate raw cache-line footprint from normalized gwfp/gwifp input."""
        mean, std = stats[0], stats[1]
        curve_log = curve_z * std + mean
        return torch.expm1(curve_log).clamp_min(0.0)

    def _decode_cache_lines(self, hw, cs_idx):
        """Recover cache capacity in 64B lines from normalized HW vector."""
        stat_idx = {16: 0, 17: 1, 18: 2}[cs_idx]
        log2_bytes = hw[:, cs_idx] * self.h1a_cache_hw_std[stat_idx] + self.h1a_cache_hw_mean[stat_idx]
        return torch.pow(2.0, log2_bytes - 6.0).clamp_min(1.0)

    def _decode_ratio_channel(self, ratio_z, stats):
        """Recover raw footprint ratio from normalized cross-granularity channels."""
        mean, std = stats[0], stats[1]
        return (ratio_z * std + mean).clamp_min(1.0)

    def _ratio_to_spatial_compression(self, ratio):
        """Map footprint ratio to a bounded spatial-compression proxy score."""
        return (self.h6a_proxy_scale * (1.0 - ratio.reciprocal())).clamp(0.0, 1.0)

    def _aggregate_proxy_features(self, proxy_curve):
        """Summarize one spatial-compression proxy curve over execution-window scales."""
        tail = min(6, proxy_curve.size(1))
        return torch.stack([
            proxy_curve.mean(dim=1),
            proxy_curve[:, -tail:].mean(dim=1),
            proxy_curve.max(dim=1).values,
        ], dim=1)

    def _aggregate_window_bin_features(self, proxy_curve, n_bins=4):
        """Summarize one proxy curve by coarse execution-window bins.

        H6c is a processed-feature proxy for "reuse bins": we do not have the
        exact Gu et al. reuse-signature transition map in the dataset, so we
        approximate bin-sensitive behavior by splitting the 19 execution-window
        samples into coarse bins and averaging within each region.

        For 19 points and 4 bins the edges are [0, 5, 10, 14, 19], giving:
          bin0  short-window (0-4)
          bin1  mid-window (5-9)
          bin2  long-window (10-13)
          bin3  very-long-window (14-18)
        """
        edges = [0, 5, 10, 14, 19]
        features = []
        for start, end in zip(edges[:-1], edges[1:]):
            features.append(proxy_curve[:, start:end].mean(dim=1))
        return torch.stack(features, dim=1)

    def _aggregate_adjacent_block_features(self, proxy_curve):
        """Summarize cache->dual-block compression for adjacent-block locality."""
        tail6 = min(6, proxy_curve.size(1))
        tail3 = min(3, proxy_curve.size(1))
        return torch.stack([
            proxy_curve.mean(dim=1),
            proxy_curve[:, -tail6:].mean(dim=1),
            proxy_curve[:, -tail3:].mean(dim=1),
            proxy_curve.max(dim=1).values,
        ], dim=1)

    def _shared_sensitivity_target_like(self, terms):
        """Aggregate metric sensitivities into a batch-invariant shared signal.

        J1a operates in the dataset's normalized label space, so per-metric
        sensitivities are already on comparable scales. Keep the aggregation
        strictly sample-wise to avoid leaking batch composition into `j_score`.
        """
        if not terms:
            raise ValueError("shared sensitivity aggregation requires at least one metric term")
        st = torch.stack(terms, dim=0)                                # [M, B, 1]
        return st.mean(dim=0)                                         # [B, 1]

    def _compute_h6a_proxy_features(self, gwfp, gwifp):
        """Build H6a proxy summaries from global-window multi-granularity footprint ratios."""
        fp_half = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwfp[:, 3, :], self.h6a_gwfp_ratio_half_stats)
        )
        fp_cache2dual = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwfp[:, 4, :], self.h6a_gwfp_ratio_cache2dual_stats)
        )
        ifp_half = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwifp[:, 3, :], self.h6a_gwifp_ratio_half_stats)
        )
        ifp_cache2dual = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwifp[:, 4, :], self.h6a_gwifp_ratio_cache2dual_stats)
        )
        return torch.cat([
            self._aggregate_proxy_features(fp_half),
            self._aggregate_proxy_features(fp_cache2dual),
            self._aggregate_proxy_features(ifp_half),
            self._aggregate_proxy_features(ifp_cache2dual),
        ], dim=1)

    def _compute_h6c_proxy_features(self, gwfp, gwifp):
        """Build H6c proxy summaries by coarse execution-window bins."""
        fp_half = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwfp[:, 3, :], self.h6a_gwfp_ratio_half_stats)
        )
        fp_cache2dual = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwfp[:, 4, :], self.h6a_gwfp_ratio_cache2dual_stats)
        )
        ifp_half = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwifp[:, 3, :], self.h6a_gwifp_ratio_half_stats)
        )
        ifp_cache2dual = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwifp[:, 4, :], self.h6a_gwifp_ratio_cache2dual_stats)
        )
        return torch.cat([
            self._aggregate_window_bin_features(fp_half),
            self._aggregate_window_bin_features(fp_cache2dual),
            self._aggregate_window_bin_features(ifp_half),
            self._aggregate_window_bin_features(ifp_cache2dual),
        ], dim=1)

    def _compute_h6d_proxy_features(self, gwfp, gwifp):
        """Build H6d adjacent-block locality proxies from cache->dual compression."""
        fp_cache2dual = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwfp[:, 4, :], self.h6a_gwfp_ratio_cache2dual_stats)
        )
        ifp_cache2dual = self._ratio_to_spatial_compression(
            self._decode_ratio_channel(gwifp[:, 4, :], self.h6a_gwifp_ratio_cache2dual_stats)
        )
        return torch.cat([
            self._aggregate_adjacent_block_features(fp_cache2dual),
            self._aggregate_adjacent_block_features(ifp_cache2dual),
        ], dim=1)

    def _compute_h6_proxy_features(self, gwfp, gwifp):
        """Dispatch H6 proxy feature construction by mode."""
        if self.h6_mode == "h6a":
            return self._compute_h6a_proxy_features(gwfp, gwifp)
        if self.h6_mode == "h6c":
            return self._compute_h6c_proxy_features(gwfp, gwifp)
        return self._compute_h6d_proxy_features(gwfp, gwifp)

    def _smooth_monotone_concave_curve(self, curve, windows):
        """Apply simple monotone + concave smoothing on the 19-point footprint curve.

        Smooth decoded footprint curves before extracting analytic features
        (H2A_ENABLED). Enforce:
          - monotone non-decreasing footprint
          - non-increasing local slope on the execution-window axis
        """
        curve = torch.cummax(curve, dim=1).values
        window_delta = (windows[1:] - windows[:-1]).clamp_min(1.0)
        slopes_x = ((curve[:, 1:] - curve[:, :-1]) / window_delta).clamp_min(0.0)
        if not self.h2a_concave:
            return curve, slopes_x
        slopes_x = torch.cummin(slopes_x, dim=1).values
        first = curve[:, :1]
        curve_smooth = torch.cat([first, first + torch.cumsum(slopes_x * window_delta, dim=1)], dim=1)
        return curve_smooth, slopes_x

    def _compute_h1a_proxy(self, curve, cache_lines):
        """H1a: locality-derived proxy features from decoded gwfp/gwifp cacheline curves.

        The 19 footprint samples are execution-window sizes 2^6, 2^7, ... and
        must not be treated as cache-size buckets. We therefore locate the cache
        capacity on the footprint value axis (fp(x) ~= cache_lines), then derive
        local slope / curvature / knee / working-set proxies around that point.
        """
        B, L = curve.shape
        device = curve.device
        windows = torch.pow(2.0, torch.arange(6, 6 + L, device=device, dtype=curve.dtype))
        curve = torch.cummax(curve, dim=1).values

        covered_mask = cache_lines >= curve[:, -1]
        cross_idx = torch.searchsorted(curve.contiguous(), cache_lines.unsqueeze(1), right=False).squeeze(1)
        cross_idx = cross_idx.clamp(0, L - 1)
        seg_idx = cross_idx.clamp(1, L - 1) - 1
        next_idx = (seg_idx + 1).clamp(max=L - 1)

        window_delta = (windows[next_idx] - windows[seg_idx]).clamp_min(1.0)
        slope = (curve.gather(1, next_idx.unsqueeze(1)).squeeze(1) -
                 curve.gather(1, seg_idx.unsqueeze(1)).squeeze(1)) / window_delta

        mass = (curve.gather(1, next_idx.unsqueeze(1)).squeeze(1) -
                curve.gather(1, seg_idx.unsqueeze(1)).squeeze(1))

        center_idx = seg_idx.clamp(1, L - 2)
        curv = (
            curve.gather(1, (center_idx + 1).unsqueeze(1)).squeeze(1)
            - 2.0 * curve.gather(1, center_idx.unsqueeze(1)).squeeze(1)
            + curve.gather(1, (center_idx - 1).unsqueeze(1)).squeeze(1)
        )

        all_slopes = (curve[:, 1:] - curve[:, :-1]) / (windows[1:] - windows[:-1]).clamp_min(1.0)
        knee_idx = all_slopes.argmax(dim=1)
        knee = (cross_idx.float() - knee_idx.float()) / max(L - 1, 1)

        ws_ratio = curve[:, -1] / cache_lines

        slope = slope.masked_fill(covered_mask, 0.0)
        mass = mass.masked_fill(covered_mask, 0.0)
        curv = curv.masked_fill(covered_mask, 0.0)

        return torch.stack([slope, mass, curv, knee, ws_ratio], dim=-1)

    def _compute_h2a_endpoint_features(self, curve, cache_lines):
        """H2a endpoint features from decoded 19-point cacheline curves.

        We use the discrete HOTL conversion on the execution-window grid
        x in {2^6, ..., 2^24}. Given cache capacity c (in 64B lines), find
        fp(x) ~= c, then derive mr_hotl(c) from the local piecewise slope.
        """
        B, L = curve.shape
        device = curve.device
        dtype = curve.dtype
        windows = torch.pow(2.0, torch.arange(6, 6 + L, device=device, dtype=dtype))
        curve, slopes_x = self._smooth_monotone_concave_curve(curve, windows)
        mass_per_octave_seq = (curve[:, 1:] - curve[:, :-1]).clamp_min(0.0)

        below = cache_lines < curve[:, 0]
        above = cache_lines > curve[:, -1]

        cross_idx = torch.searchsorted(curve.contiguous(), cache_lines.unsqueeze(1), right=False).squeeze(1)
        cross_idx = cross_idx.clamp(0, L - 1)
        seg_idx = (cross_idx - 1).clamp(0, L - 2)
        next_idx = (seg_idx + 1).clamp(max=L - 1)

        lo_fp = curve.gather(1, seg_idx.unsqueeze(1)).squeeze(1)
        hi_fp = curve.gather(1, next_idx.unsqueeze(1)).squeeze(1)
        lo_x = windows[seg_idx]
        hi_x = windows[next_idx]
        seg_slope_x = slopes_x.gather(1, seg_idx.unsqueeze(1)).squeeze(1)
        seg_mass_per_octave = mass_per_octave_seq.gather(1, seg_idx.unsqueeze(1)).squeeze(1)

        denom_fp = (hi_fp - lo_fp).clamp_min(1e-8)
        alpha = ((cache_lines - lo_fp) / denom_fp).clamp(0.0, 1.0)
        x_c = lo_x + alpha * (hi_x - lo_x)
        log_x_c = torch.log2(x_c.clamp_min(1.0))

        mr_hotl = torch.where(below, slopes_x[:, 0], seg_slope_x)
        mass_per_octave = torch.where(below, mass_per_octave_seq[:, 0], seg_mass_per_octave)
        if self.h2a_above_mode == "zero":
            mr_hotl = torch.where(above, torch.zeros_like(mr_hotl), mr_hotl)
            mass_per_octave = torch.where(above, torch.zeros_like(mass_per_octave), mass_per_octave)
        else:
            mr_hotl = torch.where(above, slopes_x[:, -1], mr_hotl)
            mass_per_octave = torch.where(above, mass_per_octave_seq[:, -1], mass_per_octave)

        center_idx = seg_idx.clamp(1, L - 2)
        curv_left = slopes_x.gather(1, (center_idx - 1).unsqueeze(1)).squeeze(1)
        curv_right = slopes_x.gather(1, center_idx.unsqueeze(1)).squeeze(1)
        curvature = curv_left - curv_right  # slope_drop > 0 for concave footprint
        curvature = torch.where(below | above, torch.zeros_like(curvature), curvature)

        slope_drop = slopes_x[:, :-1] - slopes_x[:, 1:]
        knee_idx = slope_drop.argmax(dim=1)
        knee_x = windows[(knee_idx + 1).clamp(max=L - 1)]
        knee_distance = torch.log2(x_c.clamp_min(1.0)) - torch.log2(knee_x.clamp_min(1.0))

        ws_ratio = curve[:, -1] / cache_lines
        log_ws = torch.log(ws_ratio.clamp_min(1e-8))

        return torch.stack([
            mr_hotl,
            mass_per_octave,
            curvature,
            knee_distance,
            log_x_c,
            log_ws,
            below.float(),
            above.float(),
        ], dim=-1)

    def _build_h2a_pair_features(self, feat_a, feat_b):
        """Pair-aware HOTL analytic features for (h_a, h_b) (H2A_ENABLED)."""
        mr_a, mr_b = feat_a[:, 0], feat_b[:, 0]
        slope_a, slope_b = feat_a[:, 1], feat_b[:, 1]
        curv_a, curv_b = feat_a[:, 2], feat_b[:, 2]
        knee_a, knee_b = feat_a[:, 3], feat_b[:, 3]
        log_xc_a, log_xc_b = feat_a[:, 4], feat_b[:, 4]
        ws_a, ws_b = feat_a[:, 5], feat_b[:, 5]
        below_a, above_a = feat_a[:, 6], feat_a[:, 7]
        below_b, above_b = feat_b[:, 6], feat_b[:, 7]

        abs_delta_mr = (mr_b - mr_a).abs()
        abs_delta_log_xc = (log_xc_b - log_xc_a).abs()

        return torch.stack([
            mr_a,
            mr_b,
            abs_delta_mr,
            slope_a,
            slope_b,
            curv_a,
            curv_b,
            knee_a,
            knee_b,
            log_xc_a,
            log_xc_b,
            abs_delta_log_xc,
            ws_a,
            ws_b,
            below_a,
            above_a,
            below_b,
            above_b,
        ], dim=-1)

    def _build_h3a_pair_features(self, feat_a, feat_b):
        """Return symmetric H3a context plus signed HOTL prior.

        H3a predicts a signed cache-delta correction, so the correction must be
        antisymmetric under swapping (h_a, h_b). We therefore construct:
          - a symmetric context feature vector
          - a signed prior delta_mr_hotl
        and apply the correction as signed_prior * scale(sym_ctx).
        """
        mr_a, mr_b = feat_a[:, 0], feat_b[:, 0]
        mass_a, mass_b = feat_a[:, 1], feat_b[:, 1]
        curv_a, curv_b = feat_a[:, 2], feat_b[:, 2]
        knee_a, knee_b = feat_a[:, 3], feat_b[:, 3]
        log_xc_a, log_xc_b = feat_a[:, 4], feat_b[:, 4]
        ws_a, ws_b = feat_a[:, 5], feat_b[:, 5]
        below_a, above_a = feat_a[:, 6], feat_a[:, 7]
        below_b, above_b = feat_b[:, 6], feat_b[:, 7]

        delta_mr = mr_b - mr_a
        delta_mass = mass_b - mass_a
        delta_log_xc = log_xc_b - log_xc_a
        delta_ws = ws_b - ws_a

        below_any = torch.maximum(below_a, below_b)
        above_any = torch.maximum(above_a, above_b)
        below_both = below_a * below_b
        above_both = above_a * above_b

        sym = torch.stack([
            0.5 * (mr_a + mr_b),
            delta_mr.abs(),
            0.5 * (mass_a + mass_b),
            delta_mass.abs(),
            0.5 * (curv_a + curv_b),
            (curv_b - curv_a).abs(),
            0.5 * (knee_a + knee_b),
            (knee_b - knee_a).abs(),
            0.5 * (log_xc_a + log_xc_b),
            delta_log_xc.abs(),
            0.5 * (ws_a + ws_b),
            delta_ws.abs(),
            below_any,
            above_any,
            below_both,
            above_both,
        ], dim=-1)
        return sym, delta_mr.unsqueeze(1)

    def _encode_workload(self, bbe, br, gwfp, gwifp):
        h_sem = self.sem_enc(bbe)
        gwfp_tokens = self.gwfp_enc(gwfp)                             # [B, 19, D]
        gwifp_tokens = self.gwifp_enc(gwifp)                          # [B, 19, D]
        br_feat = self.br_enc(br)                                     # [B, 64]
        all_fp = torch.cat([gwfp_tokens, gwifp_tokens], dim=1)        # [B, 38, D]
        W = h_sem + self.fp_pool(all_fp) + self.br_proj(br_feat)
        return {
            "h_sem": h_sem,
            "gwfp_tokens": gwfp_tokens,
            "gwifp_tokens": gwifp_tokens,
            "br_feat": br_feat,
            "all_fp": all_fp,
            "W": W,
        }

    def _build_a1_workload_terms(self, workload_cache):
        W = workload_cache["W"]
        gwfp_tokens = workload_cache["gwfp_tokens"]
        gwifp_tokens = workload_cache["gwifp_tokens"]
        all_fp = workload_cache["all_fp"]
        a1_terms = {}
        for name in self.CACHE_METRICS:
            if name == "l1i_mpki":
                metric_tokens = gwifp_tokens
            elif name in ("l2_mpki", "l2_miss_rate"):
                metric_tokens = all_fp
            else:
                metric_tokens = gwfp_tokens
            workload_ctx = torch.cat([W, self.a1_workload_pool(metric_tokens)], dim=-1)
            a1_terms[name] = self.a1_cache_heads[name].compute_workload_terms(workload_ctx)
        return a1_terms

    def forward_absolute(
        self,
        fp,
        ifp,
        hw,
        rfp=None,
        bbe=None,
        br=None,
        gwfp=None,
        gwifp=None,
        workload_cache=None,
        a1_terms=None,
    ):
        # rfp/fp/ifp intentionally unused
        del fp, ifp, rfp
        hw_raw = hw
        hw_model = hw
        if self.training and self.noise_std > 0:
            hw_model = hw + torch.randn_like(hw) * self.noise_std

        if workload_cache is None:
            workload_cache = self._encode_workload(bbe, br, gwfp, gwifp)
        h_sem = workload_cache["h_sem"]
        D = h_sem.size(-1)
        gwfp_tokens = workload_cache["gwfp_tokens"]
        gwifp_tokens = workload_cache["gwifp_tokens"]
        br_feat = workload_cache["br_feat"]
        all_fp = workload_cache["all_fp"]
        W = workload_cache["W"]

        # Hardware: per-parameter attention → pool
        hw_tokens = self._build_hw_tokens(hw_model)                    # [B, 47, D]
        H = self.hw_pool(hw_tokens)                                    # [B, D]

        feat = self.interaction(W, H)                                  # [B, D]

        locality_feats = {}
        if self.h1a_enabled:
            fp_cacheline = self._decode_cacheline_curve(gwfp[:, 1, :], self.h1a_gwfp_stats)
            ifp_cacheline = self._decode_cacheline_curve(gwifp[:, 1, :], self.h1a_gwifp_stats)
            l2_cacheline = fp_cacheline + ifp_cacheline
            for name in self.CACHE_METRICS:
                cs_idx = self.CACHE_SIZE_IDX[name]
                cache_lines = self._decode_cache_lines(hw_raw, cs_idx)
                proxy_curve = ifp_cacheline if name == "l1i_mpki" else fp_cacheline
                if name in ("l2_mpki", "l2_miss_rate"):
                    proxy_curve = l2_cacheline
                ft = self._compute_h1a_proxy(proxy_curve, cache_lines)
                locality_feats[name] = self.h1a_proj(ft)
        elif self.h2a_enabled:
            fp_cacheline = self._decode_cacheline_curve(gwfp[:, 1, :], self.h1a_gwfp_stats)
            ifp_cacheline = self._decode_cacheline_curve(gwifp[:, 1, :], self.h1a_gwifp_stats)
            l2_cacheline = fp_cacheline + ifp_cacheline
            h2_desc = {}
            for name in self.CACHE_METRICS:
                cs_idx = self.CACHE_SIZE_IDX[name]
                cache_lines = self._decode_cache_lines(hw_raw, cs_idx)
                raw_curve = ifp_cacheline if name == "l1i_mpki" else fp_cacheline
                if name in ("l2_mpki", "l2_miss_rate"):
                    raw_curve = l2_cacheline
                ft = self._compute_h2a_endpoint_features(raw_curve, cache_lines)
                h2_desc[name] = ft
                locality_feats[name] = self.h2a_proj(ft)

        absolute = {}
        a1_baseline = {}
        a1_coeff = {}
        a1_basis = {}
        a1_add = {}
        a1_contrib = {}

        # --- Cache metrics: HOTL conditioned readout ---
        for name in self.CACHE_METRICS:
            if self.a1_enabled:
                if a1_terms is None:
                    a1_terms = self._build_a1_workload_terms(workload_cache)
                head = self.a1_cache_heads[name]
                baseline, coeff = a1_terms[name]
                basis = head.compute_basis(hw_model)
                contrib = coeff * basis * head.additive_scale
                add = contrib.sum(dim=1, keepdim=True)
                pred = baseline + add
                a1_baseline[name] = baseline
                a1_coeff[name] = coeff
                a1_basis[name] = basis
                a1_add[name] = add
                a1_contrib[name] = contrib
            else:
                cs_idx = self.CACHE_SIZE_IDX[name]
                cache_size = hw_model[:, cs_idx:cs_idx + 1]
                fp_tokens = gwifp_tokens if name == "l1i_mpki" else gwfp_tokens
                cr = self.cache_readout[name](fp_tokens, H, cache_size)
                if self.use_l2_dual and name in self.l2_inst_readout:
                    cr_inst = self.l2_inst_readout[name](gwifp_tokens, H, cache_size)
                    cache_parts = [feat, cr, cr_inst, H]
                else:
                    cache_parts = [feat, cr, H]
                if self.h1a_enabled or self.h2a_enabled:
                    cache_parts.append(locality_feats[name])
                cache_in = torch.cat(cache_parts, dim=-1)
                pred = self.cache_heads[name](cache_in)
                if self.use_residual:
                    pred = pred + self.linear_baseline[name](hw_model)
            absolute[name] = pred

        # --- Stall metrics: HW-conditioned readout with stall-specific HW params ---
        for name in self.n1a_stall_metrics:
            stall_hw_slice = hw_model[:, N1A_STALL_HW_IDXS[name]]
            sr = self.stall_readouts[name](gwfp_tokens, H, stall_hw_slice)
            stall_in = torch.cat([feat, sr, H], dim=-1)
            pred = self.stall_heads[name](stall_in)
            if self.use_residual:
                pred = pred + self.linear_baseline[name](hw_model)
            absolute[name] = pred

        # --- Non-cache metrics: global FiLM plus optional cache-expert / H6 signals ---
        cachex_feat = None
        if self.cachex_enabled:
            cachex_signals = torch.cat([
                absolute[m] for m in self.cachex_signal_metrics
                if m in absolute
            ], dim=-1).detach()
            cachex_feat = self.cachex_proj(cachex_signals)
        h6a_raw = None
        h6a_feat = None
        if self.h6a_enabled:
            h6a_raw = self._compute_h6_proxy_features(gwfp, gwifp)
            h6a_feat = self.h6a_proj(h6a_raw)
        # R2: pre-compute rob_adapter feature
        rob_adapter_feat = None
        if self.r2_enabled:
            rob_hw_slice = hw_model[:, self.r2_hw_idxs]
            rob_adapter_feat = self.rob_adapter(torch.cat([feat, H, rob_hw_slice], dim=-1))

        for name in self.heads:
            non_cache_parts = [feat, H]
            # B1: branch/BTB structured readout (raw branch encoding + front-end HW)
            if self.b1_enabled and name in self.branch_readout:
                br_hw = hw_model[:, B1_BRANCH_HW_IDXS[name]]
                br_out = self.branch_readout[name](br_feat, H, br_hw)
                non_cache_parts.append(br_out)
            if name == "rob_stall" and self.r2_enabled and rob_adapter_feat is not None:
                non_cache_parts.append(rob_adapter_feat)
            if self.cachex_enabled and name in self.cachex_target_metrics:
                non_cache_parts.append(cachex_feat)
            if self.h6a_enabled and name in self.h6_target_metrics_effective:
                non_cache_parts.append(h6a_feat)
            head_in = torch.cat(non_cache_parts, dim=-1)
            pred = self.heads[name](head_in)
            if self.use_residual:
                pred = pred + self.linear_baseline[name](hw_model)
            absolute[name] = pred

        out = {"absolute": absolute}
        if self.cachex_enabled:
            out["_cachex_feat"] = cachex_feat
        if self.h6a_enabled:
            out["_h6_raw"] = h6a_raw
            out["_h6_feat"] = h6a_feat
            out["_h6a_raw"] = h6a_raw
            out["_h6a_feat"] = h6a_feat
        if self.a1_enabled:
            out["_a1_baseline"] = a1_baseline
            out["_a1_coeff"] = a1_coeff
            out["_a1_basis"] = a1_basis
            out["_a1_add"] = a1_add
            out["_a1_contrib"] = a1_contrib
        if (self.c2a_alpha > 0 or self.c3a_enabled or self.nc3_enabled
                or self.j1a_enabled or self.h3a_enabled or self.s2_enabled
                or self.direct_delta_enabled):
            out["_feat"] = feat
            out["_H"] = H
        if self.jepa_enabled:
            out["_jepa_repr"] = self.jepa_proj(torch.cat([feat, H], dim=-1))
        if self.h2a_enabled:
            out["_h2_desc"] = h2_desc
        return out

    def jepa_predict(self, repr_a, pair_features):
        """Predict s_b from (s_a, rich pair features)."""
        p_ab = self.jepa_pair_enc(pair_features)
        return self.jepa_predictor(torch.cat([repr_a, p_ab], dim=-1))

    def forward(self, fp, ifp, hw_a, hw_b, rfp=None, bbe=None, br=None, gwfp=None, gwifp=None,
                hw_delta=None, changed_mask=None):
        workload_cache = None
        a1_terms = None
        if self.a1_enabled:
            workload_cache = self._encode_workload(bbe, br, gwfp, gwifp)
            a1_terms = self._build_a1_workload_terms(workload_cache)
        abs_a = self.forward_absolute(
            fp, ifp, hw_a, None, bbe, br, gwfp, gwifp,
            workload_cache=workload_cache, a1_terms=a1_terms,
        )
        abs_b = self.forward_absolute(
            fp, ifp, hw_b, None, bbe, br, gwfp, gwifp,
            workload_cache=workload_cache, a1_terms=a1_terms,
        )
        if self.direct_delta_enabled:
            feat_a = abs_a["_feat"]
            feat_b = abs_b["_feat"]
            H_a = abs_a["_H"]
            H_b = abs_b["_H"]
            feat_pair = 0.5 * (feat_a + feat_b)
            feat_diff = feat_b - feat_a
            H_pair = 0.5 * (H_a + H_b)
            H_diff = H_b - H_a
            hw_signed = hw_b - hw_a
            pair_ctx = torch.cat(
                [feat_pair, feat_diff, H_pair, H_diff, hw_signed, hw_signed.abs()],
                dim=-1,
            )
            pair_hidden = self.direct_pair_trunk(pair_ctx)
            delta = {
                name: self.direct_delta_heads[name](pair_hidden)
                for name in METRIC_KEYS
            }
        else:
            delta = {k: abs_b["absolute"][k] - abs_a["absolute"][k] for k in METRIC_KEYS}

        # C2a: per-metric delta scale calibration
        if self.c2a_alpha > 0:
            c = torch.exp(self.c2a_alpha * torch.tanh(self.delta_scale_raw))
            scale = {k: c[i:i+1] for i, k in enumerate(METRIC_KEYS)}
            delta = {k: delta[k] * scale[k] for k in METRIC_KEYS}

        # S2: pair-dependent bounded sensitivity calibration
        # scale_m = exp(α * tanh(g_m([H_a, H_b, |H_b-H_a|, |hw_b-hw_a|])))
        if self.s2_enabled:
            H_a = abs_a["_H"]; H_b = abs_b["_H"]
            H_diff = (H_b - H_a).abs()
            hw_d = (hw_b - hw_a).abs()
            pair_in = torch.cat([H_a, H_b, H_diff, hw_d.float()], dim=-1)
            s2_raw = self.s2_head(pair_in)  # [B, M]
            s2_scale = torch.exp(self.s2_alpha * torch.tanh(s2_raw))
            delta = {k: delta[k] * s2_scale[:, i:i+1]
                     for i, k in enumerate(METRIC_KEYS)}

        sensitivity = {k: delta[k].abs() for k in METRIC_KEYS}
        out = {"absolute_a": abs_a["absolute"], "absolute_b": abs_b["absolute"],
               "delta": delta, "sensitivity": sensitivity}
        if self.h6a_enabled:
            out["_h6_raw"] = abs_a.get("_h6_raw")
            out["_h6_feat"] = abs_a.get("_h6_feat")
            out["_h6a_raw"] = abs_a.get("_h6a_raw")
            out["_h6a_feat"] = abs_a.get("_h6a_feat")
        if self.a1_enabled:
            out["_a1_baseline"] = abs_a.get("_a1_baseline", {})
            out["_a1_coeff"] = abs_a.get("_a1_coeff", {})
            out["_a1_add_a"] = abs_a.get("_a1_add", {})
            out["_a1_add_b"] = abs_b.get("_a1_add", {})
            out["_a1_basis_a"] = abs_a.get("_a1_basis", {})
            out["_a1_basis_b"] = abs_b.get("_a1_basis", {})
            out["_a1_contrib_a"] = abs_a.get("_a1_contrib", {})
            out["_a1_contrib_b"] = abs_b.get("_a1_contrib", {})

        h2_pair_proj = {}
        if self.h2a_enabled and self.c3a_enabled:
            desc_a = abs_a.get("_h2_desc", {})
            desc_b = abs_b.get("_h2_desc", {})
            for name in self.CACHE_METRICS:
                if name in desc_a and name in desc_b:
                    raw = self._build_h2a_pair_features(desc_a[name], desc_b[name])
                    h2_pair_proj[name] = self.h2a_pair_proj(raw)

        if self.h3a_enabled:
            feat_a = abs_a.get("_feat", None)
            feat_b = abs_b.get("_feat", None)
            H_a = abs_a.get("_H", None)
            H_b = abs_b.get("_H", None)
            desc_a = abs_a.get("_h2_desc", {})
            desc_b = abs_b.get("_h2_desc", {})
            if feat_a is not None and feat_b is not None and H_a is not None and H_b is not None:
                feat_pair = 0.5 * (feat_a + feat_b)
                H_pair = 0.5 * (H_a + H_b)
                H_diff = (H_b - H_a).abs()
                if self.h3a_detach_base:
                    feat_pair = feat_pair.detach()
                    H_pair = H_pair.detach()
                    H_diff = H_diff.detach()
                h3_prior = {}
                h3_alpha = {}
                h3_prior_scale = {}
                h3_residual = {}
                h3_correction = {}
                h3_base_delta = {
                    name: delta[name].clone()
                    for name in self.CACHE_METRICS
                    if name in delta
                }
                for idx, name in enumerate(self.CACHE_METRICS):
                    if name not in desc_a or name not in desc_b:
                        continue
                    sym_raw, signed_prior = self._build_h3a_pair_features(desc_a[name], desc_b[name])
                    pair_embed = self.h3a_pair_proj(sym_raw)
                    ctx = torch.cat([feat_pair, H_pair, H_diff, pair_embed], dim=-1)
                    alpha = torch.sigmoid(self.h3a_alpha_heads[name](ctx))
                    prior_scale = torch.tanh(self.h3a_prior_scale_raw[idx]).view(1, 1)
                    residual = 0.1 * torch.tanh(self.h3a_residual_heads[name](ctx))
                    base_sign = torch.sign(h3_base_delta[name].detach())
                    if self.h3a_mode == "prior_only":
                        correction = self.h3a_eta * signed_prior * (prior_scale * alpha)
                    elif self.h3a_mode == "residual_only":
                        correction = self.h3a_eta * base_sign * residual
                    else:
                        signed_scale = prior_scale * alpha + residual
                        correction = self.h3a_eta * signed_prior * signed_scale
                    delta[name] = delta[name] + correction
                    sensitivity[name] = delta[name].abs()
                    h3_prior[name] = signed_prior
                    h3_alpha[name] = alpha
                    h3_prior_scale[name] = prior_scale.expand_as(alpha)
                    h3_residual[name] = residual
                    h3_correction[name] = correction
                out["delta"] = delta
                out["sensitivity"] = sensitivity
                out["_h3_prior"] = h3_prior
                out["_h3_alpha"] = h3_alpha
                out["_h3_prior_scale"] = h3_prior_scale
                out["_h3_residual"] = h3_residual
                out["_h3_correction"] = h3_correction
                out["_h3_base_delta"] = h3_base_delta

        # C3a: cache-only sensitivity score head
        if self.c3a_enabled:
            feat_a = abs_a.get("_feat", None)
            feat_b = abs_b.get("_feat", None)
            H_a = abs_a.get("_H", None)
            H_b = abs_b.get("_H", None)
            if feat_a is not None and feat_b is not None and H_a is not None and H_b is not None:
                feat_pair = 0.5 * (feat_a + feat_b)
                H_pair = 0.5 * (H_a + H_b)
                score = {}
                for name in self.CACHE_METRICS:
                    # Input: [feat, H, |delta|]
                    parts = [feat_pair, H_pair]
                    if self.h2a_enabled and name in h2_pair_proj:
                        parts.append(h2_pair_proj[name])
                    parts.append(torch.log1p(sensitivity[name].detach()))
                    si = torch.cat(parts, dim=-1)
                    score[name] = self.score_heads[name](si)
                out["score"] = score
        if self.nc3_enabled:
            feat_a = abs_a.get("_feat", None)
            feat_b = abs_b.get("_feat", None)
            H_a = abs_a.get("_H", None)
            H_b = abs_b.get("_H", None)
            h6_feat_a = abs_a.get("_h6_feat", None)
            h6_feat_b = abs_b.get("_h6_feat", None)
            if feat_a is not None and feat_b is not None and H_a is not None and H_b is not None:
                feat_pair = 0.5 * (feat_a + feat_b)
                H_pair = 0.5 * (H_a + H_b)
                h6_pair = None
                if h6_feat_a is not None and h6_feat_b is not None:
                    h6_pair = 0.5 * (h6_feat_a + h6_feat_b)
                if self.nc3_detach_base:
                    feat_pair = feat_pair.detach()
                    H_pair = H_pair.detach()
                    if h6_pair is not None:
                        h6_pair = h6_pair.detach()
                nc_score = {}
                for name in self.nc3_target_metrics:
                    parts = [feat_pair, H_pair, torch.log1p(sensitivity[name].detach())]
                    if h6_pair is not None and name in self.h6_target_metrics_effective:
                        parts.append(h6_pair)
                    si = torch.cat(parts, dim=-1)
                    nc_score[name] = self.nc3_score_heads[name](si)
                out["nc_score"] = nc_score
        if self.j1a_enabled:
            feat_a = abs_a.get("_feat", None)
            feat_b = abs_b.get("_feat", None)
            H_a = abs_a.get("_H", None)
            H_b = abs_b.get("_H", None)
            h6_feat_a = abs_a.get("_h6_feat", None)
            h6_feat_b = abs_b.get("_h6_feat", None)
            if feat_a is not None and feat_b is not None and H_a is not None and H_b is not None:
                feat_pair = 0.5 * (feat_a + feat_b)
                H_pair = 0.5 * (H_a + H_b)
                h6_pair = None
                if h6_feat_a is not None and h6_feat_b is not None:
                    h6_pair = 0.5 * (h6_feat_a + h6_feat_b)
                if self.j1a_detach_base:
                    feat_pair = feat_pair.detach()
                    H_pair = H_pair.detach()
                    if h6_pair is not None:
                        h6_pair = h6_pair.detach()
                shared_terms = [sensitivity[name].detach() for name in self.j1a_target_metrics]
                if shared_terms:
                    shared_sens = self._shared_sensitivity_target_like(shared_terms)
                    parts = [feat_pair, H_pair, torch.log1p(shared_sens)]
                    if self.j1a_uses_h6 and h6_pair is not None:
                        parts.append(h6_pair)
                    si = torch.cat(parts, dim=-1)
                    out["j_score"] = self.j1a_score_head(si)

        if self.jepa_enabled:
            out["_jepa_repr_a"] = abs_a.get("_jepa_repr")
            out["_jepa_repr_b"] = abs_b.get("_jepa_repr")

        return out


if __name__ == "__main__":
    B = 4
    fp = torch.randn(B, 8, 19)
    ifp = torch.randn(B, 8, 19)
    gwfp = torch.randn(B, 8, 19)
    gwifp = torch.randn(B, 8, 19)
    hw_a = torch.randn(B, 47)
    hw_b = torch.randn(B, 47)
    bbe = torch.randn(B, 512)
    br = torch.randn(B, 2)

    m = FiLMUnifiedModel()
    print(f"FiLMUnifiedModel: {count_params(m):,} params")

    out = m(fp, ifp, hw_a, hw_b, bbe=bbe, br=br, gwfp=gwfp, gwifp=gwifp)
    assert out["absolute_a"]["cpi"].shape == (B, 1)
    assert out["delta"]["cpi"].shape == (B, 1)
    assert out["sensitivity"]["cpi"].shape == (B, 1)

    abs_out = m.forward_absolute(fp, ifp, hw_a, bbe=bbe, br=br, gwfp=gwfp, gwifp=gwifp)
    assert len(abs_out["absolute"]) == 10
    print("All tests passed.")
