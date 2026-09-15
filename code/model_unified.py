"""Unified Loop-Transformer model for 10-metric absolute performance prediction.

Architecture:
  1. SemanticEncoder (BBE 512→D) + gwfp/gwifp encoders (window tokens, no pooling)
     + BranchEncoder + RFPEncoder
  2. HW slice tokenizer: 8 physical-slice tokens for transformer input
  3. Workload token sequence: sem + gwfp(19) + gwifp(19) + br + rfp(3)
     — NO footprint pooling, 19 window tokens per family preserved.
  4. LoopTransformerBlock × 3 (shared weights): iterative refinement over
     workload + hardware token sequence (51 tokens total).
  5. 10 task-specific query tokens, each HW-conditioned via query_hw_proj.
  6. Per-task context projection → per-task MLP heads → scalar prediction.
  7. Pure L_abs training: smooth_l1(pred, target). Sensitivity derived as
     |ŷ(hw_B) - ŷ(hw_A)|.

Residual integrity: transformer is the ONLY path between workload tokens and
task queries. No bypass/shortcut that can skip the transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


METRIC_KEYS = (
    "cpi", "branch_mpki", "l1_mpki", "l1i_mpki", "l2_mpki",
    "rob_stall", "lq_stall", "sq_stall", "l2_miss_rate", "btb_miss_rate",
)

# ============================================================
# 1. Workload encoders
# ============================================================


class SemanticEncoder(nn.Module):
    """BBE 512 → D."""

    def __init__(self, input_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FootprintEncoder(nn.Module):
    """Dilated conv over [B, 8, 19] → per-window tokens [B, 19, D].

    No pooling — each of the 19 windows becomes a token so the transformer
    can learn which window corresponds to a given cache size (HOTL prior).
    """

    def __init__(self, in_channels=8, fp_len=19, embed_dim=256, dropout=0.2):
        super().__init__()
        self.fp_len = fp_len

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1, dilation=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=2, dilation=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=4, dilation=4),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout))
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU())

        self.res1 = nn.Conv1d(in_channels, 64, 1)
        self.res2 = nn.Conv1d(64, 128, 1)
        self.res3 = nn.Conv1d(128, 256, 1)

        self.token_proj = nn.Linear(256, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.window_pos = nn.Parameter(torch.zeros(1, fp_len, embed_dim))
        nn.init.trunc_normal_(self.window_pos, std=0.02)

    def forward(self, x):
        x1 = self.conv1(x) + self.res1(x)
        x2 = self.conv2(x1) + self.res2(x1)
        x3 = self.conv3(x2) + self.res3(x2)
        x4 = self.conv4(x3)  # [B, 256, fp_len]
        tokens = self.token_norm(self.token_proj(x4.transpose(1, 2)) + self.window_pos)
        return tokens


class BranchEncoder(nn.Module):
    """Branch stats [2] → D."""

    def __init__(self, input_dim=2, output_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, output_dim))

    def forward(self, x):
        return self.net(x)


class RFPEncoder(nn.Module):
    """Register dependency [24] → D + 3 tokens (int/float/all groups)."""

    def __init__(self, input_dim=24, output_dim=64, token_dim=256,
                 n_tokens=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.token_projs = nn.ModuleList([
            nn.Linear(output_dim, token_dim) for _ in range(n_tokens)
        ])

    def forward(self, x):
        feat = self.net(x)  # [B, 64]
        tokens = torch.stack([proj(feat) for proj in self.token_projs], dim=1)
        return feat, tokens  # [B, 64], [B, 3, D]


# ============================================================
# 2. Loop Transformer Block
# ============================================================


class LoopTransformerBlock(nn.Module):
    """Shared recurrent transformer block with sandwich normalization."""

    def __init__(self, dim=256, num_heads=4, ff_mult=2.0, dropout=0.1):
        super().__init__()
        ff_dim = int(dim * ff_mult)
        self.attn_norm_in = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm_out = nn.LayerNorm(dim)
        self.ffn_norm_in = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim))
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        # SA with pre-norm + post-norm (sandwich norm)
        attn_in = self.attn_norm_in(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.attn_drop(attn_out)
        x = self.attn_norm_out(x)
        # FFN with pre-norm + post-norm
        ffn_in = self.ffn_norm_in(x)
        x = x + self.ffn_drop(self.ffn(ffn_in))
        x = self.ffn_norm_out(x)
        return x


# ============================================================
# 3. Unified Model
# ============================================================


class UnifiedLoopModel(nn.Module):
    """Loop-Transformer with pure absolute-surface training.

    Forward returns absolute predictions for all 10 metrics.
    Pair delta (for evaluation) is derived: Δŷ = ŷ(hw_B) - ŷ(hw_A).
    """

    METRIC_KEYS = METRIC_KEYS

    def __init__(self, embed_dim=256, dropout=0.1, loop_steps=3,
                 num_heads=4, ff_mult=2.0, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std
        self.loop_steps = loop_steps
        D = embed_dim

        # --- Encoders ---
        self.sem_enc = SemanticEncoder(512, D, dropout)
        self.gwfp_enc = FootprintEncoder(8, 19, D, dropout)    # global-window data FP
        self.gwifp_enc = FootprintEncoder(8, 19, D, dropout)   # global-window inst FP
        self.br_enc = BranchEncoder(2, 64, dropout)
        self.rfp_enc = RFPEncoder(24, 64, D, n_tokens=3, dropout=dropout)

        # --- Workload tokenizer ---
        self.gwfp_token_count = 19
        self.gwifp_token_count = 19
        self.rfp_token_count = 3
        n_scalar = 2  # sem + br
        self.num_wl_tokens = n_scalar + self.gwfp_token_count + self.gwifp_token_count + self.rfp_token_count  # 43

        self.sem_proj = nn.Linear(D, D)
        self.br_proj = nn.Linear(64, D)
        self.wl_token_bias = nn.Parameter(torch.zeros(1, self.num_wl_tokens, D))
        self.wl_dropout = nn.Dropout(dropout)
        self.wl_norm = nn.LayerNorm(D)

        # --- HW tokenizer (8 physical slices → 8 tokens) ---
        self.hw_slices = [
            ([0, 32, 33, 34, 35], "clock_delays"),
            ([1, 2, 3, 4, 5, 6, 7], "widths"),
            ([12, 13, 14, 15, 42, 43, 44], "frontend_bp"),
            ([8, 9, 10, 11], "buffers"),
            ([17, 20, 23, 30], "l1i_cache"),
            ([16, 19, 22, 29, 45, 46], "l1d_cache"),
            ([18, 21, 24, 31], "l2_cache"),
            ([25, 26, 27, 28, 36, 37, 38, 39, 40, 41], "fu_latency"),
        ]
        self.hw_slice_proj = nn.ModuleList([
            nn.Linear(len(inds), D) for inds, _ in self.hw_slices
        ])
        self.hw_slice_indices = [inds for inds, _ in self.hw_slices]
        self.num_hw_tokens = len(self.hw_slices)  # 8
        self.hw_token_bias = nn.Parameter(torch.zeros(1, self.num_hw_tokens, D))
        self.hw_dropout = nn.Dropout(dropout)
        self.hw_norm = nn.LayerNorm(D)

        # --- Loop Transformer (shared weights) ---
        self.loop_block = LoopTransformerBlock(D, num_heads, ff_mult, dropout)

        # --- Task query tokens (10 tasks) ---
        self.num_tasks = len(METRIC_KEYS)
        self.task_queries = nn.Parameter(torch.randn(1, self.num_tasks, D) * 0.02)
        # HW-conditioned query modulation
        self.query_hw_proj = nn.Linear(47, self.num_tasks * D)

        # --- Per-task context projection ---
        self.task_ctx = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(D, D), nn.LayerNorm(D),
            )
            for name in METRIC_KEYS
        })

        # --- Per-task prediction heads ---
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(D // 2, 1),
            )
            for name in METRIC_KEYS
        })

    def _build_wl_tokens(self, h_sem, gwfp_tokens, gwifp_tokens, br_feat,
                          rfp_token):
        """sem(1) + gwfp(19) + gwifp(19) + br(1) + rfp(3) = 43 tokens.
        No mean-pooled global tokens — transformer learns window selection."""
        parts = [
            self.sem_proj(h_sem).unsqueeze(1),      # [B, 1, D]
            gwfp_tokens,                             # [B, 19, D]
            gwifp_tokens,                            # [B, 19, D]
            self.br_proj(br_feat).unsqueeze(1),     # [B, 1, D]
            rfp_token,                               # [B, 3, D]
        ]
        tokens = torch.cat(parts, dim=1)
        tokens = self.wl_dropout(tokens + self.wl_token_bias)
        return self.wl_norm(tokens)

    def _build_hw_tokens(self, hw):
        """8 physical-slice HW tokens."""
        embs = []
        for proj, inds in zip(self.hw_slice_proj, self.hw_slice_indices):
            embs.append(proj(hw[:, inds]))
        tokens = torch.stack(embs, dim=1)  # [B, 8, D]
        tokens = self.hw_dropout(tokens + self.hw_token_bias)
        return self.hw_norm(tokens)

    def forward_absolute(self, fp, ifp, hw, rfp, bbe, br, gwfp, gwifp):
        """Predict absolute metrics for one hardware config."""
        del fp, ifp  # use only global-window footprint
        B = hw.size(0)
        D = self.sem_proj.out_features

        if self.training and self.noise_std > 0:
            hw = hw + torch.randn_like(hw) * self.noise_std

        # Encode
        h_sem = self.sem_enc(bbe)
        gwfp_tokens = self.gwfp_enc(gwfp)       # [B, 19, D]
        gwifp_tokens = self.gwifp_enc(gwifp)    # [B, 19, D]
        br_feat = self.br_enc(br)
        _, rfp_token = self.rfp_enc(rfp)           # [B, 64], [B, 3, D]

        # Token sequences — no mean-pooled global tokens
        wl_tokens = self._build_wl_tokens(
            h_sem, gwfp_tokens, gwifp_tokens, br_feat, rfp_token)
        hw_tokens = self._build_hw_tokens(hw)  # [B, 8, D]

        # Concat: workload tokens + hw tokens → [B, 43+8=51, D]
        all_tokens = torch.cat([wl_tokens, hw_tokens], dim=1)

        # --- Loop Transformer (3 iterations, shared weights) ---
        state = all_tokens
        for _ in range(self.loop_steps):
            state = self.loop_block(state)

        # --- Task query readout ---
        # HW-conditioned queries: q_i = learnable_i + proj(hw)[i]
        task_q = self.task_queries.expand(B, -1, -1)  # [B, 10, D]
        hw_cond = self.query_hw_proj(hw).view(B, self.num_tasks, D)
        task_q = task_q + hw_cond  # HW-conditioned

        # Cross-attend: queries read from loop state
        # Use learned queries as Q, loop state as K/V
        readout = F.scaled_dot_product_attention(
            task_q, state, state, dropout_p=0.0,
        )

        # Per-task processing
        absolute = {}
        for i, name in enumerate(METRIC_KEYS):
            ctx = self.task_ctx[name](readout[:, i, :])
            absolute[name] = self.heads[name](ctx)
        return {"absolute": absolute}

    def forward(self, fp, ifp, hw_a, hw_b, rfp, bbe, br, gwfp, gwifp,
                hw_delta=None, changed_mask=None):
        """Forward for pair-based evaluation: both hw configs, derive delta."""
        abs_a = self.forward_absolute(fp, ifp, hw_a, rfp, bbe, br, gwfp, gwifp)
        abs_b = self.forward_absolute(fp, ifp, hw_b, rfp, bbe, br, gwfp, gwifp)

        delta = {k: abs_b["absolute"][k] - abs_a["absolute"][k]
                 for k in METRIC_KEYS}
        sensitivity = {k: delta[k].abs() for k in METRIC_KEYS}

        return {
            "absolute_a": abs_a["absolute"],
            "absolute_b": abs_b["absolute"],
            "delta": delta,
            "sensitivity": sensitivity,
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B = 4
    fp = torch.randn(B, 8, 19)
    ifp = torch.randn(B, 8, 19)
    gwfp = torch.randn(B, 8, 19)
    gwifp = torch.randn(B, 8, 19)
    hw_a = torch.randn(B, 47)
    hw_b = torch.randn(B, 47)
    rfp = torch.randn(B, 24)
    bbe = torch.randn(B, 512)
    br = torch.randn(B, 2)

    m = UnifiedLoopModel(embed_dim=256, dropout=0.1, loop_steps=3)
    print(f"UnifiedLoopModel: {count_params(m):,} params")

    out = m(fp, ifp, hw_a, hw_b, rfp, bbe, br, gwfp, gwifp)
    assert out["absolute_a"]["cpi"].shape == (B, 1), "abs_a shape"
    assert out["delta"]["cpi"].shape == (B, 1), "delta shape"
    assert out["sensitivity"]["cpi"].shape == (B, 1), "sens shape"

    abs_out = m.forward_absolute(fp, ifp, hw_a, rfp, bbe, br, gwfp, gwifp)
    assert abs_out["absolute"]["cpi"].shape == (B, 1), "abs shape"
    assert len(abs_out["absolute"]) == 10, "10 metrics"

    print("All tests passed.")
