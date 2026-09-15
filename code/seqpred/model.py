"""SeqPredictor model — TAO-style instruction-trace self-attention.

Backbone disjoint from the FiLM four-stream encoder.  Per-record token embed
(opcode + bhist + regbm + accdist) -> sub-window transformer -> attention pool
-> interval pooling -> HW conditioning -> 10 linear heads.

Default loss: SmoothL1 on normalized labels (plain absolute regression baseline).
An optional delta loss is available through LOSS=delta in train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_unified import METRIC_KEYS

N_METRICS = len(METRIC_KEYS)  # 10
REGBM_BITS = 256
ACCDIST_NM = 32
N_WINDOWS = 8
HW_DIM = 47


class AttnPool(nn.Module):
    """Masked attention pooling over a sequence -> single vector."""

    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, D]; mask: [B, L] (1 = valid)
        scores = self.proj(x) @ self.query  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(mask < 0.5, float("-inf"))
        # Rows fully masked -> uniform (avoid NaN).
        all_masked = (mask.sum(dim=1) < 0.5) if mask is not None else None
        attn = torch.softmax(scores, dim=1)  # [B, L]
        if all_masked is not None:
            attn = torch.where(all_masked.unsqueeze(1), torch.zeros_like(attn), attn)
        return (attn.unsqueeze(-1) * x).sum(dim=1)  # [B, D]


class SeqPredictor(nn.Module):
    def __init__(
        self,
        opcode_vocab_size: int,
        d: int = 128,
        nhead: int = 4,
        window_layers: int = 2,
        hw_dim: int = HW_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d

        # Per-record token embeddings.
        self.opcode_emb = nn.Embedding(opcode_vocab_size, d, padding_idx=0)
        self.bhist_emb = nn.Embedding(256, d)
        self.regbm_proj = nn.Linear(REGBM_BITS, d)
        self.accdist_proj = nn.Linear(ACCDIST_NM, d)
        self.token_norm = nn.LayerNorm(d)
        self.token_drop = nn.Dropout(dropout)

        # Sub-window encoder (shared across the 8 windows).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.window_encoder = nn.TransformerEncoder(enc_layer, num_layers=window_layers)
        self.window_pool = AttnPool(d)

        # Interval pooling over the 8 sub-window vectors (small self-attention).
        iv_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.interval_encoder = nn.TransformerEncoder(iv_layer, num_layers=1)
        self.interval_pool = AttnPool(d)

        # HW conditioning: 47 -> 128 -> d (simple MLP; NOT the FiLM GAM).
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d),
        )

        # Fuse interval_rep + HW -> trunk.
        self.fuse = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 2, d), nn.LayerNorm(d),
        )

        # 10 linear heads.
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(N_METRICS)])

    def encode_interval(self, feats: dict) -> torch.Tensor:
        """feats tensors are batched per (interval, hw) example:
          opcode_idx [B, W, L], bhist [B, W, L], regbm [B, W, L, 256],
          accdist [B, W, L, 32], mask [B, W, L].
        Returns interval representation [B, d].
        """
        op = feats["opcode_idx"]
        B, W, L = op.shape

        tok = (
            self.opcode_emb(op)
            + self.bhist_emb(feats["bhist"])
            + self.regbm_proj(feats["regbm"])
            + self.accdist_proj(feats["accdist"])
        )  # [B, W, L, d]
        tok = self.token_drop(self.token_norm(tok))

        # Process the W sub-windows as a batch of sequences.
        tok = tok.reshape(B * W, L, self.d)
        mask = feats["mask"].reshape(B * W, L)  # 1 = valid
        key_padding = mask < 0.5  # True = pad (ignored by attention)
        # Fully-padded rows would make attention NaN; encoder tolerates it but
        # we guard the pool. Avoid all-True rows tripping the encoder by giving
        # the encoder a non-all-True mask where a row is fully padded.
        fully_pad = key_padding.all(dim=1)
        safe_mask = key_padding.clone()
        safe_mask[fully_pad] = False
        enc = self.window_encoder(tok, src_key_padding_mask=safe_mask)  # [B*W, L, d]
        win_vec = self.window_pool(enc, mask)  # [B*W, d]
        win_vec = win_vec.reshape(B, W, self.d)

        # Interval pooling over the 8 sub-window vectors.
        win_valid = (feats["mask"].sum(dim=2) > 0.5).float()  # [B, W]
        iv = self.interval_encoder(
            win_vec, src_key_padding_mask=(win_valid < 0.5)
        )
        return self.interval_pool(iv, win_valid)  # [B, d]

    def forward(self, feats: dict, hw: torch.Tensor) -> torch.Tensor:
        """hw: [B, 47] normalized. Returns predictions [B, 10] (normalized space)."""
        iv_rep = self.encode_interval(feats)        # [B, d]
        hw_rep = self.hw_encoder(hw)                # [B, d]
        trunk = self.fuse(torch.cat([iv_rep, hw_rep], dim=-1))  # [B, d]
        preds = torch.cat([h(trunk) for h in self.heads], dim=-1)  # [B, 10]
        return preds
