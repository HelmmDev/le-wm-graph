"""Set-of-tokens AR predictor for the LeWM graph variant.

Forked from `module.ARPredictor`. Same depth/heads/dim_head/dropout, plus:
- Inputs are (B, T, N_max, d_obj) instead of (B, T, d_obj). Internally
  flattened to (B, T*N_max, d_obj) for transformer attention.
- Combined attention mask: causal over T AND key-padding over N_max. Token
  (t, i) attends to (t', j) iff t' <= t AND mask[b, t', j].
- Additive positional embedding: temporal (T, d_obj) + object (N_max, d_obj),
  both learned.
- AdaLN action conditioning broadcast across the N tokens within a timestep.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from module import ConditionalBlock, modulate


class _MaskedAttention(nn.Module):
    """Multi-head attention with arbitrary additive attn-mask support."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        project_out = not (heads == 1 and dim_head == dim)
        self.to_out = (
            nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D); attn_mask: (B, L, L) bool, True = allowed."""
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.heads) for t in qkv)
        # SDPA expects bool mask where True = keep; broadcast over heads.
        am = attn_mask.unsqueeze(1)  # (B, 1, L, L)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=drop)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.to_out(out)


class _MaskedConditionalBlock(nn.Module):
    """ConditionalBlock variant that takes an explicit attention mask."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        # Reuse the FF + adaLN scaffolding from a fresh ConditionalBlock to
        # keep parameter shapes & init identical to the original predictor.
        ref = ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout)
        self.mlp = ref.mlp
        self.norm1 = ref.norm1
        self.norm2 = ref.norm2
        self.adaLN_modulation = ref.adaLN_modulation
        # Replace the attention with our masked variant (identical shape).
        self.attn = _MaskedAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SetPredictor(nn.Module):
    """Set-of-tokens autoregressive predictor (B, T, N_max, d_obj) -> same shape."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        n_max: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.n_max = n_max
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim

        # Project input/cond/output if dims differ (mirrors `module.Transformer`).
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = (
            nn.Linear(hidden_dim, self.output_dim) if hidden_dim != self.output_dim else nn.Identity()
        )

        # Additive positional embeddings.
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, 1, hidden_dim) * 0.02)
        self.object_pos = nn.Parameter(torch.randn(1, 1, n_max, hidden_dim) * 0.02)

        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList(
            [
                _MaskedConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _build_attn_mask(mask: torch.Tensor) -> torch.Tensor:
        """Build (B, T*N, T*N) bool attn mask: causal over T AND key-padding over N.

        Args:
            mask: (B, T, N) bool, True = present.
        """
        B, T, N = mask.shape
        device = mask.device
        # Causal over T: (T, T) lower-triangular True.
        causal_t = torch.ones(T, T, dtype=torch.bool, device=device).tril()  # (T, T)
        # Expand to per-token (T*N, T*N) by tiling each (t, t') block.
        causal = causal_t.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)  # (T*N, T*N)
        causal = causal.unsqueeze(0).expand(B, T * N, T * N)  # (B, T*N, T*N)

        # Key-padding: column (t', j) only valid if mask[b, t', j].
        key_valid = mask.reshape(B, T * N).unsqueeze(1)  # (B, 1, T*N)
        # Combine: query token can attend to key token iff causal AND key present.
        attn = causal & key_valid
        # Safety: at least let each token attend to itself so softmax has nonzero
        # support even if a query happens to be masked (we still output for it).
        eye = torch.eye(T * N, dtype=torch.bool, device=device).unsqueeze(0)
        attn = attn | eye
        return attn

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, T, N_max, d_obj)
            c    : (B, T, act_dim) - one action per timestep, broadcast across N.
            mask : (B, T, N_max) bool - True = object present.
        Returns:
            (B, T, N_max, output_dim)
        """
        B, T, N, _ = x.shape
        if T > self.num_frames:
            raise ValueError(f"T={T} exceeds num_frames={self.num_frames}")

        # Add positional embeddings (broadcast).
        x = x + self.temporal_pos[:, :T] + self.object_pos[:, :, :N]
        x = self.dropout(x)

        # Project input and cond if needed.
        x = self.input_proj(x)
        c_proj = self.cond_proj(c)  # (B, T, hidden)

        # Broadcast cond across the N tokens within each timestep, then flatten.
        c_tok = c_proj.unsqueeze(2).expand(B, T, N, c_proj.shape[-1])  # (B,T,N,H)
        c_flat = rearrange(c_tok, "b t n d -> b (t n) d")
        x_flat = rearrange(x, "b t n d -> b (t n) d")

        # Build the combined attention mask once for this batch.
        attn_mask = self._build_attn_mask(mask)  # (B, T*N, T*N) bool

        for blk in self.layers:
            x_flat = blk(x_flat, c_flat, attn_mask)

        x_flat = self.norm(x_flat)
        x_flat = self.output_proj(x_flat)

        out = rearrange(x_flat, "b (t n) d -> b t n d", t=T, n=N)
        return out
