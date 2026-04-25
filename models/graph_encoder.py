"""Graph encoder: per-object CNN tower for the LeWM JEPA graph variant.

Locked design (see Wave 2 spec):
- Inference uses simulator GT positions (same as training).
- Variable N: pad to N_max=8 with a learned [ABSENT] token; predictor consumes
  an attention mask.
- Two width modes (toggleable):
    * native    : per-object width = D_obj
    * projected : final Linear(D_obj -> 192) for ablation
- Two capacity points (toggleable):
    * small   : ~1.2M params,  channels 48 -> 96 -> 192
    * matched : ~5.5M params,  channels 96 -> 192 -> 384 -> 768
- Patch size auto: 16 for image_size=64, 64 for image_size=224 (overrideable).

Forward path:
  pixels    : (B*T, 3, H, W)
  positions : (B*T, N_max, 2) in [-1, 1]   (zeros for absent slots)
  mask      : (B*T, N_max) bool             (True = present)

Per-object patches are cropped from `pixels` centred on each (denormalised)
position with zero-padding for boundary cases. Crops are stacked
(B*T*N_max, 3, P, P), passed through the CNN tower, reshaped back to
(B*T, N_max, D_obj), summed with a positional MLP embedding, LayerNorm-d,
then absent slots get the learned [ABSENT] token. Optional projection to 192
is the last step.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Conv -> GroupNorm(8) -> GELU."""
    # Make groups divide channels; GN(8) is the spec but fall back if not divisible.
    groups = 8 if out_ch % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, out_ch),
        nn.GELU(),
    )


def _build_cnn_tower(capacity: str, d_obj: int) -> nn.Sequential:
    """Build the per-object CNN tower.

    The tower ends with AdaptiveAvgPool2d(1) -> Linear -> LayerNorm so the
    per-token width is always `d_obj` regardless of the channel ladder.
    """
    if capacity == "small":
        # Three stages, channels 48 -> 96 -> 192. Extra refinement blocks at
        # the deeper stages so the total lands near the 1.2M-param target.
        chans = [48, 96, 192]
        blocks: list[nn.Module] = [
            _conv_block(3, chans[0], stride=1),
            _conv_block(chans[0], chans[0], stride=1),
            _conv_block(chans[0], chans[1], stride=2),
            _conv_block(chans[1], chans[1], stride=1),
            _conv_block(chans[1], chans[1], stride=1),
            _conv_block(chans[1], chans[2], stride=2),
            _conv_block(chans[2], chans[2], stride=1),
            _conv_block(chans[2], chans[2], stride=1),
        ]
        last_ch = chans[-1]
    elif capacity == "matched":
        # Four stages, 96 -> 192 -> 384 -> 768. Each stage transition uses
        # stride-2 to reduce spatial dims; intermediate refinement blocks add
        # representational capacity to hit the ~5.5M target.
        chans = [96, 192, 384, 768]
        blocks = [
            _conv_block(3, chans[0], stride=1),
            _conv_block(chans[0], chans[0], stride=1),
            _conv_block(chans[0], chans[1], stride=2),
            _conv_block(chans[1], chans[1], stride=1),
            _conv_block(chans[1], chans[2], stride=2),
            _conv_block(chans[2], chans[2], stride=1),
            _conv_block(chans[2], chans[3], stride=2),
        ]
        last_ch = chans[-1]
    else:
        raise ValueError(f"unknown capacity {capacity!r}; expected 'small' or 'matched'")

    blocks.append(nn.AdaptiveAvgPool2d(1))
    blocks.append(nn.Flatten(1))
    blocks.append(nn.Linear(last_ch, d_obj))
    blocks.append(nn.LayerNorm(d_obj))
    return nn.Sequential(*blocks)


class GraphEncoder(nn.Module):
    """Per-object CNN encoder producing N_max tokens per frame.

    Attributes:
        is_graph_encoder : True (set so jepa.encode/predict can dispatch).
        config.hidden_size : per-token width on the OUTPUT side (after optional
            projection). The downstream pipeline reshapes to (B, T*N, hidden_size)
            so this is per-token, not aggregated across objects.
        embed_dim : same as `config.hidden_size`.
    """

    is_graph_encoder = True

    def __init__(
        self,
        image_size: int,
        n_max: int = 8,
        d_obj: int = 192,
        capacity: str = "matched",
        width_mode: str = "native",
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        if width_mode not in ("native", "projected"):
            raise ValueError(f"width_mode must be 'native' or 'projected', got {width_mode!r}")

        if patch_size is None:
            # Auto: 16 for 64x64 images, 64 for 224x224 images.
            if image_size == 64:
                patch_size = 16
            elif image_size == 224:
                patch_size = 64
            else:
                # Fall back to a quarter of the image, rounded to even.
                patch_size = max(8, (image_size // 4) // 2 * 2)

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.n_max = int(n_max)
        self.d_obj = int(d_obj)
        self.capacity = capacity
        self.width_mode = width_mode

        # Per-object CNN tower (shared across objects within a frame).
        self.cnn = _build_cnn_tower(capacity, d_obj)

        # Positional MLP: (x, y) -> d_obj.
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, d_obj),
        )

        # Layer norm applied after summing patch features and positional encoding.
        self.token_norm = nn.LayerNorm(d_obj)

        # Learned [ABSENT] token (added to absent slots after the LayerNorm so
        # it doesn't get re-normed; downstream LayerNorm in the predictor will
        # take care of any drift).
        self.absent_token = nn.Parameter(torch.randn(d_obj) * 0.02)

        # Optional projection back to 192 for the ablation width mode.
        if width_mode == "projected":
            self.out_proj: nn.Module = nn.Linear(d_obj, 192)
            out_dim = 192
        else:
            self.out_proj = nn.Identity()
            out_dim = d_obj

        self.embed_dim = out_dim
        self.config = SimpleNamespace(hidden_size=out_dim)

        # Capacity verification (uses utils.capacity_report.count_params, no FLOPs
        # so we don't pay an fvcore trace at construction time on every run).
        self._verify_capacity()

    # --------------------------------------------------------------- internals
    def _verify_capacity(self) -> None:
        """Warn if param count drifts >10% from the target for this capacity."""
        try:
            from utils.capacity_report import count_params
        except Exception:  # pragma: no cover - utils path unavailable
            return
        targets = {"small": 1.2e6, "matched": 5.5e6}
        if self.capacity not in targets:
            return
        target = targets[self.capacity]
        actual = count_params(self, trainable_only=True)
        rel = abs(actual - target) / target
        if rel > 0.10:
            print(
                f"[GraphEncoder] WARNING: capacity={self.capacity!r} param count "
                f"{actual / 1e6:.2f}M is {rel * 100:.1f}% off target "
                f"{target / 1e6:.2f}M (>10%)."
            )

    def _crop_patches(
        self, pixels: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Crop one PxP patch per object centred on each (denormalised) position.

        Args:
            pixels    : (M, 3, H, W) where M = B*T
            positions : (M, N_max, 2) in [-1, 1]

        Returns:
            patches : (M, N_max, 3, P, P)
        """
        M, _, H, W = pixels.shape
        N = self.n_max
        P = self.patch_size
        half = P // 2

        # Pad the source image by `half` so any centre is valid.
        pad = half
        padded = F.pad(pixels, (pad, pad, pad, pad))  # (M, 3, H+2p, W+2p)

        # Denormalise: positions are in [-1, 1] over the original image.
        # Map to pixel coordinates in [0, W) and [0, H), then offset by pad
        # because the padded image starts at -pad in original coords.
        x_norm = positions[..., 0]  # (M, N)
        y_norm = positions[..., 1]
        x_pix = ((x_norm + 1.0) * 0.5 * W).long().clamp(0, W - 1) + pad  # (M, N)
        y_pix = ((y_norm + 1.0) * 0.5 * H).long().clamp(0, H - 1) + pad

        # Top-left of each crop in padded coords.
        x0 = x_pix - half  # (M, N)
        y0 = y_pix - half

        # Build per-object index grids (M, N, P, P) -> gather from padded image.
        device = pixels.device
        py = torch.arange(P, device=device).view(1, 1, P, 1)
        px = torch.arange(P, device=device).view(1, 1, 1, P)
        ys = y0.view(M, N, 1, 1) + py  # (M, N, P, P)
        xs = x0.view(M, N, 1, 1) + px

        # Gather: result (M, N, 3, P, P).
        # Use advanced indexing on the padded tensor.
        m_idx = torch.arange(M, device=device).view(M, 1, 1, 1).expand(M, N, P, P)
        # padded[m, c, y, x] -> result indexed over c.
        # Shape after indexing padded[m_idx, :, ys, xs]: (M, N, P, P, 3); we need
        # to permute the channel dim to position 2.
        gathered = padded[m_idx, :, ys, xs]  # (M, N, P, P, 3)
        patches = gathered.permute(0, 1, 4, 2, 3).contiguous()  # (M, N, 3, P, P)
        return patches

    # -------------------------------------------------------------------- API
    def forward(
        self,
        pixels: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        interpolate_pos_encoding: bool = True,  # noqa: ARG002 - HF-compat shim
    ) -> SimpleNamespace:
        """Encode a flat batch of frames into N_max object tokens each.

        Args:
            pixels    : (B*T, 3, H, W)
            positions : (B*T, N_max, 2) in [-1, 1]
            mask      : (B*T, N_max) bool, True = object present
        Returns:
            namespace with `.last_hidden_state` (B*T, N_max, hidden_size),
            `.config.hidden_size`, and `.mask` (the input mask, passed through
            for the predictor).
        """
        if pixels.dim() != 4:
            raise ValueError(f"pixels must be (M, 3, H, W); got {tuple(pixels.shape)}")
        M = pixels.shape[0]
        N = self.n_max
        if positions.shape[:2] != (M, N) or positions.shape[-1] != 2:
            raise ValueError(
                f"positions must be (M, {N}, 2); got {tuple(positions.shape)}"
            )
        if mask.shape != (M, N):
            raise ValueError(f"mask must be (M, {N}); got {tuple(mask.shape)}")

        # Crop per-object patches with zero-padding boundary handling.
        patches = self._crop_patches(pixels, positions)  # (M, N, 3, P, P)
        flat = patches.view(M * N, 3, self.patch_size, self.patch_size)

        # CNN tower.
        feat = self.cnn(flat)  # (M*N, d_obj)
        feat = feat.view(M, N, self.d_obj)

        # Positional encoding (per object).
        pos_emb = self.pos_mlp(positions)  # (M, N, d_obj)
        tokens = self.token_norm(feat + pos_emb)

        # Replace absent slots with the learned [ABSENT] token. We expand to
        # the right shape and use `where` to swap in.
        absent = self.absent_token.view(1, 1, self.d_obj).expand(M, N, self.d_obj)
        present = mask.unsqueeze(-1)  # (M, N, 1)
        tokens = torch.where(present, tokens, absent)

        # Optional projection.
        tokens = self.out_proj(tokens)  # (M, N, hidden_size)

        return SimpleNamespace(
            last_hidden_state=tokens,
            config=self.config,
            mask=mask,
        )
