"""Synthetic CPU shape tests for the graph encoder + set predictor + JEPA roundtrip.

Run from the le-wm/ project root:
    python -m tests.test_graph_encoder_shapes
or under pytest:
    pytest tests/test_graph_encoder_shapes.py -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import GraphEncoder, SetPredictor  # noqa: E402
from utils.capacity_report import count_params  # noqa: E402


def _synth_batch(B=2, T=3, N=8, H=64, present=5):
    pixels = torch.randn(B, T, 3, H, H) * 0.5
    positions = torch.rand(B, T, N, 2) * 2 - 1  # in [-1, 1]
    mask = torch.zeros(B, T, N, dtype=torch.bool)
    mask[..., :present] = True
    positions[~mask] = 0.0
    return pixels, positions, mask


# --------------------------------------------------------------------------- #
# Encoder shape coverage
# --------------------------------------------------------------------------- #
def test_encoder_shapes():
    pixels, positions, mask = _synth_batch()
    px_flat = pixels.view(-1, 3, 64, 64)
    pos_flat = positions.view(-1, 8, 2)
    mask_flat = mask.view(-1, 8)

    cases = [
        # (d_obj, width_mode, expected_dim)
        (192, "native", 192),
        (192, "projected", 192),
        (64, "native", 64),
        (64, "projected", 192),
    ]
    for d_obj, mode, exp in cases:
        enc = GraphEncoder(image_size=64, d_obj=d_obj, capacity="small", width_mode=mode)
        out = enc(px_flat, positions=pos_flat, mask=mask_flat)
        ts = out.last_hidden_state
        assert ts.shape == (6, 8, exp), f"{d_obj}/{mode}: got {tuple(ts.shape)} want (6,8,{exp})"
        assert torch.isfinite(ts).all(), f"{d_obj}/{mode}: produced NaN/Inf"
        assert out.config.hidden_size == exp
        assert enc.embed_dim == exp
        assert getattr(enc, "is_graph_encoder", False) is True
    print("[ok] encoder shape grid")


def test_encoder_absent_token_used():
    """Absent slots should equal the (projected) ABSENT token, not the CNN output."""
    pixels, positions, mask = _synth_batch(present=5)
    px_flat = pixels.view(-1, 3, 64, 64)
    pos_flat = positions.view(-1, 8, 2)
    mask_flat = mask.view(-1, 8)
    enc = GraphEncoder(image_size=64, d_obj=192, capacity="small", width_mode="native")
    enc.eval()
    with torch.no_grad():
        out = enc(px_flat, positions=pos_flat, mask=mask_flat)
        # All absent rows across the batch should be IDENTICAL (single learned token).
        absent_rows = out.last_hidden_state[:, 5:, :].reshape(-1, 192)
        ref = absent_rows[0]
        assert torch.allclose(absent_rows, ref.unsqueeze(0).expand_as(absent_rows), atol=1e-6), \
            "absent slots are not all identical -> ABSENT token replacement broken"
        # And present slots should NOT match the absent token.
        present_rows = out.last_hidden_state[:, :5, :].reshape(-1, 192)
        diffs = (present_rows - ref).abs().sum(dim=-1)
        assert (diffs > 1e-3).all(), "present slots accidentally equal ABSENT token"
    print("[ok] absent-token replacement")


# --------------------------------------------------------------------------- #
# Predictor shape
# --------------------------------------------------------------------------- #
def test_predictor_shape():
    B, T, N, D = 2, 3, 8, 192
    tokens = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 4)
    mask = torch.zeros(B, T, N, dtype=torch.bool)
    mask[..., :5] = True

    pred = SetPredictor(
        num_frames=T,
        depth=2,
        heads=4,
        mlp_dim=512,
        input_dim=D,
        hidden_dim=D,
        n_max=N,
        dim_head=48,
        dropout=0.0,
        emb_dropout=0.0,
    )
    # The predictor expects act dim == hidden_dim for the AdaLN cond_proj
    # (mirrors module.Transformer convention). Project actions to match.
    act_proj = nn.Linear(4, D)
    out = pred(tokens, act_proj(actions), mask)
    assert out.shape == (B, T, N, D), f"got {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    print("[ok] predictor shape")


# --------------------------------------------------------------------------- #
# JEPA encode -> predict roundtrip on a mocked batch
# --------------------------------------------------------------------------- #
def test_jepa_roundtrip():
    from jepa import JEPA

    B, T, N, D = 2, 3, 8, 192
    enc = GraphEncoder(image_size=64, d_obj=D, capacity="small", width_mode="native")
    pred = SetPredictor(
        num_frames=T,
        depth=2,
        heads=4,
        mlp_dim=512,
        input_dim=D,
        hidden_dim=D,
        n_max=N,
        dim_head=48,
    )
    act_enc = nn.Sequential(nn.Linear(4, D))
    model = JEPA(encoder=enc, predictor=pred, action_encoder=act_enc)

    pixels, positions, mask = _synth_batch(B=B, T=T, N=N, H=64, present=5)
    info = {
        "pixels": pixels,
        "positions": positions,
        "mask": mask,
        "action": torch.randn(B, T, 4),
    }
    info = model.encode(info)
    assert info["emb"].shape == (B, T, N, D), f"emb {tuple(info['emb'].shape)}"
    assert info["emb_mask"].shape == (B, T, N)
    assert info["act_emb"].shape == (B, T, D)
    preds = model.predict(info["emb"], info["act_emb"], mask=info["emb_mask"])
    assert preds.shape == (B, T, N, D), f"preds {tuple(preds.shape)}"

    # Loss-style reduction: only over present slots (per the spec).
    target = info["emb"][:, 1:].detach()  # any next-step style target
    pred_slice = preds[:, :-1]
    m = info["emb_mask"][:, 1:].unsqueeze(-1).float()
    sq = ((pred_slice - target) ** 2) * m
    loss = sq.sum() / (m.sum() * D + 1e-6)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    print(f"[ok] jepa roundtrip; mocked loss={loss.item():.4f}")


# --------------------------------------------------------------------------- #
# Capacity targets
# --------------------------------------------------------------------------- #
def test_capacity_targets():
    targets = {"small": 1.2e6, "matched": 5.5e6}
    for cap, tgt in targets.items():
        enc = GraphEncoder(image_size=64, capacity=cap, d_obj=192, width_mode="native")
        n = count_params(enc)
        rel = abs(n - tgt) / tgt
        print(f"[cap] {cap}: {n / 1e6:.2f}M params (target {tgt / 1e6:.2f}M, off {rel * 100:.1f}%)")
        assert rel <= 0.15, f"capacity {cap}: {n/1e6:.2f}M outside +/-15% of {tgt/1e6:.2f}M"
    print("[ok] capacity within +/-15%")


def main() -> None:
    test_encoder_shapes()
    test_encoder_absent_token_used()
    test_predictor_shape()
    test_jepa_roundtrip()
    test_capacity_targets()
    print("\nALL OK")


if __name__ == "__main__":
    main()
