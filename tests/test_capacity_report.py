"""Smoke test for utils.capacity_report on the LeWM ViT-tiny encoder.

Run from the le-wm/ project root:
    python -m tests.test_capacity_report
or under pytest:
    pytest tests/test_capacity_report.py -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Allow running as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.capacity_report import (  # noqa: E402
    count_params,
    format_capacity_table,
    report_capacity,
)


class ViTEncoderAdapter(nn.Module):
    """Wrap the HF ViT so fvcore can call it with a single positional tensor.

    JEPA invokes `encoder(pixels, interpolate_pos_encoding=True)` and reads the
    cls token off `output.last_hidden_state[:, 0]`. We mirror that here so the
    measured FLOPs match the encode path used in training.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixels, interpolate_pos_encoding=True)
        return out.last_hidden_state[:, 0]


def build_vit_tiny_encoder() -> nn.Module:
    import stable_pretraining as spt

    return spt.backbone.utils.vit_hf(
        "tiny",
        patch_size=14,
        image_size=224,
        pretrained=False,
        use_mask_token=False,
    )


def test_vit_tiny_capacity():
    encoder = build_vit_tiny_encoder()
    adapter = ViTEncoderAdapter(encoder)

    # Sanity: the adapter should not introduce parameters.
    assert count_params(adapter) == count_params(encoder), (
        "ViTEncoderAdapter should be parameter-free wrt the underlying encoder"
    )

    report = report_capacity("ViT-tiny encoder", adapter, input_shape=(3, 224, 224))

    # Paper claims ViT-tiny ~5M params; allow a generous range.
    assert 3e6 <= report["params"] <= 8e6, (
        f"Unexpected ViT-tiny param count: {report['params']:,}"
    )
    # ViT-tiny @ 224 should be in the ~1-3 GFLOP ballpark.
    assert 0.3 <= report["gflops_per_step"] <= 5.0, (
        f"Unexpected ViT-tiny GFLOPs: {report['gflops_per_step']:.2f}"
    )

    table = format_capacity_table([report])
    print("\n" + table)
    return report


if __name__ == "__main__":
    test_vit_tiny_capacity()
    print("\nOK")
