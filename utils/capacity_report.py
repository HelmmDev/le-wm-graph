"""Encoder capacity reporting: parameter count + per-step FLOPs.

Used to enforce capacity-matched comparisons between flat ViT and graph-encoder
variants in LeWM. Always report params + FLOPs alongside any results table.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from fvcore.nn import FlopCountAnalysis


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def compute_flops(module: nn.Module, input_shape: tuple) -> int:
    """Compute FLOPs for one forward pass on a CPU input.

    `input_shape` does NOT include batch dim; batch=1 is added here.
    Module is moved to CPU and put in eval mode for the measurement; we
    restore the original training flag after.
    """
    x = torch.zeros((1,) + tuple(input_shape), dtype=torch.float32)
    was_training = module.training
    module.eval()
    # Silence fvcore's "unsupported op" chatter — it's noisy and not actionable
    # for HF transformers (einops, scaled_dot_product_attention, etc.).
    fvcore_log = logging.getLogger("fvcore.nn.jit_analysis")
    prev_level = fvcore_log.level
    fvcore_log.setLevel(logging.ERROR)
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(module.cpu(), x).total()
    finally:
        fvcore_log.setLevel(prev_level)
        if was_training:
            module.train()
    return int(flops)


def report_capacity(name: str, module: nn.Module, input_shape: tuple) -> dict:
    """Build a capacity report dict for one module."""
    n = count_params(module, trainable_only=True)
    f = compute_flops(module, input_shape)
    return {
        "name": name,
        "params": n,
        "flops_per_step": f,
        "params_M": n / 1e6,
        "gflops_per_step": f / 1e9,
    }


def format_capacity_table(reports: list[dict]) -> str:
    """Pretty markdown table for paste into results docs."""
    header = "| name | params (M) | GFLOPs/step |\n|---|---:|---:|\n"
    rows = [
        f"| {r['name']} | {r['params_M']:.2f} | {r['gflops_per_step']:.2f} |"
        for r in reports
    ]
    return header + "\n".join(rows)
