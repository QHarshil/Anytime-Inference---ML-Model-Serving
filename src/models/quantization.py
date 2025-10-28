"""Quantisation utilities."""
from __future__ import annotations

from typing import Iterable, Set

import torch


def dynamic_quantise(model: torch.nn.Module, modules: Iterable[type]) -> torch.nn.Module:
    """Apply dynamic quantisation to ``model``.

    Dynamic quantisation works on CPU-only modules and does not require a
    calibration dataset. The helper is intentionally lightweight and raises a
    clear error when ``torch.quantization`` is unavailable (e.g., older CPU
    builds).
    """

    if not hasattr(torch, "quantization"):
        raise RuntimeError("torch.quantization is not available in this build")

    return torch.quantization.quantize_dynamic(model, set(modules), dtype=torch.qint8)


__all__ = ["dynamic_quantise"]
