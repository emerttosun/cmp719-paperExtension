from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor, nn


@torch.no_grad()
def accuracy_top1(logits: Tensor, targets: Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item() * 100.0)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def try_count_flops(model: nn.Module, image_size: int, device: torch.device) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model.eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    return float(FlopCountAnalysis(model, sample).total())


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 20,
    steps: int = 100,
) -> float:
    model.eval()
    sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / steps
