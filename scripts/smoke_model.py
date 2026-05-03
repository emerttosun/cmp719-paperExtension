from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fasternet_ext.metrics import count_parameters
from src.fasternet_ext.models.fasternet import build_fasternet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FasterNet output shapes for all CGM placements.")
    parser.add_argument("--model", choices=["fasternet_t0", "fasternet_t1"], default="fasternet_t0")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--cgm-mode", choices=["sigmoid", "residual"], default="sigmoid")
    parser.add_argument("--cgm-type", choices=["se", "eca"], default="se")
    parser.add_argument("--cgm-pooling", choices=["gap", "gap_gmp"], default="gap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = torch.randn(2, 3, args.image_size, args.image_size)
    for placement in ["none", "all", "early", "late", "s1", "s2", "s3", "s4", "s2s3", "s2s4"]:
        model = build_fasternet(
            args.model,
            args.num_classes,
            args.image_size,
            cgm_placement=placement,
            cgm_mode=args.cgm_mode,
            cgm_type=args.cgm_type,
            cgm_pooling=args.cgm_pooling,
        )
        y = model(x)
        assert tuple(y.shape) == (2, args.num_classes), (placement, y.shape)
        print(f"{placement:>5} | output={tuple(y.shape)} | params={count_parameters(model):,}")


if __name__ == "__main__":
    main()
