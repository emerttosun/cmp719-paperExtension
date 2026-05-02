from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.fasternet_ext.metrics import count_parameters, measure_latency_ms, try_count_flops
from src.fasternet_ext.models.fasternet import build_fasternet


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FasterNet-T0/T1 on CIFAR-100.")
    parser.add_argument("--model", choices=["fasternet_t0", "fasternet_t1"], default="fasternet_t0")
    parser.add_argument("--dataset", choices=["cifar100"], default="cifar100")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cgm-placement", choices=["none", "all", "early", "late"], default="none")
    parser.add_argument("--cgm-reduction", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--save-gate-stats", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    resize = [] if args.image_size == 32 else [transforms.Resize((args.image_size, args.image_size))]
    train_transform = transforms.Compose(
        [
            *resize,
            transforms.RandomCrop(args.image_size, padding=4 if args.image_size == 32 else 0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train_set = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, 100


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    limit_batches: Optional[int] = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for batch_idx, (images, targets) in enumerate(progress):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
        if is_train:
            loss.backward()
            optimizer.step()
        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size
        progress.set_postfix(loss=total_loss / max(total_seen, 1), acc=100.0 * total_correct / max(total_seen, 1))
    return total_loss / max(total_seen, 1), 100.0 * total_correct / max(total_seen, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes = build_dataloaders(args)
    model = build_fasternet(
        model_name=args.model,
        num_classes=num_classes,
        image_size=args.image_size,
        cgm_placement=args.cgm_placement,
        cgm_reduction=args.cgm_reduction,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    params = count_parameters(model)
    flops = try_count_flops(model, args.image_size, device)
    print(f"Model: {args.model} | CGM: {args.cgm_placement} | params: {params:,}")
    if flops is not None:
        print(f"FLOPs: {flops / 1e6:.2f}M")

    metrics_path = output_dir / f"{args.model}_{args.cgm_placement}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc1", "val_loss", "val_acc1"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, device, optimizer, args.limit_train_batches
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, device, optimizer=None, limit_batches=args.limit_val_batches
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc1": train_acc,
                "val_loss": val_loss,
                "val_acc1": val_acc,
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, indent=2))

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "cgm_placement": args.cgm_placement,
        "cgm_reduction": args.cgm_reduction,
        "params": params,
        "flops": flops,
    }
    if args.measure_latency:
        summary["latency_ms_b1"] = measure_latency_ms(model, args.image_size, device, batch_size=1)
        print(f"Latency: {summary['latency_ms_b1']:.3f} ms/image")
    if args.save_gate_stats and hasattr(model, "collect_gate_means"):
        summary["gate_means"] = model.collect_gate_means()

    summary_path = output_dir / f"{args.model}_{args.cgm_placement}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
