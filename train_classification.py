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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from urllib.error import HTTPError, URLError
from tqdm import tqdm

from src.fasternet_ext.metrics import count_parameters, measure_latency_ms, try_count_flops
from src.fasternet_ext.models.fasternet import build_fasternet


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_MIRRORS = (
    "https://zenodo.org/records/10089977/files/cifar-100-python.tar.gz?download=1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FasterNet-T0/T1 on CIFAR-100 or Tiny-ImageNet.")
    parser.add_argument("--model", choices=["fasternet_t0", "fasternet_t1"], default="fasternet_t0")
    parser.add_argument(
        "--dataset",
        choices=["cifar100", "tiny_imagenet", "fake_cifar100"],
        default="cifar100",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["hf", "torchvision"],
        default="hf",
        help="Use Hugging Face datasets by default; torchvision uses the original CIFAR URL.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cgm-placement",
        choices=["none", "all", "early", "late", "s1", "s2", "s3", "s4", "s2s3", "s2s4"],
        default="none",
    )
    parser.add_argument("--cgm-reduction", type=int, default=4)
    parser.add_argument(
        "--cgm-mode",
        choices=["sigmoid", "residual", "centered"],
        default="sigmoid",
        help=(
            "sigmoid multiplies by gates in [0,1]; residual scales with "
            "1 + alpha * (sigmoid - 0.5); centered scales with "
            "1 + alpha * (2 * sigmoid - 1). With centered and alpha=0.5, "
            "the scale range is [0.5, 1.5]."
        ),
    )
    parser.add_argument(
        "--cgm-type",
        choices=["se", "eca"],
        default="se",
        help="Gate generator type: SE-style bottleneck MLP or ECA-style 1D channel convolution.",
    )
    parser.add_argument(
        "--cgm-pooling",
        choices=["gap", "gap_gmp"],
        default="gap",
        help="Pooling signal used by CGM: GAP only, or shared-weight GAP+GMP.",
    )
    parser.add_argument(
        "--eca-kernel-size",
        type=int,
        default=3,
        help="Odd 1D convolution kernel size used when --cgm-type eca.",
    )
    parser.add_argument(
        "--cgm-alpha",
        type=float,
        default=0.5,
        help=(
            "Identity-centered CGM strength. For --cgm-mode centered, alpha=0.5 "
            "gives scale in [0.5, 1.5]. For --cgm-mode residual, alpha=0.5 "
            "gives scale in [0.75, 1.25]."
        ),
    )
    parser.add_argument(
        "--cgm-init-bias",
        type=float,
        default=0.0,
        help=(
            "Identity initialization bias for sigmoid CGM's gate output layer. With b>0, "
            "the gate's last layer is reset to weight=0, bias=b after global "
            "init, so sigmoid(b) is the initial gate value (b=4 -> ~0.982, "
            "near identity). For gap_gmp, each pooling branch uses b/2 so the "
            "summed logits still start at b. Default 0.0 disables the override "
            "and preserves the original random init for backward compatibility."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--save-gate-stats", action="store_true")
    parser.add_argument(
        "--gate-hist-bins",
        type=int,
        default=10,
        help="Number of bins used for saved gate/scale histograms when --save-gate-stats is set.",
    )
    parser.add_argument(
        "--gate-stats-batches",
        type=int,
        default=1,
        help=(
            "Number of validation batches used to refresh saved gate stats after training. "
            "Use a larger value for more stable histograms; default 1 keeps the extra cost small."
        ),
    )
    parser.add_argument(
        "--disable-cifar-mirror",
        action="store_true",
        help="Do not try mirror downloads if the default torchvision CIFAR-100 URL fails.",
    )
    parser.add_argument(
        "--fallback-fake-data",
        action="store_true",
        help="Use synthetic CIFAR-shaped data if CIFAR-100 download is temporarily unavailable.",
    )
    args = parser.parse_args()
    if args.cgm_init_bias > 0.0 and args.cgm_mode != "sigmoid":
        parser.error("--cgm-init-bias > 0 is only supported with --cgm-mode sigmoid.")
    if args.gate_hist_bins < 1:
        parser.error("--gate-hist-bins must be at least 1.")
    if args.gate_stats_batches < 1:
        parser.error("--gate-stats-batches must be at least 1.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_fake_cifar100(args: argparse.Namespace) -> tuple[datasets.FakeData, datasets.FakeData]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    image_shape = (3, args.image_size, args.image_size)
    train_set = datasets.FakeData(
        size=1024,
        image_size=image_shape,
        num_classes=100,
        transform=transform,
        random_offset=args.seed,
    )
    val_set = datasets.FakeData(
        size=256,
        image_size=image_shape,
        num_classes=100,
        transform=transform,
        random_offset=args.seed + 10_000,
    )
    return train_set, val_set


class HFImageClassificationDataset(Dataset):
    def __init__(self, split, image_key: str, label_key: str, transform) -> None:
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        item = self.split[index]
        image = item[self.image_key].convert("RGB")
        label = int(item[self.label_key])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class HFCIFAR100Dataset(HFImageClassificationDataset):
    def __init__(self, split, transform) -> None:
        super().__init__(split, image_key="img", label_key="fine_label", transform=transform)


def build_hf_cifar100(args: argparse.Namespace, train_transform, val_transform):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset source requires the `datasets` package. "
            "Run `pip install -r requirements.txt` and try again."
        ) from exc

    try:
        dataset = load_dataset("uoft-cs/cifar100", cache_dir=args.data_dir)
        train_set = HFCIFAR100Dataset(dataset["train"], train_transform)
        val_set = HFCIFAR100Dataset(dataset["test"], val_transform)
        return train_set, val_set
    except Exception as exc:
        if args.fallback_fake_data:
            print(
                "Hugging Face CIFAR-100 load failed; falling back to synthetic FakeData "
                "for pipeline testing. Do not report FakeData accuracy as a real result."
            )
            print(f"Hugging Face error was: {exc}")
            return build_fake_cifar100(args)
        raise RuntimeError(
            "Hugging Face CIFAR-100 load failed. Retry the command later, or add "
            "--fallback-fake-data only for smoke-testing the training pipeline."
        ) from exc


def build_hf_tiny_imagenet(args: argparse.Namespace, train_transform, val_transform):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Tiny-ImageNet requires the `datasets` package. "
            "Run `pip install -r requirements.txt` and try again."
        ) from exc

    try:
        dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=args.data_dir)
        train_set = HFImageClassificationDataset(
            dataset["train"], image_key="image", label_key="label", transform=train_transform
        )
        val_set = HFImageClassificationDataset(
            dataset["valid"], image_key="image", label_key="label", transform=val_transform
        )
        return train_set, val_set
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face Tiny-ImageNet load failed. Retry later or verify that "
            "`zh-plus/tiny-imagenet` is accessible from the current runtime."
        ) from exc


def build_cifar100(args: argparse.Namespace, train_transform, val_transform):
    try:
        train_set = datasets.CIFAR100(
            args.data_dir, train=True, download=True, transform=train_transform
        )
        val_set = datasets.CIFAR100(
            args.data_dir, train=False, download=True, transform=val_transform
        )
        return train_set, val_set
    except (HTTPError, URLError, RuntimeError) as exc:
        if not args.disable_cifar_mirror:
            print(f"Default CIFAR-100 download failed: {exc}")
            print("Trying CIFAR-100 mirror download...")
            for mirror_url in CIFAR100_MIRRORS:
                try:
                    download_and_extract_archive(
                        url=mirror_url,
                        download_root=args.data_dir,
                        filename=CIFAR100_ARCHIVE,
                        md5=CIFAR100_MD5,
                    )
                    train_set = datasets.CIFAR100(
                        args.data_dir, train=True, download=False, transform=train_transform
                    )
                    val_set = datasets.CIFAR100(
                        args.data_dir, train=False, download=False, transform=val_transform
                    )
                    print(f"CIFAR-100 loaded from mirror: {mirror_url}")
                    return train_set, val_set
                except (HTTPError, URLError, RuntimeError) as mirror_exc:
                    print(f"Mirror failed: {mirror_url}")
                    print(f"Mirror error was: {mirror_exc}")
        if args.fallback_fake_data:
            print(
                "CIFAR-100 download failed; falling back to synthetic FakeData for pipeline testing. "
                "Do not report FakeData accuracy as a real result."
            )
            print(f"Download error was: {exc}")
            return build_fake_cifar100(args)
        raise RuntimeError(
            "CIFAR-100 download failed. This is usually a temporary network/server issue. "
            "Retry the command later, or add --fallback-fake-data only for smoke-testing the "
            "training pipeline without real CIFAR-100 results."
        ) from exc


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if args.dataset == "tiny_imagenet" else (CIFAR100_MEAN, CIFAR100_STD)
    resize = [] if args.image_size == 32 else [transforms.Resize((args.image_size, args.image_size))]
    train_transform = transforms.Compose(
        [
            *resize,
            transforms.RandomCrop(args.image_size, padding=4 if args.image_size == 32 else 0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if args.dataset == "fake_cifar100":
        train_set, val_set = build_fake_cifar100(args)
        num_classes = 100
    elif args.dataset == "tiny_imagenet":
        train_set, val_set = build_hf_tiny_imagenet(args, train_transform, val_transform)
        num_classes = 200
    elif args.dataset_source == "hf":
        train_set, val_set = build_hf_cifar100(args, train_transform, val_transform)
        num_classes = 100
    else:
        train_set, val_set = build_cifar100(args, train_transform, val_transform)
        num_classes = 100
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
    return train_loader, val_loader, num_classes


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


@torch.no_grad()
def tensor_stats(tensor: torch.Tensor, bins: int, hist_min: float, hist_max: float) -> dict[str, object]:
    flat = tensor.detach().float().reshape(-1).cpu()
    hist = torch.histc(flat, bins=bins, min=hist_min, max=hist_max)
    edges = torch.linspace(hist_min, hist_max, bins + 1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "hist_counts": [int(value) for value in hist.tolist()],
        "hist_edges": [float(value) for value in edges.tolist()],
    }


@torch.no_grad()
def channel_means(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().float().mean(dim=(0, 2, 3)).cpu().tolist()]


@torch.no_grad()
def collect_gate_summary_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bins: int,
    limit_batches: int,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    total_seen = 0
    gate_batches: dict[str, list[torch.Tensor]] = {}
    scale_batches: dict[str, list[torch.Tensor]] = {}
    scale_ranges: dict[str, tuple[float, float]] = {}
    for batch_idx, (images, _) in enumerate(tqdm(loader, leave=False, desc="gate-stats")):
        if batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        model(images)
        total_seen += images.shape[0]
        for name, module in model.named_modules():
            latest_gate = getattr(module, "latest_gate", None)
            latest_scale = getattr(module, "latest_scale", None)
            if latest_gate is None or latest_scale is None:
                continue
            gate_batches.setdefault(name, []).append(latest_gate.detach().cpu())
            scale_batches.setdefault(name, []).append(latest_scale.detach().cpu())
            mode = getattr(module, "mode", "sigmoid")
            alpha = float(getattr(module, "alpha", 0.0))
            hist_min = 0.0 if mode == "sigmoid" else max(0.0, 1.0 - abs(alpha))
            hist_max = 1.0 if mode == "sigmoid" else 1.0 + abs(alpha)
            scale_ranges[name] = (hist_min, hist_max)

    summary: dict[str, object] = {"gate_stats_num_images": total_seen}
    gate_tensors = {name: torch.cat(values, dim=0) for name, values in gate_batches.items()}
    scale_tensors = {name: torch.cat(values, dim=0) for name, values in scale_batches.items()}
    summary["gate_means"] = {name: float(tensor.mean()) for name, tensor in gate_tensors.items()}
    summary["scale_means"] = {name: float(tensor.mean()) for name, tensor in scale_tensors.items()}
    summary["gate_stats"] = {
        name: tensor_stats(tensor, bins=bins, hist_min=0.0, hist_max=1.0)
        for name, tensor in gate_tensors.items()
    }
    summary["scale_stats"] = {
        name: tensor_stats(tensor, bins=bins, hist_min=scale_ranges[name][0], hist_max=scale_ranges[name][1])
        for name, tensor in scale_tensors.items()
    }
    summary["gate_vectors"] = {name: channel_means(tensor) for name, tensor in gate_tensors.items()}
    summary["scale_vectors"] = {name: channel_means(tensor) for name, tensor in scale_tensors.items()}

    model.train(was_training)
    return summary


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
        cgm_mode=args.cgm_mode,
        cgm_alpha=args.cgm_alpha,
        cgm_type=args.cgm_type,
        eca_kernel_size=args.eca_kernel_size,
        cgm_pooling=args.cgm_pooling,
        cgm_init_bias=args.cgm_init_bias,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    params = count_parameters(model)
    flops = try_count_flops(model, args.image_size, device)
    print(
        f"Model: {args.model} | CGM: {args.cgm_placement} | "
        f"type: {args.cgm_type} | pooling: {args.cgm_pooling} | "
        f"mode: {args.cgm_mode} | params: {params:,}"
    )
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
        "image_size": args.image_size,
        "cgm_placement": args.cgm_placement,
        "cgm_reduction": args.cgm_reduction,
        "cgm_mode": args.cgm_mode,
        "cgm_alpha": args.cgm_alpha,
        "cgm_type": args.cgm_type,
        "cgm_pooling": args.cgm_pooling,
        "cgm_init_bias": args.cgm_init_bias,
        "eca_kernel_size": args.eca_kernel_size,
        "params": params,
        "flops": flops,
    }
    if args.save_gate_stats:
        summary.update(
            collect_gate_summary_on_loader(
                model,
                val_loader,
                device,
                bins=args.gate_hist_bins,
                limit_batches=args.gate_stats_batches,
            )
        )
    if args.measure_latency:
        summary["latency_ms_b1"] = measure_latency_ms(model, args.image_size, device, batch_size=1)
        print(f"Latency: {summary['latency_ms_b1']:.3f} ms/image")

    summary_path = output_dir / f"{args.model}_{args.cgm_placement}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
