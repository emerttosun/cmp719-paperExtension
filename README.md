# CMP719 FasterNet Paper Extension

This repository contains the progress report and a lightweight experiment
pipeline for extending FasterNet with a Channel Gate Module (CGM).

## Project idea

The base paper is **Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
Networks**. The extension adds a small SE-style channel gate immediately after
PConv, but only on the channel subset processed by PConv. The goal is to test
whether input-adaptive channel recalibration improves CIFAR-100 accuracy without
adding meaningful FLOP or latency overhead.

## Quick start on Colab/Kaggle

```bash
pip install -r requirements.txt
python train_classification.py --model fasternet_t0 --dataset cifar100 --epochs 1 --batch-size 64
```

By default, CIFAR-100 is loaded from the Hugging Face dataset
`uoft-cs/cifar100`, avoiding the occasionally unavailable Toronto CIFAR host.
To force the original torchvision/Toronto loader, add
`--dataset-source torchvision`.

Fast smoke test:

```bash
python train_classification.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --dataset-source hf \
  --epochs 1 \
  --batch-size 64 \
  --limit-train-batches 5 \
  --limit-val-batches 2
```

If CIFAR-100 loading temporarily fails, use synthetic data only to verify the
training pipeline:

```bash
python train_classification.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --dataset-source hf \
  --epochs 1 \
  --batch-size 64 \
  --limit-train-batches 5 \
  --limit-val-batches 2 \
  --fallback-fake-data
```

Do not report FakeData accuracy as a real CIFAR-100 result.

CGM variant:

```bash
python train_classification.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --dataset-source hf \
  --epochs 1 \
  --batch-size 64 \
  --cgm-placement all \
  --measure-latency \
  --save-gate-stats
```

Model-only shape smoke test:

```bash
python scripts/smoke_model.py --model fasternet_t0 --image-size 32 --num-classes 100
```

Supported CGM placements:

- `none`: vanilla FasterNet baseline
- `all`: add CGM after PConv in all four stages
- `early`: add CGM only in stages 1 and 2
- `late`: add CGM only in stages 3 and 4
- `s1`, `s2`, `s3`, `s4`: add CGM only in one stage for per-stage ablation

Supported CGM modes:

- `sigmoid`: original gate, `x = x * sigmoid(gate)`
- `residual`: identity-preserving gate, `x = x * (1 + alpha * (sigmoid(gate) - 0.5))`

Supported CGM types:

- `se`: SE-style gate generator, `GAP -> 1x1 Conv -> ReLU -> 1x1 Conv -> Sigmoid`
- `eca`: ECA-style gate generator, `GAP -> 1D channel Conv -> Sigmoid`

Residual CGM example:

```bash
python train_classification.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --dataset-source hf \
  --epochs 20 \
  --batch-size 128 \
  --cgm-placement early \
  --cgm-mode residual \
  --measure-latency \
  --save-gate-stats \
  --output-dir runs_cifar_residual_e20
```

ECA-style CGM example:

```bash
python train_classification.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --dataset-source hf \
  --epochs 20 \
  --batch-size 128 \
  --cgm-placement early \
  --cgm-type eca \
  --measure-latency \
  --save-gate-stats \
  --output-dir runs_cifar_eca_e20
```

## Suggested preliminary experiments

```bash
python train_classification.py --model fasternet_t0 --cgm-placement none --epochs 20 --batch-size 128 --measure-latency
python train_classification.py --model fasternet_t0 --cgm-placement all --epochs 20 --batch-size 128 --measure-latency --save-gate-stats
python train_classification.py --model fasternet_t0 --cgm-placement early --epochs 20 --batch-size 128 --measure-latency
python train_classification.py --model fasternet_t0 --cgm-placement late --epochs 20 --batch-size 128 --measure-latency
```

Per-stage ablation:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s1 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s2 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s3 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s4 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
```

Outputs are written under `runs/` as CSV metrics and JSON summaries.

To plot layer-wise gate means after a CGM run:

```bash
python scripts/plot_gate_means.py runs/fasternet_t0_all_summary.json
```

## Tiny-ImageNet resized experiment

Tiny-ImageNet can be used to test whether the CIFAR-100 trend transfers to a
larger, more ImageNet-like dataset. The loader uses the Hugging Face dataset
`zh-plus/tiny-imagenet`, which has 200 classes and 64x64 images. We resize it to
224x224 for a closer match to the original FasterNet setting.

Smoke test:

```bash
python train_classification.py \
  --dataset tiny_imagenet \
  --dataset-source hf \
  --image-size 224 \
  --model fasternet_t0 \
  --epochs 1 \
  --batch-size 64 \
  --limit-train-batches 5 \
  --limit-val-batches 2 \
  --measure-latency
```

Recommended first comparison:

```bash
python train_classification.py --dataset tiny_imagenet --dataset-source hf --image-size 224 --model fasternet_t0 --epochs 20 --batch-size 64 --cgm-placement none --measure-latency --output-dir runs_tiny_e20
python train_classification.py --dataset tiny_imagenet --dataset-source hf --image-size 224 --model fasternet_t0 --epochs 20 --batch-size 64 --cgm-placement early --measure-latency --save-gate-stats --output-dir runs_tiny_e20
```

If Colab runs out of memory, reduce `--batch-size` to `32`.
