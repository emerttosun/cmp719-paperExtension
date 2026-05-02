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
python train_cifar.py --model fasternet_t0 --dataset cifar100 --epochs 1 --batch-size 64
```

Fast smoke test:

```bash
python train_cifar.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --epochs 1 \
  --batch-size 64 \
  --limit-train-batches 5 \
  --limit-val-batches 2
```

If CIFAR-100 download temporarily fails with an HTTP 503 error, use synthetic
data only to verify the training pipeline:

```bash
python train_cifar.py \
  --model fasternet_t0 \
  --dataset cifar100 \
  --epochs 1 \
  --batch-size 64 \
  --limit-train-batches 5 \
  --limit-val-batches 2 \
  --fallback-fake-data
```

Do not report FakeData accuracy as a real CIFAR-100 result.

CGM variant:

```bash
python train_cifar.py \
  --model fasternet_t0 \
  --dataset cifar100 \
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

## Suggested preliminary experiments

```bash
python train_cifar.py --model fasternet_t0 --cgm-placement none --epochs 20 --batch-size 128 --measure-latency
python train_cifar.py --model fasternet_t0 --cgm-placement all --epochs 20 --batch-size 128 --measure-latency --save-gate-stats
python train_cifar.py --model fasternet_t0 --cgm-placement early --epochs 20 --batch-size 128 --measure-latency
python train_cifar.py --model fasternet_t0 --cgm-placement late --epochs 20 --batch-size 128 --measure-latency
```

Outputs are written under `runs/` as CSV metrics and JSON summaries.

To plot layer-wise gate means after a CGM run:

```bash
python scripts/plot_gate_means.py runs/fasternet_t0_all_summary.json
```
