# Final Report Worklog

Bu dosya final raporu yazarken kullanilacak canli calisma notudur. Colab'dan yeni sonuclar geldikce buraya islenecek. Amac, final raporda hangi deneyin neden yapildigini, hangi komutla calistigini ve nasil yorumlanacagini kaybetmemektir.

## 1. Current Final Story

Ana hipotez:

> PConv sonrasi hafif Channel Gate Module (CGM), FasterNet'in accuracy/latency trade-off'unu iyilestirebilir mi?

Simdiki preliminary sonuc:

- CIFAR-100 32x32 uzerinde FasterNet-T0 icin `early` CGM kucuk ama tekrar eden bir gain verdi.
- Gain buyuk degil; 40 epoch'ta baseline guclenince fark kuculdu.
- T1 40 epoch seed 42'de `early` CGM baseline'i gecti.
- Tiny-ImageNet'e transfer temiz degil; mevcut recipe Tiny'de overfit sinyali verdi.
- FLOPs overhead cok kucuk olsa da latency overhead gercek. Bu, FasterNet'in "FLOPs tek basina yeterli degildir" argumanini destekliyor.

Final raporda ana pozisyon:

> CGM partially supports the proposal: early-stage gating gives modest gains on CIFAR-100, but the effect is recipe-, dataset-, and training-length dependent. Stronger augmentation experiments test whether the gain survives a stronger baseline.

## 2. Augmentation Plan

Yeni kodda `--augmentation` secenegi var:

- `simple`: eski recipe. Onceki deneylerle uyumlu kalir.
- `cifar`: CIFAR-100 icin stronger augmentation.
- `tiny`: Tiny-ImageNet icin stronger augmentation.

Summary JSON'a tek alan yazilir:

- `simple` icin `"augmentation": "basic"`
- `cifar` veya `tiny` icin `"augmentation": "stronger"`

### 2.1 CIFAR Stronger Augmentation

`--augmentation cifar`:

- `RandomCrop(image_size, padding=4)`
- `RandomHorizontalFlip`
- `TrivialAugmentWide`
- `ToTensor`
- `Normalize(CIFAR100 mean/std)`
- `RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3))`

Amac:

- Mevcut CIFAR gain'inin stronger image-level augmentation altinda kalip kalmadigini test etmek.
- MixUp/CutMix/EMA simdilik yok; cunku loss/evaluation pipeline'i da degistirir ve CGM etkisini izole etmeyi zorlastirir.

### 2.2 Tiny Stronger Augmentation

`--augmentation tiny`:

- `RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.75, 1.33))`
- `RandomHorizontalFlip`
- `TrivialAugmentWide`
- `ToTensor`
- `Normalize(ImageNet mean/std)`
- `RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3))`

Amac:

- Tiny-ImageNet'te gorulen overfit sinyalini azaltmak.
- Deterministic `Resize(224)` yerine daha degisken crop/scale gostererek train-val gap'i dusurmek.

## 3. Immediate Experiment Plan: T0 CIFAR Stronger Augmentation

Ilk karar:

> Once FasterNet-T0 uzerinde CIFAR-100 stronger augmentation full placement sweep yapilacak. T0 sonucuna gore T1 deneyleri daraltilacak veya genisletilecek.

Deney matrisi:

| Dataset | Model | Image Size | Epoch | Augmentation | Placements | Seeds |
|---|---|---:|---:|---|---|---|
| CIFAR-100 | FasterNet-T0 | 32 | 20 | `cifar` / stronger | `none`, `all`, `early`, `late` | 42, 7 |

Toplam 8 run.

Karar sorusu:

> Stronger CIFAR augmentation altinda `early` CGM hala `none` baseline'i geciyor mu?

Olasiliklar:

- `early` hala pozitifse: CGM gain daha robust gorunur; T1'e yatirim mantikli.
- Gain kuculurse: CGM faydasi stronger baseline ile azaliyor yorumu yapilir.
- Gain kaybolursa veya negatife donerse: CGM faydasi simple recipe'ye bagimli olabilir; T1 deneyleri sadece critical pair (`none`, `early`) olarak daraltilabilir.

## 4. Colab Commands: T0 CIFAR Stronger Augmentation

Once repo guncelle:

```bash
!git pull
```

### Seed 42 / CGM-none / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement none --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s42
```

### Seed 42 / CGM-all / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement all --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s42
```

### Seed 42 / CGM-early / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s42
```

### Seed 42 / CGM-late / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement late --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s42
```

### Seed 7 / CGM-none / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement none --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s7
```

### Seed 7 / CGM-all / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement all --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s7
```

### Seed 7 / CGM-early / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement early --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s7
```

### Seed 7 / CGM-late / T0 / CIFAR stronger

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement late --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s7
```

## 5. Result Table Template: T0 CIFAR Stronger Augmentation

Doldurulacak tablo:

| Recipe | Placement | Seed 42 Val Acc | Seed 7 Val Acc | Mean Val Acc | Params | FLOPs | Latency | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Basic/simple 20e | `none` | 50.52 | 50.21 | 50.37 | 2,751,160 | 27.626M | 1.009 ms | Existing result |
| Basic/simple 20e | `all` | 50.94 | 50.47 | 50.71 | 2,765,062 | 27.650M | 1.674 ms | Existing result |
| Basic/simple 20e | `early` | 51.03 | 51.35 | 51.19 | 2,751,662 | 27.632M | 1.177 ms | Existing result |
| Basic/simple 20e | `late` | 50.50 | 50.37 | 50.44 | 2,764,560 | 27.645M | 1.521 ms | Existing result |
| Stronger/cifar 20e | `none` | 45.06 | TBD | TBD | 2,751,160 | 27.63M | 4.230 ms | Seed 42 done; best val epoch 19, final val 44.95 |
| Stronger/cifar 20e | `all` | TBD | TBD | TBD | TBD | TBD | TBD | New run |
| Stronger/cifar 20e | `early` | 45.42 | TBD | TBD | 2,751,662 | 27.63M | 4.843 ms | Seed 42 done; best/final val 45.42 |
| Stronger/cifar 20e | `late` | TBD | TBD | TBD | TBD | TBD | TBD | New run |

### Completed Run Notes

#### 2026-06-04 / T0 CIFAR stronger / seed 42 / CGM-none

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement none --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s42
```

Result:

- Final epoch train acc: 37.148
- Final epoch val acc: 44.95
- Best val acc: 45.06 at epoch 19
- Params: 2,751,160
- FLOPs: 27.63M
- Latency: 4.230 ms/image
- Output files:
  - `runs_cifar_t0_aug_e20_s42/fasternet_t0_none_metrics.csv`
  - `runs_cifar_t0_aug_e20_s42/fasternet_t0_none_summary.json`

Initial interpretation:

- Stronger CIFAR augmentation makes optimization harder in 20 epochs: train acc is much lower than the earlier basic/simple 20e baseline.
- Validation accuracy is also lower than the basic/simple 20e baseline, so 20 epochs may be too short for this stronger recipe.
- Latency should be compared mainly against the other augmented runs from the same Colab/runtime, not directly against previous latency numbers from different environments.

#### 2026-06-04 / T0 CIFAR stronger / seed 42 / CGM-early

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e20_s42
```

Result:

- Final epoch train acc: 37.254
- Final epoch val acc: 45.42
- Best val acc: 45.42 at epoch 19/20
- Params: 2,751,662
- FLOPs: 27.63M
- Latency: 4.843 ms/image
- Output files:
  - `runs_cifar_t0_aug_e20_s42/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_aug_e20_s42/fasternet_t0_early_summary.json`

Comparison vs stronger `none` seed 42:

- Best-val gain: `45.42 - 45.06 = +0.36`
- Final-val gain: `45.42 - 44.95 = +0.47`
- Latency overhead: `4.843 - 4.230 = +0.613 ms`

Initial interpretation:

- `early` still beats `none` under stronger CIFAR augmentation on seed 42, but the gain is modest.
- Stronger augmentation appears under-trained at 20 epochs for both baseline and early CGM, so this may need either more epochs or a milder augmentation recipe for final-quality results.

## 6. How To Read The T0 Augmentation Results

Bakilacak ana metrikler:

1. `early - none` mean accuracy gain.
2. `all` latency overhead vs `early`.
3. `late` accuracy/latency trade-off.
4. Stronger recipe baseline'i ne kadar guclendirdi.
5. Stronger recipe altinda CGM gain buyudu mu, kuculdu mu, kayboldu mu.

Rapor yorum kaliplari:

### If Early Still Wins

> Under stronger CIFAR augmentation, early-stage CGM remains better than the baseline, suggesting that the channel-gating benefit is not only an artifact of the simple training recipe.

### If Early Gain Shrinks

> Stronger augmentation improves the baseline and reduces the marginal gain from CGM, indicating that part of the preliminary improvement may come from regularisation-like behavior under the simpler recipe.

### If Early Loses

> The CGM gain does not survive the stronger CIFAR recipe, suggesting that the proposed gate is recipe-dependent rather than a universally beneficial architectural addition.

## 6.1 Follow-up: T0 CIFAR Stronger Augmentation, 40 Epoch Critical Pair

20 epoch stronger CIFAR runs looked under-trained, so we decided to test whether the same recipe needs more training.

Critical pair:

- T0 / CIFAR / stronger augmentation / seed 42 / `none`
- T0 / CIFAR / stronger augmentation / seed 42 / `early`

Rationale for not running full placement sweep at this stage:

- Stronger augmentation makes each run longer and increases GPU cost.
- `early` is already the main candidate from the earlier T0 placement sweep.
- The immediate final-report question is whether the main candidate survives a stronger baseline, so the most efficient comparison is the critical pair `none` vs `early`.
- `all` and `late` can be skipped unless the critical pair suggests that a broader placement sweep is worth the extra GPU time.

### Result Table

| Recipe | Epoch | Placement | Seed | Train Acc Final | Val Acc Final | Best Val Acc | Params | FLOPs | Latency | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Stronger/cifar | 40 | `none` | 42 | 47.29 | 52.13 | 52.28 | 2,751,160 | 27.63M | 4.259 ms | Best val epoch 39 |
| Stronger/cifar | 40 | `early` | 42 | 47.47 | 52.83 | 52.83 | 2,751,662 | 27.63M | 4.750 ms | Best/final val epoch 40 |

### Completed Run Notes

#### 2026-06-04 / T0 CIFAR stronger / seed 42 / CGM-none / 40 epoch

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 40 --batch-size 128 --seed 42 --cgm-placement none --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e40_s42
```

Result:

- Final epoch train acc: 47.29
- Final epoch val acc: 52.13
- Best val acc: 52.28 at epoch 39
- Params: 2,751,160
- FLOPs: 27.63M
- Latency: 4.259 ms/image
- Output files:
  - `runs_cifar_t0_aug_e40_s42/fasternet_t0_none_metrics.csv`
  - `runs_cifar_t0_aug_e40_s42/fasternet_t0_none_summary.json`

Initial interpretation:

- The stronger CIFAR recipe was not simply bad; 20 epochs were too short.
- Baseline improved from best val 45.06 at 20e to 52.28 at 40e.
- Final train acc is still much lower than the basic/simple 40e baseline train acc, so stronger augmentation regularizes training heavily.
- Need `early` 40e seed 42 to check whether CGM still adds value under this stronger 40e baseline.

#### 2026-06-04 / T0 CIFAR stronger / seed 42 / CGM-early / 40 epoch

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 40 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode sigmoid --augmentation cifar --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_aug_e40_s42
```

Result:

- Final epoch train acc: 47.47
- Final epoch val acc: 52.83
- Best val acc: 52.83 at epoch 40
- Params: 2,751,662
- FLOPs: 27.63M
- Latency: 4.750 ms/image
- Output files:
  - `runs_cifar_t0_aug_e40_s42/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_aug_e40_s42/fasternet_t0_early_summary.json`

Comparison vs stronger `none` 40e seed 42:

- Best-val gain: `52.83 - 52.28 = +0.55`
- Final-val gain: `52.83 - 52.13 = +0.70`
- Latency overhead: `4.750 - 4.259 = +0.491 ms`

Initial interpretation:

- With enough training, stronger CIFAR augmentation recovers useful validation accuracy.
- `early` still improves over `none` under the stronger 40e recipe.
- The gain is modest but larger than the simple 40e T0 gain noted earlier (`+0.17`), at least for seed 42.
- Need seed 7 critical pair before making a reproducibility claim.

## 7. Next Decision: T1 Plan After T0

T0 stronger augmentation sonucundan sonra T1 icin karar:

### If T0 `early` is still useful

Run T1 stronger augmentation:

- `none`
- `early`
- optionally `all`, `late`
- first seed 42, then seed 7 if promising

### If T0 `early` is not useful

Run only critical T1 check:

- `none`
- `early`
- seed 42

If negative, do not spend GPU on full T1 sweep.

## 8. Tiny-ImageNet Final Plan

Tiny-ImageNet simdilik T0 CIFAR kararindan sonra yapilacak.

Mevcut Tiny bulgusu:

- Tiny 224, 20e baseline: train 61.54, val 45.28.
- Tiny 224, 40e baseline: train 82.86, val 46.84.
- Train-val gap yaklasik 36 puan, overfit sinyali.
- Tiny 64, 40e baseline: train 91.64, val 42.66; daha sert overfit.

Bu nedenle Tiny icin onerilen yeni recipe:

- `--image-size 224`
- `--augmentation tiny`
- first run: `none` vs `early`
- seed 42 first

Tiny final question:

> Stronger Tiny augmentation, CIFAR'daki CGM gain'inin daha zor/higher-resolution setting'e transfer etmesini sagliyor mu?

### 8.1 Tiny Stronger Augmentation Results

| Recipe | Epoch | Placement | Seed | Train Acc Final | Val Acc Final | Best Val Acc | Params | FLOPs | Latency | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Basic/simple | 20 | `none` | 42 | 61.54 | 45.28 | 45.28 | 2,880,700 | 337.017M | 1.016 ms | Existing Tiny baseline rerun |
| Basic/simple | 20 | `early` | 42 | 60.40 | 45.06 | 45.06 | 2,881,202 | 337.080M | 1.188 ms | Existing Tiny early sigmoid |
| Basic/simple | 40 | `none` | 42 | 82.86 | 46.84 | 46.84 | 2,880,700 | 337.017M | 1.012 ms | Large train-val gap |
| Stronger/tiny | 40 | `none` | 42 | 50.03 | 49.22 | 49.22 | 2,880,700 | 337.02M | 3.906 ms | Stronger augmentation baseline done |
| Stronger/tiny | 40 | `early` | 42 | TBD | TBD | TBD | TBD | TBD | TBD | Next critical run |

#### 2026-06-04 / Tiny stronger / T0 / seed 42 / CGM-none / 40 epoch

Command:

```bash
!python train_classification.py --dataset tiny_imagenet --dataset-source hf --image-size 224 --model fasternet_t0 --epochs 40 --batch-size 64 --seed 42 --cgm-placement none --cgm-mode sigmoid --augmentation tiny --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_tiny_t0_aug_e40_s42
```

Result:

- Final epoch train acc: 50.03
- Final epoch val acc: 49.22
- Best val acc: 49.22 at epoch 40
- Params: 2,880,700
- FLOPs: 337.02M
- Latency: 3.906 ms/image
- Output files:
  - `runs_tiny_t0_aug_e40_s42/fasternet_t0_none_metrics.csv`
  - `runs_tiny_t0_aug_e40_s42/fasternet_t0_none_summary.json`

Comparison vs basic/simple Tiny baseline:

- Basic/simple 40e baseline: train 82.86, val 46.84.
- Stronger/tiny 40e baseline: train 50.03, val 49.22.
- Val improvement: `49.22 - 46.84 = +2.38`.
- Train-val gap changed from about `36.02` points to about `0.81` points.

Initial interpretation:

- Tiny-specific stronger augmentation substantially reduces the overfit seen in the basic/simple 40e Tiny baseline.
- Validation accuracy improves by +2.38 points over the basic/simple 40e baseline.
- This strongly supports the final-report motivation for applying augmentation on Tiny-ImageNet rather than making it a main CIFAR experiment.
- Need `early` 40e with `--augmentation tiny` to check whether CGM transfers under the improved Tiny training recipe.

## 9. Residual Scaling Ablation

Goal:

- Test whether an identity-centered gate is better than the default suppress-only sigmoid gate.
- Keep this as a design ablation, not the main result.
- Dataset/model: CIFAR-100, FasterNet-T0, no stronger augmentation (`--augmentation simple`), `early` placement.

Reference baselines:

| Variant | Epoch | Seed | Final Val Acc | Notes |
|---|---:|---:|---:|---|
| `none` | 20 | 42 | 50.52 | No CGM baseline |
| `early sigmoid` | 20 | 42 | 51.03 | Main CGM design baseline |
| `early sigmoid` | 20 | 7 | 51.35 | Main CGM design baseline |

### 9.1 Alpha 0.5

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode residual --cgm-alpha 0.5 --augmentation simple --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_residual_a05_e20_s42
```

Result:

- Final epoch train acc: 61.014
- Final epoch val acc: 51.12
- Best val acc: 51.47 at epoch 18
- Params: 2,751,662
- FLOPs: 27.63M
- Latency: 5.039 ms/image
- Output files:
  - `runs_cifar_t0_residual_a05_e20_s42/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_residual_a05_e20_s42/fasternet_t0_early_summary.json`

Initial interpretation:

- Residual alpha 0.5 beats seed-42 early sigmoid on best validation accuracy (`51.47` vs `51.03`), but final epoch accuracy is only slightly higher (`51.12` vs `51.03`).
- This is promising enough to run alpha 1.0 before deciding whether residual scaling deserves a second seed.
- Do not overinterpret latency yet: this run reports much higher latency than earlier CIFAR simple runs, so latency should only be compared if measured under the same runtime/device conditions.

### 9.2 Alpha 1.0

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode residual --cgm-alpha 1.0 --augmentation simple --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_residual_a10_e20_s42
```

Result:

- Final epoch train acc: 61.006
- Final epoch val acc: 51.18
- Best val acc: 51.49 at epoch 18
- Params: 2,751,662
- FLOPs: 27.63M
- Latency: 4.824 ms/image
- Output files:
  - `runs_cifar_t0_residual_a10_e20_s42/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_residual_a10_e20_s42/fasternet_t0_early_summary.json`

Initial interpretation:

- Residual alpha 1.0 is essentially tied with alpha 0.5: best val `51.49` vs `51.47`, final val `51.18` vs `51.12`.
- Both residual settings are above seed-42 early sigmoid (`51.03`) and above the no-CGM seed-42 baseline (`50.52`).
- The two alpha runs being almost identical suggests that, in this CIFAR T0 early setting, the learned gates stay close enough to 0.5 that changing alpha from 0.5 to 1.0 has only a small practical effect.
- Next decision: if residual scaling is worth including as more than a small ablation, repeat alpha 1.0 with seed 7; otherwise report it as a single-seed design probe.

### 9.3 Alpha 1.0, Seed 7

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement early --cgm-mode residual --cgm-alpha 1.0 --augmentation simple --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_residual_a10_e20_s7
```

Result:

- Final epoch train acc: 60.18
- Final epoch val acc: 51.01
- Best val acc: 51.03 at epoch 18
- Params: 2,751,662
- FLOPs: 27.63M
- Latency: 5.286 ms/image
- Output files:
  - `runs_cifar_t0_residual_a10_e20_s7/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_residual_a10_e20_s7/fasternet_t0_early_summary.json`

Two-seed comparison:

| Variant | Seeds | Mean Final Val Acc | Mean Best Val Acc | Notes |
|---|---|---:|---:|---|
| `early sigmoid` | 42, 7 | 51.19 | TBD | Main design |
| `early residual alpha=1.0` | 42, 7 | 51.095 | 51.26 | Identity-centered design probe |

Interpretation:

- Residual alpha 1.0 does not clearly beat the simpler sigmoid gate once seed 7 is included.
- It remains above the no-CGM baseline, so the result supports CGM usefulness, but not replacing sigmoid as the main gate mode.
- Final-report phrasing: residual scaling was tested as an identity-centered alternative; it produced competitive but not consistently better results than vanilla sigmoid, so the main method stays `early + sigmoid`.

### 9.4 Alpha 0.5, Seed 7

Command:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement early --cgm-mode residual --cgm-alpha 0.5 --augmentation simple --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_residual_a05_e20_s7
```

Result:

- Final epoch train acc: 60.154
- Final epoch val acc: 51.07
- Best val acc: 51.07 at epoch 18/20
- Params: 2,751,662
- FLOPs: 27.63M
- Latency: 5.211 ms/image
- Output files:
  - `runs_cifar_t0_residual_a05_e20_s7/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_residual_a05_e20_s7/fasternet_t0_early_summary.json`

Updated two-seed comparison:

| Variant | Seeds | Mean Final Val Acc | Mean Best Val Acc | Notes |
|---|---|---:|---:|---|
| `early sigmoid` | 42, 7 | 51.19 | TBD | Main design |
| `early residual alpha=0.5` | 42, 7 | 51.095 | 51.27 | Identity-centered design probe |
| `early residual alpha=1.0` | 42, 7 | 51.095 | 51.26 | Identity-centered design probe |

Final interpretation for residual ablation:

- Alpha 0.5 and alpha 1.0 are effectively tied.
- Both residual settings are competitive with the sigmoid gate, but neither clearly improves the two-seed result.
- This supports keeping `early + sigmoid` as the main design because it is simpler and already performs at least as well on average.
- Gate/scale statistics should be inspected from the saved summary JSON files before final writing. If scale means stay close to `1.0`, report that residual gates mostly remained near identity, explaining why alpha changes had little effect.

Suggested final-report phrasing:

> We also tested identity-centered residual scaling as an alternative to the default sigmoid gate. Although residual scaling was competitive, its two-seed CIFAR-100 result did not consistently improve over the simpler sigmoid gate, so we kept early sigmoid CGM as the main design.

Why this ablation is useful:

- It shows that the residual/identity-centered design was considered and tested rather than ignored.
- It makes the final method choice more mature: we did not simply pick the first gate formula; we compared it against a plausible safer alternative.
- It should be presented as a compact design ablation, not as the main result.

Gate-statistics checks to do before final writing:

- `scale_mean`
  - For sigmoid, scale is directly the gate and usually lies in `[0, 1]`.
  - For residual, scale is `1 + alpha * (gate - 0.5)`.
  - If residual scale mean stays near `1.0`, we can say the identity-centered design mostly stayed close to identity.
- `scale_std`
  - If it is low, the gate is not strongly separating channels.
  - If alpha 0.5 and alpha 1.0 have very similar scale distributions, this can explain why their accuracies were almost identical.
- Stage/block-level gate and scale means
  - With `early` placement, CGM appears only in a few early PConv blocks.
  - Check which blocks suppress or amplify channels more strongly.

Possible final-report sentence after inspecting JSON files:

> Gate-statistics inspection showed that residual scaling stayed close to identity on average, which helps explain why increasing alpha from 0.5 to 1.0 changed accuracy only marginally.

Files to preserve from Colab:

- `runs_cifar_t0_residual_a05_e20_s42/fasternet_t0_early_summary.json`
- `runs_cifar_t0_residual_a05_e20_s7/fasternet_t0_early_summary.json`
- `runs_cifar_t0_residual_a10_e20_s42/fasternet_t0_early_summary.json`
- `runs_cifar_t0_residual_a10_e20_s7/fasternet_t0_early_summary.json`
- comparison:
  - `runs_cifar_t0_early_e20_s42_v2/fasternet_t0_early_summary.json`
  - seed 7 sigmoid summary, if available

Final-report scope:

- Inspect the gates, but do not make this a main figure unless the statistics are unusually clear.
- Best use: one or two sentences supporting the residual-scaling decision.

## 10. Files To Watch

Her run sonunda beklenen dosyalar:

- `fasternet_t0_none_metrics.csv`
- `fasternet_t0_none_summary.json`
- `fasternet_t0_all_metrics.csv`
- `fasternet_t0_all_summary.json`
- `fasternet_t0_early_metrics.csv`
- `fasternet_t0_early_summary.json`
- `fasternet_t0_late_metrics.csv`
- `fasternet_t0_late_summary.json`

Summary JSON'dan alinacak alanlar:

- `augmentation`
- `params`
- `flops`
- `latency_ms_b1`
- `gate_means` / `gate_stats` if present

Metrics CSV'den alinacak alanlar:

- En iyi validation accuracy mi kullanilacak?
- Final epoch validation accuracy mi kullanilacak?

Not:

> Onceki tablolarda bazi degerler best validation accuracy gibi kullanildi. Final raporda bunu netlestirmek gerekiyor. Tavsiye: tablolarda "best val acc over training" olarak raporla ve metinde acikla.

## 11. GAP+STD Descriptor Ablation

Motivation:

- GAP only tells the gate the mean activation of each PConv-processed channel.
- STD adds spatial dispersion: whether a channel is broadly active or concentrated in sharper spatial patterns.
- This stays on-proposal because the gate still observes and scales only the PConv-processed channel subset (`c_p`).
- This is a better on-proposal design probe than a competitive softmax gate because it enriches the descriptor without forcing zero-sum channel competition or unstable amplification.

Implementation:

- Added `--cgm-pooling gap_std`.
- For SE-style CGM, the descriptor is `concat(GAP(x), STD(x))`.
- The first 1x1 gate layer receives `2C` descriptor channels and still outputs gates for the original `C` PConv channels.
- `gap_std` is intentionally limited to `--cgm-type se`.

First experiment:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode sigmoid --cgm-pooling gap_std --augmentation simple --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_gapstd_e20_s42
```

Compare against:

- `early sigmoid + gap`, seed 42, 20e: final val `51.03`.
- `none`, seed 42, 20e: final val `50.52`.

Decision rule:

- If `gap_std` clearly improves over `51.03`, run seed 7.
- If it is similar or worse, keep it as a compact negative design ablation.

Possible final-report phrasing if negative:

> We also tested a GAP+STD descriptor to expose spatial dispersion to the gate, but it did not improve over the simpler GAP descriptor in our CIFAR-100 T0 ablation.

Possible final-report phrasing if positive:

> Adding per-channel spatial dispersion through a GAP+STD descriptor improved the early CGM result, suggesting that PConv channel selection benefits from seeing not only average activation strength but also spatial concentration.

### 11.1 GAP+STD, Seed 42 Result

Result:

- Final epoch train acc: 59.32
- Final epoch val acc: 50.15
- Best val acc: 50.22 at epoch 19
- Params: 2,751,882
- FLOPs: 27.63M
- Latency: 4.843 ms/image
- Output files:
  - `runs_cifar_t0_gapstd_e20_s42/fasternet_t0_early_metrics.csv`
  - `runs_cifar_t0_gapstd_e20_s42/fasternet_t0_early_summary.json`

Comparison:

- `early sigmoid + GAP`, seed 42, 20e: final val `51.03`.
- `early sigmoid + GAP+STD`, seed 42, 20e: final val `50.15`, best val `50.22`.
- `none`, seed 42, 20e: final val `50.52`.

Interpretation:

- GAP+STD did not improve over the simpler GAP descriptor.
- It also did not beat the no-CGM seed-42 baseline in this 20-epoch CIFAR T0 run.
- The added descriptor signal may have made the gate harder to optimize, or the mean activation signal may already be sufficient for this lightweight CGM.
- Do not run seed 7 unless we specifically want a stronger negative result; current evidence is enough to keep GAP as the main descriptor.

Final-report phrasing:

> We tested a GAP+STD descriptor to provide the gate with both mean activation and spatial dispersion, but it underperformed the simpler GAP descriptor in the CIFAR-100 T0 ablation. We therefore kept GAP as the default gate descriptor.

## 12. Final Report Additions To Remember

Final rapora eklenmesi en mantikli seyler:

1. T0 basic vs stronger augmentation placement table.
2. Gate means combined figure.
3. T0 40 epoch simple result as training-length sensitivity.
4. Tiny negative/overfit result as dataset-transfer limitation.
5. Reduction ratio result (`r=4` best) as compact ablation.
6. Additional variants summary:
   - GAP+GMP did not stabilize.
   - Identity init did not beat vanilla early.
   - Centered/residual was tested as an identity-centered design variant; current CIFAR alpha 0.5 result is promising but still single-seed.
   - ECA was not the main winner.

## 13. RepPConv Experiment Plan

Motivation:

- CGM scales the PConv-processed channels after the spatial 3x3 convolution.
- The following PWConv1 can partially absorb static channel rescaling, which may explain why CGM gains are small and seed-sensitive.
- RepPConv changes the PConv training parameterization itself: during training, the processed channel subset uses parallel 3x3, 1x1, and identity branches.
- The intended inference story is structural re-parameterization: these branches can be folded into a single 3x3 convolution, so inference should recover the original PConv graph.

Implemented switch:

- CLI flag: `--pconv-reparam`
- Default: off, preserving all previous runs.
- When enabled, each PConv over `X_p` becomes:

```text
RepPConv(X_p) = BN(Conv3x3(X_p)) + BN(Conv1x1(X_p)) + BN(X_p)
```

Current implementation status:

- Training-time RepPConv is implemented.
- `--pconv-reparam` is passed from CLI to `build_fasternet` and then to every `PartialConv3`.
- Summary JSON records `"pconv_reparam": true/false`.
- Smoke tests passed:
  - model forward with RepPConv only
  - model forward with RepPConv + early CGM
  - one-batch fake CIFAR training run
- Branch folding / deploy conversion is not implemented yet. For final report claims about inference latency matching baseline, we must either implement and test folding or phrase it as theoretical/future deploy conversion.

Important interpretation note:

- Training-time parameter count and FLOPs increase because the branches are present.
- The theoretical benefit is that these branches are foldable into one 3x3 PConv at inference.
- Until folding is implemented and verified, latency measured by `--measure-latency` is training-graph latency, not fused deploy latency.

First Colab smoke/real run:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement none --pconv-reparam --augmentation simple --measure-latency --output-dir runs_cifar_t0_reppconv_e20_s42
```

Compare against:

- Baseline T0 CIFAR simple 20e seed 42: final val `50.52`.
- CGM early T0 CIFAR simple 20e seed 42: final val `51.03`.

Decision rule:

- If RepPConv seed 42 is near or above CGM early (`51.03`), run 40e and/or seed 7.
- If RepPConv clearly underperforms, do not expand it for final report.

If first run is promising, next commands:

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 42 --cgm-placement early --cgm-mode sigmoid --pconv-reparam --augmentation simple --measure-latency --save-gate-stats --gate-hist-bins 20 --gate-stats-batches 10 --output-dir runs_cifar_t0_reppconv_cgm_e20_s42
```

```bash
!python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --seed 7 --cgm-placement none --pconv-reparam --augmentation simple --measure-latency --output-dir runs_cifar_t0_reppconv_e20_s7
```

Final report framing if positive:

> RepPConv improves the PConv training parameterization directly through structural over-parameterization, whereas CGM applies a lightweight input-dependent channel rescaling after PConv. This tests whether improving the spatial branch itself provides a stronger accuracy-latency trade-off than post-PConv gating.

Final report framing if negative:

> We also tested a structurally over-parameterized RepPConv training branch, but it did not improve the small-scale CIFAR-100 result enough to justify replacing the simpler PConv+CGM design.
