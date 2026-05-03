# CMP719 Project Context

Bu dosya sohbet cok buyudugu icin projenin mevcut teknik baglamini kisa tutmak amaciyla yazildi. Rapor yazarken veya yeni bir sohbette devam ederken buradan baslanabilir.

## Project Goal

Secilen paper: FasterNet, "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks".

Ana extension fikri: FasterNet'in Partial Convolution (PConv) blogundan hemen sonra hafif bir Channel Gate Module (CGM) eklemek. CGM sadece PConv tarafindan islenen kanal subset'i uzerinde calisir; bypass edilen kanallara dokunmaz.

Arastirma sorusu:

> PConv sonrasi channel gating, accuracy'yi artirirken FasterNet'in dusuk parametre/FLOPs/latency karakterini koruyabilir mi?

## Architecture Decisions

- Tum official FasterNet reposu alinmadi; detection, mmdet ve ImageNet-specific agir kisimlar disarida birakildi.
- Sade ve Colab/Kaggle dostu bir PyTorch pipeline kuruldu.
- Model hedefi ilk asamada FasterNet-T0; T1 opsiyonel final genisletmesi.
- Dataset hedefi hizli deney icin CIFAR-100; Tiny-ImageNet 224 resize transfer kontrolu icin eklendi.
- CIFAR-100 varsayilan olarak Hugging Face `uoft-cs/cifar100` uzerinden yukleniyor, cunku Toronto CIFAR host'u Colab'da 503 verebiliyor.
- Tiny-ImageNet Hugging Face `zh-plus/tiny-imagenet` uzerinden yukleniyor.
- CGM placement secenekleri:
  - `none`: baseline FasterNet
  - `all`: tum stage'lerde CGM
  - `early`: stage 1 + stage 2
  - `late`: stage 3 + stage 4
  - `s1`, `s2`, `s3`, `s4`: tek tek stage ablation
  - `s2s3`: stage 2 + stage 3 orta-stage aday kombinasyonu
  - `s2s4`: stage 2 + stage 4 komsu olmayan stage kombinasyonu
- CGM type secenekleri:
  - `se`: GAP -> 1x1 Conv -> ReLU -> 1x1 Conv -> sigmoid
  - `eca`: GAP -> 1D channel conv -> sigmoid
- CGM mode secenekleri:
  - `sigmoid`: `x = x * sigmoid(gate)`
  - `residual`: `x = x * (1 + alpha * (sigmoid(gate) - 0.5))`
- Kaydedilen olcumler:
  - top-1 accuracy
  - train/validation loss
  - parameter count
  - FLOPs
  - batch-1 latency
  - gate mean ve scale mean

## Modified / Important Files

- `train_classification.py`: CIFAR-100/Tiny-ImageNet training script, CLI arguments, logging, latency/FLOPs/gate stats.
- `src/fasternet_ext/models/fasternet.py`: FasterNet-T0/T1, PConv, CGM, placement/type/mode logic.
- `src/fasternet_ext/metrics.py`: params, FLOPs, latency helpers.
- `scripts/smoke_model.py`: model forward/shape/parameter smoke test.
- `scripts/plot_gate_means.py`: saved JSON summary'den gate mean plot uretir.
- `notebooks/train_classification.ipynb`: Colab/Kaggle calistirma notebook'u.
- `README.md`: kurulum, komutlar ve desteklenen CGM opsiyonlari.
- `PROJECT_PLAN.md`: preliminary ve final icin daha genis plan.
- `PROJECT_INTERNAL_REPORT.md`: deney notlari ve ara yorumlar.
- `main.tex`: progress report LaTeX dosyasi; son deneylerle guncellenmesi gerekiyor.
- `references.bib`: rapor referanslari.

## Current Experimental Findings

CIFAR-100, FasterNet-T0, 20 epoch, image size 32:

- Baseline `none`: seed 42 val acc 50.52, seed 7 val acc 50.21, mean 50.37.
- `early`: seed 42 val acc 51.03, seed 7 val acc 51.35, mean 51.19.
- `all`: mean yaklasik 50.71, latency belirgin daha yuksek.
- `late`: mean yaklasik 50.44, baseline'a cok yakin.
- `residual early`: seed 42 51.27, seed 7 51.00; baseline'dan iyi ama `sigmoid early` ortalamasindan biraz dusuk.
- `eca early`: seed 42 50.76; cok az parametre ekliyor ama gate neredeyse neutral kaliyor.

Per-stage ablation, CIFAR-100, seed 42:

- `s1`: val acc 50.52, latency 1.084 ms.
- `s2`: val acc 50.89, latency 1.150 ms.
- `s3`: val acc 51.39, latency 1.452 ms.
- `s4`: val acc 50.81, latency 1.144 ms.

En onemli yeni yorum:

> CGM'nin etkisi stage-dependent. Her yere eklemek iyi degil. Tek seed'de `s3-only` en iyi accuracy'yi verdi, fakat latency maliyeti daha yuksek. Bunu ana bulgu yapmadan once `s3` icin ikinci seed gerekir.

Tiny-ImageNet 224, FasterNet-T0, 20 epoch:

- Baseline `none`: val acc 45.38, latency 1.029 ms.
- `early`: val acc 45.04, latency 1.167 ms.

Tiny sonucu CIFAR kazancinin direkt transfer olmadigini gosteriyor. Bu sonuc finalde "dataset/resolution sensitivity" olarak yorumlanabilir; ana preliminary claim icin CIFAR stage ablation daha guclu.

## Remaining Work

Preliminary progress report icin:

- `main.tex` gercek sonuclarla guncellenmeli.
- "No final numerical results" gibi eski ifadeler kaldirilmali.
- CIFAR-100 baseline/early/all/late iki-seed tablosu eklenmeli.
- Per-stage ablation tablosu eklenmeli; `s3` icin ikinci seed yoksa bunu acikca preliminary olarak yaz.
- Gate mean plot veya tablo eklenmeli.
- Latency/FLOPs farki FasterNet paper'inin ana argumaniyla baglanmali.
- Tiny-ImageNet sonucu varsa "CIFAR kazanci transferde garanti degil" diye durustce yorumlanmali.

Hemen siradaki deney:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s3 --seed 7 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_seed7_e20
```

Final icin opsiyonel gelistirmeler:

- `s3` ikinci seed iyi gelirse `s3` ana accuracy adayi, `early/s2` latency-accuracy trade-off adayi olarak raporlanabilir.
- `s2+s3` gibi yeni bir combined placement eklenebilir.
- `s2s3` placement koda eklendi; seed 42/7 ile denenmesi gerekiyor.
- `s2s4` placement koda eklendi; komsu olmayan stage kombinasyonu kontrolu olarak denenebilir.
- 3 seed ortalama ve standart sapma raporlanabilir.
- 30/50 epoch daha uzun CIFAR-100 deneyleri yapilabilir.
- FasterNet-T1 baseline vs en iyi CGM varyanti denenebilir.
- Gate histogram, dogru/yanlis ornek gate analizi ve layer-wise gate visualizations eklenebilir.

## Important Constraints

- Colab/Kaggle hedef ortam; lokal makinede Python/PyTorch calismayabilir.
- Network kisitlari nedeniyle CIFAR icin Hugging Face loader daha guvenilir.
- FakeData sadece pipeline smoke test icindir; accuracy sonucu olarak raporlanmamalidir.
- 20 epoch sonuclari preliminary'dir; kucuk accuracy farklari seed etkisine duyarlidir.
- Latency degerleri ortam/GPU'ya gore degisir; ayni ortamda karsilastirma olarak kullanilmali.
- FLOPs tek basina yeterli yorum degildir; FasterNet projesinde latency ve operator overhead onemlidir.
- `s3` sonucu umut verici ama su an tek seed oldugu icin kesin sonuc gibi yazilmamali.
