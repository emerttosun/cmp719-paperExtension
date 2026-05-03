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
- CGM pooling secenekleri:
  - `gap`: eski davranis, sadece Global Average Pooling
  - `gap_gmp`: ayni gate agirliklariyla GAP + Global Max Pooling sinyallerini birlestirir
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

Follow-up placement results:

| Variant | Seed | Val Acc | Params | FLOPs | Latency | Note |
|---|---:|---:|---:|---:|---:|---|
| `s3` | 7 | 50.29 | 2,757,960 | 27.638M | 1.487 ms | Seed 42'deki 51.39 stabilize olmadi |
| `s2s3` | 42 | 51.04 | 2,758,410 | 27.641M | 1.559 ms | Early ile benzer accuracy, daha yavas |
| `s2s4` | 42 | 51.06 | 2,758,210 | 27.636M | 1.264 ms | Early'ye cok yakin, makul latency |

En onemli yeni yorum:

> CGM'nin etkisi stage-dependent. Her yere eklemek iyi degil. `s3-only` seed 42'de parlak gorundu ama seed 7'de dusuk geldi, bu yuzden ana yontem olmamali. Su an en guvenli ana aday `early`; en ilginc ikinci aday ise `s2s4`.

Tiny-ImageNet 224, FasterNet-T0, 20 epoch:

- Baseline `none`: val acc 45.38, latency 1.029 ms.
- `early`: val acc 45.04, latency 1.167 ms.

Tiny sonucu CIFAR kazancinin direkt transfer olmadigini gosteriyor. Bu sonuc finalde "dataset/resolution sensitivity" olarak yorumlanabilir; ana preliminary claim icin CIFAR stage ablation daha guclu.

Additional completed tests after the first ablations:

| Setting | Variant | Seed | Val Acc | Params | FLOPs | Latency | Main note |
|---|---|---:|---:|---:|---:|---:|---|
| CIFAR-100 32, 20e | `early + gap_gmp` | 42 | 50.90 | 2,751,662 | 27.632M | 1.317 ms | Dual pooling hurt early CGM |
| CIFAR-100 32, 20e | `s2s4 + gap_gmp` | 42 | 51.39 | 2,758,210 | 27.643M | 1.419 ms | Good single-seed result |
| CIFAR-100 32, 20e | `s2s4 + gap_gmp` | 7 | 50.59 | 2,758,210 | 27.643M | 1.429 ms | Gain did not stabilize |
| CIFAR-100 32, 20e | `early + init b=4` | 42 | 51.25 | 2,751,662 | 27.632M | 1.212 ms | Identity init works but remains near 0.98 |
| CIFAR-100 32, 20e | `early + init b=4` | 7 | 51.06 | 2,751,662 | 27.632M | 1.202 ms | Mean below vanilla early |
| CIFAR-100 32, 20e | `early + init b=2` | 42 | 51.20 | 2,751,662 | 27.632M | 1.201 ms | Gate remains near 0.87 |
| CIFAR-100 32, 40e | `none` | 42 | 55.51 | 2,751,160 | 27.626M | 1.035 ms | Longer training greatly improves baseline |
| CIFAR-100 32, 40e | `early + gap` | 42 | 55.68 | 2,751,662 | 27.632M | 1.193 ms | Best 40e result, small gain |
| CIFAR-100 32, 40e | `early + init b=2` | 42 | 55.64 | 2,751,662 | 27.632M | 1.200 ms | Does not beat vanilla early |
| CIFAR-100 32, 40e | `s2s4 + gap` | 42 | 55.37 | 2,758,210 | 27.636M | 1.261 ms | Worse than baseline at 40e |
| CIFAR-100 64, 20e | `none` | 42 | 56.69 | 2,751,160 | 108.892M | 1.046 ms | Resize boosts overall accuracy |
| CIFAR-100 64, 20e | `early + gap` | 42 | 56.92 | 2,751,662 | 108.913M | 1.223 ms | Still positive, but modest +0.23 |

Current consolidated conclusion:

> The proposal is partially supported but not strongly validated. Early-stage SE-CGM with GAP is the most reliable variant: it improves CIFAR-100 at 20 epochs and remains slightly better at 40 epochs. However, the gain is modest, shrinks with longer training, does not transfer cleanly to Tiny-ImageNet, and most stronger-looking variants are seed-sensitive or slower.

## Remaining Work

Preliminary progress report icin:

- `main.tex` gercek sonuclarla guncellenmeli.
- "No final numerical results" gibi eski ifadeler kaldirilmali.
- CIFAR-100 baseline/early/all/late iki-seed tablosu eklenmeli.
- Per-stage ablation tablosu eklenmeli; `s3` icin ikinci seed yoksa bunu acikca preliminary olarak yaz.
- Gate mean plot veya tablo eklenmeli.
- Latency/FLOPs farki FasterNet paper'inin ana argumaniyla baglanmali.
- Tiny-ImageNet sonucu varsa "CIFAR kazanci transferde garanti degil" diye durustce yorumlanmali.

Hemen siradaki deney / rapor onceligi:

- Ana yontemi `early + gap` olarak raporla.
- 20 epoch ve 40 epoch tablolarini birlikte ver.
- `gap_gmp`, `identity init`, `s2s4`, `s3`, `64x64`, Tiny-ImageNet sonuclarini ablation/sensitivity olarak yaz.
- Sonucu "small but reproducible on CIFAR, limited generality" diye konumlandir.

Final icin opsiyonel gelistirmeler:

- Daha guclu training recipe: warmup, TrivialAugment, MixUp, EMA. Baseline ve CGM ayni recipe ile egitilmeli.
- Gate histogram/std kaydi: mean tek basina yeterli degil; kanal bazli ayrisma gorulmeli.
- Centered gate tasarimi: `scale = 2 * sigmoid(logits)` ile azaltma ve artirma birlikte denenebilir.
- Bypass gate: PConv islenen kanallar ve bypass kanallari ayri gate'lemek FasterNet'e daha ozgu bir katkidir.
- T1 deneyi: baseline vs `early + gap`; proposal'in farkli model boyutuna genellenip genellenmedigini test eder.
- CIFAR-100 64x64 icin seed 7 tekrar veya 40 epoch, sadece zaman kalirsa.
- 3 seed ortalama ve standart sapma raporlanabilir.
- Gate histogram, dogru/yanlis ornek gate analizi ve layer-wise gate visualizations eklenebilir.

## Important Constraints

- Colab/Kaggle hedef ortam; lokal makinede Python/PyTorch calismayabilir.
- Network kisitlari nedeniyle CIFAR icin Hugging Face loader daha guvenilir.
- FakeData sadece pipeline smoke test icindir; accuracy sonucu olarak raporlanmamalidir.
- 20 epoch sonuclari preliminary'dir; kucuk accuracy farklari seed etkisine duyarlidir.
- Latency degerleri ortam/GPU'ya gore degisir; ayni ortamda karsilastirma olarak kullanilmali.
- FLOPs tek basina yeterli yorum degildir; FasterNet projesinde latency ve operator overhead onemlidir.
- `s3` seed 42 sonucu tek basina kesin sonuc gibi yazilmamali; seed 7'de 50.29'a dustu.
