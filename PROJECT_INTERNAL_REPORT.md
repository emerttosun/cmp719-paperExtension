# CMP719 Internal Experiment Report

Bu dosya rapor yazarken kullanmak icin projenin mevcut durumunu ve ara kararlarini ozetler. Nihai rapor dili daha akademik olacak; burasi daha cok kendi notumuzdur.

## 1. Proposal'daki Temel Fikir

Secilen calisma FasterNet paper'i: "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks".

Paper'in ana problemi: FLOPs'u azaltmak her zaman latency'yi azaltmaz. Memory access, operator tipi ve donanimin efektif kullanimi da gercek hizi belirler.

Bizim proposal fikrimiz:

- FasterNet'in PConv blogundan sonra hafif bir Channel Gate Module (CGM) eklemek.
- CGM'yi sadece PConv'un isledigi kanal subset'i uzerinde calistirmak.
- Amac: PConv tarafindan uretilen spatial feature kanallarini input'a gore yeniden agirliklandirmak.

Ana arastirma sorusu:

> PConv sonrasi channel gating, accuracy'yi artirirken parametre/FLOPs/latency overhead'ini dusuk tutabilir mi?

## 2. Su Ana Kadar Yapilanlar

Kod:

- Sade PyTorch FasterNet-T0/T1 implementasyonu yazildi.
- PConv ve FasterNet block yapisi kuruldu.
- CGM eklendi.
- `none`, `all`, `early`, `late` placement secenekleri eklendi.
- CIFAR-100 Hugging Face uzerinden calisir hale getirildi.
- FLOPs, parametre ve latency olcumleri eklendi.
- Gate mean degerleri JSON olarak kaydediliyor.
- Colab/Kaggle calistirma akisi hazirlandi.

Deney:

- CIFAR-100 uzerinde FasterNet-T0 ile 20 epoch deneyler yapildi.
- Iki seed kullanildi: seed 42 ve seed 7.
- Varyantlar: baseline, CGM-All, CGM-Early, CGM-Late.

## 3. CIFAR-100 Preliminary Sonuclari

Iki seed ortalama sonuclari:

| Variant | Seed 42 Val Acc | Seed 7 Val Acc | Mean Val Acc | Params | FLOPs | Approx. Latency |
|---|---:|---:|---:|---:|---:|---:|
| FasterNet-T0, CGM-none | 50.52 | 50.21 | 50.37 | 2,751,160 | 27.626M | ~1.009 ms |
| FasterNet-T0, CGM-all | 50.94 | 50.47 | 50.71 | 2,765,062 | 27.650M | ~1.674 ms |
| FasterNet-T0, CGM-early | 51.03 | 51.35 | 51.19 | 2,751,662 | 27.632M | ~1.177 ms |
| FasterNet-T0, CGM-late | 50.50 | 50.37 | 50.44 | 2,764,560 | 27.645M | ~1.521 ms |

Ana yorum:

- CGM-Early iki seed'de de en iyi validation accuracy verdi.
- Ortalama olarak baseline'a gore yaklasik `+0.82%` accuracy artisi var.
- CGM-Early sadece `+502` parametre ekliyor.
- FLOPs artisi cok kucuk.
- Latency artisi var ama CGM-All ve CGM-Late'e gore daha makul.
- CGM-All ve CGM-Late daha yuksek latency getiriyor; bu FasterNet paper'inin "FLOPs tek basina yeterli degildir" argumanini destekliyor.

Bu sonuclar buyuk bir performans sicrama iddiasi degil. Daha dogru yorum:

> Selective early-stage CGM promising bir trade-off gosteriyor, fakat final iddia icin daha uzun egitim, daha fazla seed veya daha uygun dataset gerekir.

## 4. Su Anki Eksiklikler

Preliminary report icin eksikler:

- `main.tex` henuz gercek deney sonuclariyla guncellenmedi.
- "No final numerical results" ifadesi kaldirilmali.
- CIFAR-100 iki-seed sonuclari tablo olarak eklenmeli.
- Gate plot rapora eklenmeli.
- Sonuclarin preliminary oldugu acikca yazilmali.

Teknik/deneysel eksikler:

- Tiny-ImageNet henuz denenmedi.
- FasterNet-T1 denenmedi.
- 50 epoch veya daha uzun egitim yok.
- CGM reduction ratio ablation yok (`r=8`, `r=16`).
- Pretrained FasterNet fine-tuning yok.
- Gate histogram / dogru-yanlis ornek analizi yok.

## 5. Neden Tiny-ImageNet Denemek Mantikli?

CIFAR-100 hizli ve kontrollu bir preliminary dataset. Ancak 32x32 oldugu icin FasterNet'in orijinal ImageNet-scale tasarimina tam uymuyor.

Tiny-ImageNet:

- 200 class iceriyor.
- 100k train, 10k validation image var.
- Orijinal image size 64x64.
- CIFAR-100'den daha zor.
- ImageNet baglamina CIFAR'dan daha yakin.

Tiny-ImageNet'i 224x224'e resize ederek denemek sunu test eder:

> CGM-Early'nin CIFAR-100'da gordugumuz etkisi daha yuksek cozunurluklu ve daha zor bir dataset'te de korunuyor mu?

Bu final proje icin iyi bir genisletme olur.

## 6. Tiny-ImageNet Icin Onerilen Deney

Tum varyantlari hemen calistirmak yerine once en kritik iki varyanti calistirmak daha mantikli:

1. Baseline: `cgm-placement none`
2. En iyi CIFAR varyanti: `cgm-placement early`

Onerilen ilk smoke test:

```bash
python train_classification.py --dataset tiny_imagenet --dataset-source hf --image-size 224 --model fasternet_t0 --epochs 1 --batch-size 64 --limit-train-batches 5 --limit-val-batches 2 --measure-latency
```

Onerilen preliminary Tiny-ImageNet deneyleri:

```bash
python train_classification.py --dataset tiny_imagenet --dataset-source hf --image-size 224 --model fasternet_t0 --epochs 20 --batch-size 64 --cgm-placement none --measure-latency --output-dir runs_tiny_e20
python train_classification.py --dataset tiny_imagenet --dataset-source hf --image-size 224 --model fasternet_t0 --epochs 20 --batch-size 64 --cgm-placement early --measure-latency --save-gate-stats --output-dir runs_tiny_e20
```

Not: 224x224 input CIFAR deneylerinden cok daha yavas olacaktir. Colab GPU memory sorun yaratirsa `--batch-size 32` kullanilacak.

## 7. Bundan Sonraki Mantikli Plan

Preliminary progress report icin:

1. CIFAR-100 sonuclarini rapora ekle.
2. CGM-Early'nin en iyi iki-seed trade-off oldugunu yaz.
3. Latency/FLOPs farkini yorumla.
4. Gate plot ekle.
5. Tiny-ImageNet'i "next step / final validation" olarak yaz.

Final proje icin:

1. Tiny-ImageNet baseline vs CGM-Early deneyini ekle.
2. CIFAR-100'da daha uzun egitim veya 3 seed ile sonucu saglamlastir.
3. CGM-Early icin `r=8` veya `r=16` dene.
4. Zaman kalirsa FasterNet-T1 baseline vs CGM-Early dene.
5. Gate visualization'i histogram ve dogru/yanlis ornek analiziyle genislet.

## 8. Tiny-ImageNet Sonucu ve Yeni CGM Gelistirme Fikri

Tiny-ImageNet 224x224 uzerinde ilk baseline vs early sonucu alindi:

| Dataset | Variant | Val Acc | Params | FLOPs | Latency |
|---|---:|---:|---:|---:|---:|
| Tiny-ImageNet 224 | CGM-none | 45.38 | 2,880,700 | 337.017M | 1.029 ms |
| Tiny-ImageNet 224 | CGM-early | 45.04 | 2,881,202 | 337.080M | 1.167 ms |

Bu sonuc CIFAR'daki kazancin Tiny-ImageNet'e direkt tasinmadigini gosterdi. CGM-Early burada baseline'dan `-0.34%` dusuk kaldi ve latency artti.

Tiny-ImageNet CGM-Early gate mean degerleri:

```text
stages.0.blocks.0.spatial_mixing.channel_gate = 0.5699
stages.2.blocks.0.spatial_mixing.channel_gate = 0.5370
stages.2.blocks.1.spatial_mixing.channel_gate = 0.2673
```

Bu degerler semantik olarak hangi feature'in bastirildigini kanitlamaz. Ancak mevcut sigmoid CGM'nin bir early block'ta PConv-processed kanallari ortalamada guclu sekilde down-scale ettigini gosterir.

Bu nedenle yeni denenecek fikir:

> Residual / identity-preserving CGM

Eski CGM:

```text
x = x * sigmoid(gate)
```

Yeni residual CGM:

```text
scale = 1 + alpha * (sigmoid(gate) - 0.5)
x = x * scale
```

Varsayilan `alpha=0.5` icin scale araligi:

```text
0.75 - 1.25
```

Boylece CGM feature'i tamamen bastirmak yerine baseline etrafinda hafif azaltma/artirma yapar. Bu, ozellikle Tiny-ImageNet gibi daha zor datasetlerde feature akisini koruyabilir.

Ilk denenecek CIFAR komutu:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement early --cgm-mode residual --measure-latency --save-gate-stats --output-dir runs_cifar_residual_e20
```

Residual CGM seed 42/7 sonucunda baseline'dan iyi ama sigmoid CGM-Early ortalamasindan biraz dusuk kaldi. Bu nedenle bir sonraki aday ECA-style CGM'dir.

ECA-style CGM fikri:

- Mevcut SE-style CGM bottleneck kullanir: `GAP -> 1x1 Conv -> ReLU -> 1x1 Conv -> Sigmoid`.
- ECA-style CGM bottleneck kullanmaz: `GAP -> 1D channel Conv -> Sigmoid`.
- Amac, PConv sonrasi channel interaction'i daha az bilgi sikistirmasiyla yakalamaktir.

Ilk denenecek ECA komutu:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement early --cgm-type eca --measure-latency --save-gate-stats --output-dir runs_cifar_eca_e20
```

## 9. Per-Stage Ablation Plani

Early placement CIFAR-100'da iyi calisti, fakat bunun Stage 1'den mi Stage 2'den mi geldigi henuz bilinmiyor. Bu nedenle tek tek stage ablation eklendi:

- `s1`: sadece Stage 1
- `s2`: sadece Stage 2
- `s3`: sadece Stage 3
- `s4`: sadece Stage 4

Amac:

- Early sonucunu hangi stage'in surukledigini bulmak.
- Eger `s1` veya `s2` tek basina yeterliyse, daha hafif ve daha net bir extension elde etmek.
- `s3/s4` zayif kalirsa late placement'in neden etkili olmadigini daha iyi aciklamak.

Calistirilacak komutlar:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s1 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s2 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s3 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s4 --measure-latency --save-gate-stats --output-dir runs_cifar_stage_e20
```

## 10. Per-Stage ve Combined Placement Sonuclari

Per-stage ablation, CIFAR-100, FasterNet-T0, 20 epoch, seed 42:

| Variant | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---:|---:|---:|---:|---|
| `s1` | 50.52 | 2,751,212 | 27.629M | 1.084 ms | Baseline ile ayni |
| `s2` | 50.89 | 2,751,610 | 27.629M | 1.150 ms | Kucuk ve ucuz kazanc |
| `s3` | 51.39 | 2,757,960 | 27.638M | 1.452 ms | Tek seed'de en yuksek, ama pahali |
| `s4` | 50.81 | 2,757,760 | 27.633M | 1.144 ms | Kucuk kazanc |

`s3` icin seed 7 tekrari:

| Variant | Seed | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---:|---:|---:|---:|---:|---|
| `s3` | 7 | 50.29 | 2,757,960 | 27.638M | 1.487 ms | Seed 42 sonucu stabil degil |

Combined placement denemeleri:

| Variant | Seed | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---:|---:|---:|---:|---:|---|
| `s2s3` | 42 | 51.04 | 2,758,410 | 27.641M | 1.559 ms | Accuracy early ile benzer, latency daha kotu |
| `s2s4` | 42 | 51.06 | 2,758,210 | 27.636M | 1.264 ms | Early'ye yakin, latency makul |

Yeni yorum:

> `early` hala en guvenli ana yontem. `s3` seed-sensitive gorunuyor. `s2s3`, `early` kadar iyi accuracy verse de latency maliyeti yuksek. `s2s4` ise ilginc bir ikinci aday: seed 42'de early ile benzer accuracy ve makul latency verdi.

`s2s4` gate mean degerlerinde son stage ikinci block oldukca yuksek cikti:

```text
stages.6.blocks.0.spatial_mixing.channel_gate = 0.5671
stages.6.blocks.1.spatial_mixing.channel_gate = 0.7816
```

Bu, `s2s4` varyantinda gec stage gate'lerinin bazi PConv kanallarini ortalamada daha guclu gecirdigini gosterir. Bu semantik olarak hangi feature'in secildigini kanitlamaz, ama stage-specific gate behavior icin raporda kullanilabilecek guzel bir gozlemdir.

## 11. Yeni Aday: GAP+GMP Pooling

`early` en iyi trade-off oldugu icin CGM'nin pooling sinyali guclendirildi.

Eski CGM pooling:

```text
GAP(x) -> shared gate -> sigmoid
```

Yeni opsiyonel pooling:

```text
GAP(x) -> shared gate logits
GMP(x) -> same shared gate logits
sum logits -> sigmoid
```

Yeni CLI parametresi:

```bash
--cgm-pooling gap_gmp
```

Ilk denenmesi gereken iki aday:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement early --cgm-pooling gap_gmp --seed 42 --measure-latency --save-gate-stats --output-dir runs_cifar_early_gap_gmp_e20
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement s2s4 --cgm-pooling gap_gmp --seed 42 --measure-latency --save-gate-stats --output-dir runs_cifar_s2s4_gap_gmp_e20
```

Karar kurali:

- Eger `early + gap_gmp` eski `early gap` ortalamasi olan 51.19'u gecerse ana yontem `Early Dual-Pooling CGM` olabilir.
- Eger `s2s4 + gap_gmp` en iyi sonucu verirse ana aday `Selective S2+S4 Dual-Pooling CGM` olabilir.
- Ikisi de iyilesmezse eski `early gap` ana yontem kalir; GAP+GMP ablation olarak raporlanir.

## 12. Sonradan Tamamlanan Testlerin Ozeti

Bu bolum, rapor yazarken "hangi deneyler yapildi?" sorusuna hizli cevap vermek icin eklendi.

### 12.1 GAP+GMP Sonuclari

| Variant | Seed | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---:|---:|---:|---:|---:|---|
| `early + gap_gmp` | 42 | 50.90 | 2,751,662 | 27.632M | 1.317 ms | Early icin iyilesme saglamadi |
| `s2s4 + gap_gmp` | 42 | 51.39 | 2,758,210 | 27.643M | 1.419 ms | Tek seed'de yuksek |
| `s2s4 + gap_gmp` | 7 | 50.59 | 2,758,210 | 27.643M | 1.429 ms | Seed 7'de dusuk, stabil degil |

Yorum:

> GAP+GMP pooling, erken stage'de gate'i daha agresif hale getirdi fakat accuracy'yi artirmadi. `s2s4` icin tek seed'de iyi gorundu ancak seed 7'de stabil kalmadi.

### 12.2 Identity Init Sonuclari

Identity init fikri, sigmoid gate'in baslangicta `0.5` ile PConv kanallarini yarilamasini engellemek icin denendi.

| Variant | Seed | Val Acc | Gate Means | Latency | Yorum |
|---|---:|---:|---|---:|---|
| `early + init b=4` | 42 | 51.25 | ~0.977-0.979 | 1.212 ms | Baseline ve vanilla seed42'den iyi |
| `early + init b=4` | 7 | 51.06 | ~0.977-0.980 | 1.202 ms | Vanilla early seed7'den dusuk |
| `early + init b=2` | 42 | 51.20 | ~0.867-0.878 | 1.201 ms | Gate daha az pasif ama vanilla'yi gecmedi |

Iki seed ortalamasi:

| Variant | Mean Val Acc | Yorum |
|---|---:|---|
| `early + gap` | 51.19 | En stabil 20e ana yontem |
| `early + init b=4` | 51.16 | Cok yakin ama daha iyi degil |

Yorum:

> Identity-biased initialization teorik olarak anlamli ve baseline'dan iyi, fakat bu deneylerde vanilla `early + gap` yontemini net gecmedi. Gate mean'lerin init degerine yakin kalmasi, gate'in mevcut recipe/epoch sayisinda sinirli hareket ettigini gosteriyor.

### 12.3 40 Epoch Sonuclari

40 epoch ile baseline guclendi ve CGM farki kuculdu.

| Variant | Seed | Val Acc | Train Acc | Params | FLOPs | Latency | Yorum |
|---|---:|---:|---:|---:|---:|---:|---|
| `none` | 42 | 55.51 | 77.29 | 2,751,160 | 27.626M | 1.035 ms | Guclu baseline |
| `early + gap` | 42 | 55.68 | 77.88 | 2,751,662 | 27.632M | 1.193 ms | En iyi 40e sonuc |
| `early + init b=2` | 42 | 55.64 | 78.15 | 2,751,662 | 27.632M | 1.200 ms | Vanilla early'den az dusuk |
| `s2s4 + gap` | 42 | 55.37 | 77.41 | 2,758,210 | 27.636M | 1.261 ms | Baseline'dan dusuk |

Yorum:

> CGM etkisi 40 epoch'ta da kaybolmadi, fakat fark `+0.17` ile kucuk kaldi. Bu, CGM'nin daha cok kisa egitimde yardimci oldugunu veya baseline'in uzun egitimde farki kapattigini dusunduruyor.

### 12.4 CIFAR-100 64x64 Resize Sonuclari

CIFAR-100 native 32x32 oldugu icin 64x64 deneyleri ana protokol degil, resolution sensitivity analizidir.

| Variant | Image Size | Epoch | Seed | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `none` | 64 | 20 | 42 | 56.69 | 2,751,160 | 108.892M | 1.046 ms | Resize accuracy'yi ciddi artirdi |
| `early + gap` | 64 | 20 | 42 | 56.92 | 2,751,662 | 108.913M | 1.223 ms | Baseline'a gore +0.23 |

Yorum:

> 64x64 resize, baseline ve CGM performansini ciddi artirdi. Ancak CGM'nin relatif kazanci yine kucuk kaldi. Bu nedenle dusuk cozunurluk tek basina CGM etkisinin sinirli kalmasini aciklamiyor.

## 13. Genel Sonuc ve Sonraki Olasiliklar

Genel sonuc:

> Proposal kismen desteklendi. `early + gap` CGM, CIFAR-100'da kucuk ve tekrar eden bir iyilesme sagliyor. Ancak etki guclu degil; uzun egitimde fark azaliyor, Tiny-ImageNet'e transfer etmiyor ve daha karmasik varyantlar stabil kazanc vermiyor.

Yapilmis ana test kategorileri:

- Baseline vs `all`/`early`/`late`, 20 epoch, iki seed.
- Per-stage ablation: `s1`, `s2`, `s3`, `s4`.
- Combined placement: `s2s3`, `s2s4`.
- CGM mode/type: residual, ECA.
- Pooling: GAP vs GAP+GMP.
- Initialization: sigmoid identity init `b=4`, `b=2`.
- Daha uzun egitim: 40 epoch baseline, early, early+init, s2s4.
- Dataset/resolution: Tiny-ImageNet 224, CIFAR-100 resized 64.
- Efficiency: params, FLOPs, latency.
- Gate analysis: layer-wise gate means.

Bundan sonra denenebilecekler:

- Stronger recipe: warmup, TrivialAugment, MixUp, EMA. Adil karsilastirma icin baseline ve CGM ayni recipe ile egitilmeli.
- Centered sigmoid gate: `scale = 1 + alpha * (2 * sigmoid(logits) - 1)`. `alpha=0.5` icin scale araligi `0.5-1.5` olur. Bu, gate'in hem azaltma hem artirma yapmasini saglarken tamamen sert `0-2` araligindan daha kontrolludur.
- Bypass channel gate: PConv islenen kanallar ve bypass kanallarini ayri gate'lemek, FasterNet'e daha ozgu bir extension olur.
- Gate std/histogram: mean yerine kanal bazli dagilim kaydedilmeli; gate gercekten ayrisiyor mu daha net gorulur.
- T1 modeli: `fasternet_t1` baseline vs `early + gap`.
- 64x64 icin seed 7 veya 40 epoch tekrar, sadece zaman/GPU yeterse.

## 14. Yeni Ana Aday: Centered Early CGM ve Guclu Gate Analysis

Mevcut sigmoid CGM sadece `0-1` araliginda scale uretiyor; bu nedenle PConv kanallarini guclendirmek yerine sadece bastirabiliyor veya en fazla ayni seviyeye yaklastirabiliyor. Proposal'daki "suppress or amplify" fikrine daha uygun yeni aday:

```text
gate = sigmoid(logits)
scale = 1 + alpha * (2 * gate - 1)
x = x * scale
```

`alpha=0.5` icin scale araligi:

```text
0.5 - 1.5
```

Bu mod `centered` olarak koda eklendi. Ana denenecek konfigurasyon:

```bash
python train_classification.py --dataset cifar100 --dataset-source hf --image-size 32 --model fasternet_t0 --epochs 20 --batch-size 128 --cgm-placement early --cgm-mode centered --cgm-alpha 0.5 --measure-latency --save-gate-stats --output-dir runs_cifar_centered_e20
```

Gate analysis de guclendirildi. Yeni summary JSON artik sadece mean degil, her CGM katmani icin sunlari da kaydediyor:

- gate mean/std/min/max
- gate histogram
- effective scale mean/std/min/max
- effective scale histogram

Plot komutlari:

```bash
python scripts/plot_gate_means.py runs_cifar_centered_e20/fasternet_t0_early_summary.json --stat gate
python scripts/plot_gate_means.py runs_cifar_centered_e20/fasternet_t0_early_summary.json --stat scale
python scripts/plot_gate_means.py runs_cifar_centered_e20/fasternet_t0_early_summary.json --stat scale --hist
```
