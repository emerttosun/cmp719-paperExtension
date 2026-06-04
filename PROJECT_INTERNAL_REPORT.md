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
python scripts/plot_gate_means.py runs_cifar_centered_e20/fasternet_t0_early_summary.json --stat scale --print-vectors
```

Onemli not: Onceki implementation'da `--measure-latency` gate stats'tan once calistigi icin, kaydedilen `latest_gate` degerleri validation image'lari yerine latency olcumundeki random tensor tarafindan ezilebiliyordu. Bu nedenle gate analysis validation loader uzerinden yenilenecek sekilde guncellendi. Yeni summary `gate_stats_num_images`, `gate_vectors` ve `scale_vectors` alanlarini da kaydeder.

## 15. Tiny-ImageNet Centered CGM Sonuclari

Centered CGM, proposal'daki "suppress or amplify" fikrini sigmoid-only gate'ten daha iyi temsil etmek icin denendi. Tiny-ImageNet 224 uzerinde ayni seed ve ayni 20 epoch protokolunde baseline tekrar calistirildi.

| Dataset | Variant | Alpha | Seed | Train Acc | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Tiny-ImageNet 224 | `none` | - | 42 | 61.54 | 45.28 | 2,880,700 | 337.017M | 1.016 ms | Guncel baseline |
| Tiny-ImageNet 224 | `early + centered` | 0.50 | 42 | 60.16 | 44.42 | 2,881,202 | 337.080M | 1.253 ms | Genis scale araligi zarar verdi |
| Tiny-ImageNet 224 | `early + centered` | 0.25 | 42 | 60.20 | 44.47 | 2,881,202 | 337.080M | 1.229 ms | Daha kontrollu ama baseline'i gecmedi |

Gate/scale analizi:

- `alpha=0.5` icin scale araligi `0.5-1.5`. Stage 1 ve Stage 2 kanal ortalamalari cogunlukla amplify yonunde kaydi. Stage 2 block 1 sample-level scale araligi `0.537-1.475` oldu; yani module oldukca agresif suppress/amplify yapabildi. Buna ragmen validation accuracy baseline'dan `-0.86` dusuk kaldi.
- `alpha=0.25` icin scale araligi `0.75-1.25`. Stage 1 yine hafif amplify etti (`scale mean 1.023`), fakat Stage 2 block 0 ve block 1 suppress yonune kaydi (`scale mean 0.971` ve `0.930`). Kanal ortalamalarinda Stage 2 block 0 icin 13/20 kanal suppressed, 6/20 amplified; Stage 2 block 1 icin 20/20 kanal suppressed gorundu.
- Alpha'yi daraltmak gate davranisini daha kontrollu ve stage-dependent hale getirdi, fakat accuracy'yi toparlamadi (`44.42 -> 44.47`).

Yeni Tiny yorumu:

> Tiny-ImageNet'te problem sadece gate'in fazla agresif olmasi degil. Hem genis amplify agirlikli centered CGM hem de daha dar/suppress agirlikli centered CGM baseline'dan dusuk kaldi. Bu, CIFAR-100'da gorulen CGM kazancinin Tiny-ImageNet'e genellenmedigini ve early PConv channel recalibration'in daha yuksek cozunurluklu/daha zor setting'de feature akisini bozabilecegini gosteriyor.

Efficiency yorumu:

> Parametre ve FLOPs overhead'i cok kucuk (`+502` parametre, yaklasik `+0.063M` FLOPs), ancak latency `1.016 ms` baseline'dan `1.229-1.253 ms` araligina cikti. Bu, FasterNet paper'inin "FLOPs tek basina yeterli degildir" argumanini destekleyen bir negatif sonuc olarak raporlanabilir.

## 16. FasterNet-T1 CIFAR-100 Sonuclari

T1 deneyi, CGM etkisinin sadece kucuk T0 modeline mi ait oldugunu yoksa daha genis FasterNet varyantina da tasinip tasinmadigini anlamak icin yapildi.

### 16.1 T1 20 Epoch Sonuclari

| Model | Variant | Seed | Epoch | Train Acc | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| T1 | `none` | 42 | 20 | 65.91 | 54.00 | 6,440,164 | 69.807M | 1.055 ms | Baseline |
| T1 | `none` | 7 | 20 | 66.49 | 53.63 | 6,440,164 | 69.807M | 1.062 ms | Baseline seed tekrar |
| T1 | `early + sigmoid` | 42 | 20 | 65.26 | 52.79 | 6,441,416 | 69.816M | 1.245 ms | Kisa egitimde zararli |
| T1 | `early + centered` | 42 | 20 | 65.76 | 52.98 | 6,441,416 | 69.816M | 1.301 ms | Sigmoid'den az iyi ama baseline altinda |
| T1 | `s1 + centered` | 42 | 20 | 66.53 | 53.41 | 6,440,312 | 69.811M | 1.149 ms | Early'den iyi, baseline altinda |
| T1 | `s2 + centered` | 42 | 20 | 66.85 | 54.15 | 6,441,268 | 69.812M | 1.224 ms | Tek seed'de baseline'a yakin/az ustu |

20 epoch yorumu:

> T1'de 20 epoch sonuclari CGM icin yanıltici olabilir. `early + sigmoid` ve `early + centered` baseline'dan belirgin dusuk kaldi. `s2 + centered` tek seed'de baseline'i cok az gecti, fakat fark baseline seed varyansindan kucuk oldugu icin guclu claim yapmaya yetmez.

### 16.2 T1 40 Epoch Sigmoid Stage Ablation

T1 daha buyuk bir model oldugu icin 40 epoch tekrar yapildi. 40 epoch'ta sigmoid CGM baseline'i gecti ve stage-wise trend daha anlamli hale geldi.

| Model | Variant | Seed | Epoch | Train Acc | Val Acc | Gain vs Baseline | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T1 | `none` | 42 | 40 | 85.66 | 56.71 | - | 6,440,164 | 69.807M | 1.056 ms | 40e baseline |
| T1 | `early + sigmoid` | 42 | 40 | 84.90 | 57.37 | +0.66 | 6,441,416 | 69.816M | 1.244 ms | En iyi accuracy |
| T1 | `s1 + sigmoid` | 42 | 40 | 85.94 | 57.22 | +0.51 | 6,440,312 | 69.811M | 1.120 ms | En iyi latency/accuracy trade-off |
| T1 | `s2 + sigmoid` | 42 | 40 | 85.95 | 57.20 | +0.49 | 6,441,268 | 69.812M | 1.191 ms | S1'e yakin gain, daha yavas |
| T1 | `s3 + sigmoid` | 42 | 40 | 85.49 | 56.95 | +0.24 | 6,457,188 | 69.832M | 1.517 ms | Kucuk gain, pahali latency |
| T1 | `s4 + sigmoid` | 42 | 40 | 85.18 | 56.84 | +0.13 | 6,456,868 | 69.825M | 1.181 ms | Kucuk gain |

40 epoch yorumu:

> T1 icin CGM etkisi egitim suresine duyarlidir. 20 epoch'ta early CGM zararli gorunurken, 40 epoch'ta `early + sigmoid` baseline'a gore `+0.66` val acc kazanci sagladi. Train accuracy baseline'dan biraz dusuk (`84.90` vs `85.66`) ama validation daha yuksek oldugu icin sigmoid CGM'nin T1'de regularization/channel filtering etkisi olabilir.

Stage-wise yorum:

- `s1` ve `s2` tek basina benzer kazanc sagladi (`+0.51`, `+0.49`).
- `early = s1+s2`, en yuksek accuracy'yi verdi (`57.37`), ancak latency maliyeti daha yuksek.
- `s1`, en iyi accuracy/latency trade-off olarak gorunuyor: baseline'a gore `+0.51` val acc, sadece `+0.064 ms` latency.
- `s3` ve `s4` daha genis/gec stage'ler oldugu icin daha fazla parametre/latency getirdi; accuracy kazanci daha kucuk kaldi.

Yeni T1 sonucu:

> T1 deneyleri, CGM'nin sadece T0'a ozgu olmadigini fakat yeterli egitim ve dogru placement gerektirdigini gosteriyor. 40 epoch'ta erken stage sigmoid CGM faydali hale geldi; buna karsin FLOPs artisi cok kucuk kalirken latency artisi belirgin oldu. Bu sonuc hem proposal'in early-stage fikrini destekler, hem de FasterNet'in "FLOPs tek basina yeterli degildir" argumanini tekrar dogrular.

## 17. CGM Reduction Ratio Ablation

Proposal'da CGM bottleneck reduction ratio `r` icin ablation planlanmisti. Varsayilan deger `r=4` idi. `r=2` daha genis/kapasiteli gate, `r=8` daha dar/hafif gate anlamina gelir.

| Model | Setting | Reduction `r` | Epoch | Seed | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| T0 | `none` | - | 20 | 42 | 50.52 | 2,751,160 | 27.626M | ~1.009 ms | Baseline |
| T0 | `early + sigmoid` | 2 | 20 | 42 | 50.64 | 2,752,135 | 27.632M | 1.188 ms | Baseline'dan az iyi, r=4'ten dusuk |
| T0 | `early + sigmoid` | 4 | 20 | 42 | 51.03 | 2,751,662 | 27.632M | ~1.18 ms | En iyi T0 r |
| T0 | `early + sigmoid` | 8 | 20 | 42 | 50.39 | 2,751,395 | 27.632M | 1.192 ms | Fazla dar bottleneck, baseline altinda |
| T1 | `none` | - | 40 | 42 | 56.71 | 6,440,164 | 69.807M | 1.056 ms | Baseline |
| T1 | `early + sigmoid` | 2 | 40 | 42 | 57.06 | 6,442,588 | 69.818M | 1.227 ms | Baseline'dan iyi, r=4'ten dusuk |
| T1 | `early + sigmoid` | 4 | 40 | 42 | 57.37 | 6,441,416 | 69.816M | 1.244 ms | En iyi T1 r |
| T1 | `early + sigmoid` | 8 | 40 | 42 | 56.84 | 6,440,830 | 69.816M | 1.236 ms | Gain zayif |

Reduction ratio yorumu:

> Hem T0 hem T1 icin en iyi sonuc `r=4` ile geldi. `r=2` daha fazla gate kapasitesi sunsa da accuracy'yi artirmadi; `r=8` ise daha hafif olmasina ragmen etkiyi zayiflatti. Latency farklari reduction ratio'lar arasinda kucuk oldugu icin `r=4` en iyi accuracy-overhead trade-off olarak kaldi.

## 18. Son Eklenen Sonuclar ve Duzeltmeler

Bu bolum, sonradan elde edilen ama yukaridaki ana tablolara tam islenmemis sonuclari toplar.

### 18.1 Tiny-ImageNet Early Sigmoid Tekrari

Tiny-ImageNet 224 uzerinde early sigmoid sonucu tekrar calistirildi. Bu sonuc, centered denemelerden onceki proposal'a en yakin CGM konfigudur: `early + sigmoid + r=4 + GAP`.

| Dataset | Variant | Epoch | Seed | Train Acc | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Tiny-ImageNet 224 | `none` | 20 | 42 | 61.54 | 45.28 | 2,880,700 | 337.017M | 1.016 ms | Guncel baseline |
| Tiny-ImageNet 224 | `early + sigmoid` | 20 | 42 | 60.40 | 45.06 | 2,881,202 | 337.080M | 1.188 ms | Baseline'a cok yakin ama gecemedi |
| Tiny-ImageNet 224 | `early + centered, alpha=0.5` | 20 | 42 | 60.16 | 44.42 | 2,881,202 | 337.080M | 1.253 ms | Genis suppress/amplify araligi zararli |
| Tiny-ImageNet 224 | `early + centered, alpha=0.25` | 20 | 42 | 60.20 | 44.47 | 2,881,202 | 337.080M | 1.229 ms | Daha kontrollu ama baseline altinda |

Tiny icin guncel yorum:

> Proposal'a en yakin sigmoid CGM, Tiny-ImageNet'te centered varyantlardan daha iyi ve baseline'a yakin kaldi (`45.06` vs `45.28`). Ancak yine de baseline'i gecmedi ve latency'yi artirdi. Bu nedenle Tiny tarafinda "CGM aktif calisiyor ama transfer kazanci yok" yorumu en dogru yorumdur.

Tiny early sigmoid gate mean degerleri:

```text
stages.0.blocks.0.spatial_mixing.channel_gate = 0.5678
stages.2.blocks.0.spatial_mixing.channel_gate = 0.5796
stages.2.blocks.1.spatial_mixing.channel_gate = 0.4168
```

Bu gate davranisi su anlama gelir:

- Sigmoid mode'da scale = gate oldugu icin tum degerler `0-1` araliginda suppression derecesidir.
- Stage 1 ve Stage 2 block 0, PConv kanallarini ortalamada yaklasik `0.57-0.58` ile geciriyor.
- Stage 2 block 1 daha sert bastiriyor (`0.42` civari).
- Bu bastirma Tiny'de validation kazancina donusmedi.

### 18.2 Tiny-ImageNet 40 Epoch Baseline ve Overfit

Tiny-ImageNet 224 baseline 40 epoch calistirildi.

| Dataset | Variant | Image Size | Epoch | Seed | Train Acc | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tiny-ImageNet | `none` | 224 | 20 | 42 | 61.54 | 45.28 | 2,880,700 | 337.017M | 1.016 ms | 20e baseline |
| Tiny-ImageNet | `none` | 224 | 40 | 42 | 82.86 | 46.84 | 2,880,700 | 337.017M | 1.012 ms | Val artti ama train-val gap cok buyudu |

Yorum:

> 40 epoch baseline, validation'i `45.28 -> 46.84` artirdi fakat train accuracy `82.86`'ya cikti. Train-val farki yaklasik `36` puan oldugu icin Tiny tarafinda mevcut recipe ciddi overfit ediyor. Bu durumda CGM'yi Tiny'de daha uzun egitimle test etmek tek basina yeterli olmayabilir; daha guclu augmentation/regularization gerekebilir.

### 18.3 Tiny-ImageNet 64x64 Baseline

Tiny-ImageNet'in native cozunurlugune daha yakin `64x64` baseline denendi.

| Dataset | Variant | Image Size | Epoch | Seed | Train Acc | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tiny-ImageNet | `none` | 64 | 40 | 42 | 91.64 | 42.66 | 2,879,260 | 109.020M | 1.023 ms | 224'e gore daha dusuk val, daha sert overfit |

Yorum:

> Tiny 64x64, FLOPs'u dusurdu ama validation'i toparlamadi. Aksine train accuracy `91.64`, validation `42.66` oldu. Bu, Tiny tarafindaki sorunun sadece input resolution olmadigini; mevcut training recipe'nin overfit ettigini gosteriyor.

### 18.4 T1 Early Sigmoid Seed 7

T1 40 epoch icin early sigmoid seed 7 de calistirildi.

| Model | Variant | Epoch | Seed | Train Acc | Val Acc | Params | FLOPs | Latency | Yorum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| T1 | `none` | 40 | 42 | 85.66 | 56.71 | 6,440,164 | 69.807M | 1.056 ms | Baseline seed 42 |
| T1 | `early + sigmoid` | 40 | 42 | 84.90 | 57.37 | 6,441,416 | 69.816M | 1.244 ms | +0.66 vs seed42 baseline |
| T1 | `early + sigmoid` | 40 | 7 | 85.40 | 56.79 | 6,441,416 | 69.816M | 1.242 ms | Baseline seed7 40e eksik |

Yorum:

> T1 early sigmoid seed 7 sonucu `56.79`. Bu, seed 42 early sonucundan dusuk ama seed 42 baseline'in biraz ustunde. Ancak adil iki-seed T1 40e karsilastirmasi icin baseline seed 7 henuz eksik. Bu nedenle T1 icin en dogru ifade: "40 epoch seed 42'de net kazanc var; seed 7 early sonucu makul ama baseline seed7 olmadan reproducibility claim sinirli."

### 18.5 Corrected Gate Stats Aggregation Notu

Gate analysis pipeline iki kere iyilestirildi:

1. Ilk hata: latency measurement random input forward'i, validation'dan gelen son gate degerlerini ezebiliyordu.
2. Ikinci hata: validation gate stats birden fazla batch istense bile efektif olarak son batch'e yakin bilgi verebiliyordu.

Guncel pipeline:

- Gate stats validation loader uzerinde ayrica toplanir.
- `--gate-stats-batches` kadar validation batch'i kullanilir.
- JSON'a `gate_stats_num_images` yazilir.
- `gate_vectors` ve `scale_vectors`, secilen validation batch'leri uzerinden kanal bazli ortalama verir.

Bu nedenle raporda kullanilacak gate analysis icin yeni/corrected summary'ler tercih edilmeli. Eski random-input veya tek-batch etkisi tasiyan plotlar nihai yorum icin kullanilmamali.
