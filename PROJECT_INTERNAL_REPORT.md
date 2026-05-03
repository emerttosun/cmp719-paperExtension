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
