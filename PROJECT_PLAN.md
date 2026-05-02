# CMP719 FasterNet Extension Plan

Bu dosya projenin su anki durumunu, preliminary progress report icin eksik kalan isleri ve final proje icin dusunulebilecek genisletmeleri ozetler.

## 1. Projenin Ana Amaci

Secilen paper FasterNet paper'i:

> Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks

Paper'in ana fikri sudur: Bir modelin hizli olmasi icin sadece FLOPs'un dusuk olmasi yetmez. Gercek inference hizi memory access, operator tipi ve donanim kullanimi gibi faktorlerden de etkilenir.

Bizim extension fikrimiz:

> FasterNet'teki PConv ciktisindan sonra hafif bir Channel Gate Module (CGM) eklemek.

Amacimiz CGM'nin PConv tarafindan islenen kanallari input'a gore agirliklandirip agirliklandiramayacagini test etmek. Yani soru su:

> PConv sonrasi channel gating, accuracy'yi artirirken FasterNet'in dusuk overhead/hiz avantajini koruyabilir mi?

## 2. Su Ana Kadar Yapilanlar

Kod tarafinda yapilanlar:

- FasterNet-T0/T1 icin sade PyTorch implementasyonu hazirlandi.
- PConv mantigi implemente edildi.
- PConv'dan sonra calisan Channel Gate Module eklendi.
- CGM sadece PConv'un isledigi kanal subset'i uzerinde calisacak sekilde tasarlandi.
- CGM placement secenekleri eklendi:
  - `none`: baseline FasterNet
  - `all`: tum stage'lerde CGM
  - `early`: sadece erken stage'lerde CGM
  - `late`: sadece gec stage'lerde CGM
- CIFAR-100 Hugging Face uzerinden yuklenecek hale getirildi.
- Colab/Kaggle icin notebook ve komutlar hazirlandi.
- Parametre sayisi, FLOPs ve latency olcumleri eklendi.
- Gate mean degerleri JSON olarak kaydediliyor.

Deney tarafinda yapilanlar:

- FasterNet-T0, CIFAR-100 uzerinde 20 epoch egitildi.
- Baseline, CGM-All, CGM-Early ve CGM-Late karsilastirildi.

20 epoch CIFAR-100 preliminary sonuclari:

| Variant | Train Acc | Val Acc | Params | FLOPs | Latency |
|---|---:|---:|---:|---:|---:|
| FasterNet-T0, CGM-none | 59.92 | 50.52 | 2,751,160 | 27.626M | 1.008 ms |
| FasterNet-T0, CGM-all | 59.27 | 50.94 | 2,765,062 | 27.650M | 1.660 ms |
| FasterNet-T0, CGM-early | 60.71 | 51.03 | 2,751,662 | 27.632M | 1.181 ms |
| FasterNet-T0, CGM-late | 59.34 | 50.50 | 2,764,560 | 27.645M | 1.523 ms |

Ilk yorum:

- CGM-Early su an en iyi trade-off'u veriyor.
- Accuracy baseline'a gore yaklasik `+0.51%` artti.
- Parametre artisi cok kucuk: `+502`.
- FLOPs artisi cok kucuk.
- Latency artisi var ama CGM-All ve CGM-Late'e gore daha makul.
- CGM-All daha fazla latency getiriyor; bu da FasterNet paper'inin "FLOPs tek basina yeterli degildir" argumanini destekliyor.

## 3. Preliminary Progress Report Icin Eksikler

Su anki sonuc iyi bir baslangic ama preliminary report'a koymadan once tamamlanmasi gereken isler var.

### 3.1 Raporu gercek sonuclarla guncelle

`main.tex` su anda hala preliminary status kisminda sayisal sonuc olmadigini soyluyor. Bu artik dogru degil.

Yapilacaklar:

- "No final numerical training results are reported yet" cumlesi kaldirilacak.
- CIFAR-100 20 epoch tablosu eklenecek.
- CGM-Early'nin en iyi preliminary trade-off oldugu yazilacak.
- Sonuclarin henuz final kanit olmadigi, preliminary oldugu acikca belirtilecek.

### 3.2 Sonuclari dogru yorumla

Rapor sunu iddia etmemeli:

> CGM kesin olarak FasterNet'i iyilestirir.

Daha dogru iddia:

> Preliminary CIFAR-100 results suggest that selective early-stage CGM can slightly improve accuracy with negligible parameter/FLOP overhead, while all-stage gating increases latency more strongly.

Bu daha savunulabilir ve akademik olarak daha durust.

### 3.3 Gate visualization ekle

Proposal'da gate visualization vardi. Su anda gate mean degerleri kaydediliyor ama gorsel rapora eklenmedi.

Minimum yapilacak:

```bash
python scripts/plot_gate_means.py runs/fasternet_t0_early_summary.json
```

Rapor icin bir figure:

- CGM-Early layer-wise mean gate values

Bu figure sunu gostermek icin kullanilacak:

> Gate degerleri tamamen sabit degil; model bazi PConv channel gruplarini digerlerinden farkli agirliklandirmayi ogreniyor.

### 3.4 Deney komutlarini raporlanabilir hale getir

Progress report'a veya README'ye su komutlar net yazilmali:

```bash
python train_classification.py --model fasternet_t0 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement none --measure-latency
python train_classification.py --model fasternet_t0 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement early --measure-latency --save-gate-stats
```

Bu, sonuclarin nasil uretildigini gosterir.

## 4. Preliminary Icin Opsiyonel Ama Faydayli Ek Isler

Eger zaman varsa preliminary report'u guclendirmek icin asagidakilerden biri yapilabilir.

### 4.1 Bir seed tekrari

Su an sonuclar tek seed ile alindi. Accuracy farki kucuk oldugu icin seed tekrari faydali olur.

Onerilen minimum tekrar:

```bash
python train_classification.py --model fasternet_t0 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement none --seed 7 --measure-latency
python train_classification.py --model fasternet_t0 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement early --seed 7 --measure-latency --save-gate-stats
```

Amac:

- CGM-Early kazanci sadece tek seed sansi mi, yoksa tekrar ediyor mu gormek.

### 4.2 CGM reduction ratio deneyi

CGM icinde bottleneck reduction ratio su an `r=4`.

Onerilen deneyler:

```bash
python train_classification.py --model fasternet_t0 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement early --cgm-reduction 8 --measure-latency --save-gate-stats
python train_classification.py --model fasternet_t0 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement early --cgm-reduction 16 --measure-latency --save-gate-stats
```

Amac:

- Daha hafif gate ile benzer accuracy korunuyor mu?
- Latency daha da azalabilir mi?

Bu preliminary icin sart degil, final icin daha uygun olabilir.

## 5. Final Proje Icin Dusunulebilecek Genisletmeler

Final icin proje daha guclu hale getirilebilir. Hepsini yapmak sart degil; zaman ve GPU durumuna gore secilecek.

### 5.1 Daha uzun CIFAR-100 egitimi

20 epoch preliminary icin yeterli olabilir ama final icin daha uzun egitim daha guvenilir olur.

Oneri:

- 50 epoch baseline
- 50 epoch CGM-Early
- Istege bagli CGM-All

Bu, accuracy farkinin kalici olup olmadigini gosterir.

### 5.2 Multiple seed

Final icin en guclu iyilestirme budur.

Oneri:

- Seed 42
- Seed 7
- Seed 123

Varyantlar:

- Baseline
- CGM-Early

Raporlama:

| Variant | Mean Val Acc | Std | Mean Latency |
|---|---:|---:|---:|
| Baseline | ? | ? | ? |
| CGM-Early | ? | ? | ? |

Bu, kucuk accuracy farkini daha guvenilir hale getirir.

### 5.3 FasterNet-T1 deneyi

T0 en kucuk model. Finalde T1 denenirse extension'in daha buyuk modelde de ise yarayip yaramadigi gorulur.

Minimum T1 deneyleri:

```bash
python train_classification.py --model fasternet_t1 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement none --measure-latency
python train_classification.py --model fasternet_t1 --dataset cifar100 --dataset-source hf --epochs 20 --batch-size 128 --cgm-placement early --measure-latency --save-gate-stats
```

Beklenen katkisi:

- CGM-Early sadece kucuk T0'da mi iyi, yoksa T1'de de benzer trade-off var mi?

### 5.4 Tiny-ImageNet

Tiny-ImageNet, CIFAR-100'e gore daha buyuk ve daha zor bir dataset.

Avantaji:

- 64x64 input oldugu icin spatial feature extraction daha anlamli olabilir.
- Paper'in ImageNet odakli baglamina CIFAR'dan biraz daha yakindir.

Dezavantaji:

- Daha uzun surer.
- Data preparation daha zahmetlidir.
- Progress report icin sart degildir.

Final icin karar:

- Eger CIFAR-100 sonuclari cok zayif kalirsa veya daha guclu benchmark istenirse Tiny-ImageNet eklenebilir.
- Ama once CIFAR-100'da seed/longer training ile sonucu saglamlastirmak daha risksizdir.

### 5.5 Pretrained FasterNet fine-tuning

Official FasterNet ImageNet pretrained checkpoint'leri kullanilabilir.

Fikir:

- Official FasterNet-T0 pretrained weights yuklenir.
- CIFAR-100 icin classifier head degistirilir.
- CGM katmanlari random initialize edilir.
- Baseline ve CGM-Early fine-tune edilir.

Avantaji:

- Daha yuksek accuracy gelebilir.
- Official pretrained model kullanmak raporu guclendirir.

Riski:

- Official checkpoint key'leri bizim sade implementasyonla birebir uyusmayabilir.
- Weight mapping gerekebilir.
- Bu final icin opsiyonel bir genisletmedir.

### 5.6 Daha detayli gate analysis

Final icin gate visualization daha guclu hale getirilebilir.

Yapilabilecekler:

- Layer-wise mean gate plot
- Gate histogram
- Dogru siniflandirilan orneklerde gate dagilimi
- Yanlis siniflandirilan orneklerde gate dagilimi
- Early vs all gate davranisi karsilastirmasi

Bu bolum projenin yorumlanabilirlik tarafini guclendirir.

## 6. Onerilen Yol Haritasi

### Preliminary progress report icin

1. Mevcut CIFAR-100 20 epoch sonuclarini rapora ekle.
2. CGM-Early'nin en iyi preliminary trade-off oldugunu yaz.
3. FLOPs ve latency farkinin neden onemli oldugunu acikla.
4. Gate mean plot uret ve rapora koy.
5. Sonuclarin preliminary oldugunu, finalde daha uzun/multiple seed deneyleriyle dogrulanacagini belirt.

### Final proje icin

1. CIFAR-100 50 epoch baseline vs CGM-Early calistir.
2. En az 2-3 seed ile baseline vs CGM-Early tekrarla.
3. CGM reduction ratio `r=8` veya `r=16` dene.
4. Zaman kalirsa FasterNet-T1 baseline vs CGM-Early calistir.
5. Daha detayli gate visualization ekle.
6. Tiny-ImageNet veya pretrained fine-tuning'i sadece zaman/GPU yeterse ekle.

## 7. Simdiki En Mantikli Sonraki Adim

Su anda en onemli is koddan cok raporu guncellemektir.

Hemen yapilacaklar:

1. `main.tex` icindeki preliminary results kismi guncellenecek.
2. 20 epoch CIFAR-100 tablosu eklenecek.
3. "No final numerical results" ifadesi kaldirilacak.
4. Gate plot uretilecek ve rapora eklenecek.

Bu tamamlandiginda progress report teslimi icin proje daha tutarli hale gelir.
