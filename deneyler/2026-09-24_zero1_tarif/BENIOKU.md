# 2022 kazanan tarifinin yeniden üretimi — sonuçlar (2026-09-24)

> 2026-09-25: `models_pipeline/zero1_deneyleri/` + `outputs/zero1_deneyleri/` buraya taşındı
> (bkz. `deneyler/BENIOKU.md`). Güncel tüm sonuçlar: `deneyler/PANO.md`.

## Kök neden

`end_to_end_pipeline.py` ile eğitilen modellerin hepsi RAW'ın altında kalıyordu
(en iyisi `flat_f24_k4`, global 0.378; RAW 0.461). Mimari kazananlarla aynıydı;
fark eğitim tarifindeydi:

1. **batch_size = 1.** Ekim 2022 checkpoint adları
   `arsiv/autoencoder_gpu_froom_kaggle.py` şablonudur (`model.fit(batch_size=1,
   shuffle=True)`). Aynı f32_k3 tarifinde yalnız batch değişince hızlı vekil:
   batch 8 → 0.27–0.31, batch 1 → 0.75–0.81. Eski sweep 2/8/16/32 taradı.
2. **Giriş [0, 1] (zero_1).** TPU ve k_32_3 kazananlarının Bern verisindeki BCE'si
   zero_1 girişle 0.39–0.41, minus1_1 ile 1.19–1.25. Benchmark her modele
   minus1_1 verir; kazananlar bu yüzden 2× kontrastta doygun, ikili çıktı üretir.
3. **Her epoch karıştırma.** `autoencoder_unified.py` yolları yalnız bir kez
   karıştırıyordu; eklendi (`SHUFFLE_TRAIN`).
4. **Checkpoint seçimi.** batch 1'de kalite epoch ve seed'e göre dalgalanır;
   her epoch kaydedilip 150 örnekli marj kapısıyla seçilmeli.

## Resmi benchmark (1000 sorgu, global success_25m, 4 grup ortalaması)

| Model | roi_500m | roi_2000m | global |
|---|---|---|---|
| RAW | 0.682 | 0.581 | 0.461 |
| eski pipeline en iyi (flat_f24_k4) | 0.615 | 0.513 | 0.378 |
| yeni: f32_k3, 3100 örnek, e3 | 0.744 | 0.650 | 0.479 |
| yeni: f32_k3, tam veri, e5 | 0.788 | 0.704 | **0.564** |
| yeni: güncel pipeline (TF), `z1_f32_k3_b1` | 0.803 | 0.721 | **0.569** |
| yeni: 3 modelin topluluğu | 0.793 | 0.722 | **0.615** |
| 2022 06_modeller (4000 sorgu) | 0.735 | 0.645 | 0.504 |
| 2022 deneme_6 f32_k5 (4000) | 0.789 | 0.714 | 0.601 |
| 2022 k_32_3 e1900 (4000) | 0.824 | 0.748 | 0.620 |
| 2022 TPU f48_k4 (4000) | 0.848 | 0.781 | 0.659 |

Global ayrıntı — tam veri e5: G2B 0.594 / 0.572, B2G 0.604 / 0.485;
topluluk: G2B 0.659 / 0.625, B2G 0.656 / 0.522 (clean / hard_v1).
RAW 4000 sorguda 0.477, 1000 sorguda 0.461 (sorgu kümesi farkı ≈ 0.016).

Çıktılar: `sonuclar/benchmark/urgup1000_{b1_f32k3_e3,tam_f32k3_e5,topluluk3}/` (özetler;
ham veri taşınamadı, `outputs/zero1_deneyleri/<aday>/` içinde).
Modeller: adaylar `modeller/adaylar/`, topluluklar `modeller/topluluklar/`, eğitim
koşuları (her epoch `.npz` + `keras/*.h5`) `modeller/egitim/`. Kapı ve vekil ölçümleri
`sonuclar/kapi/`, `sonuclar/vekil/`; araçlar `benchmark/deney_araclari/`.

## Marj kapısı (150 örnek, eğitim bölgesi) — seçim ölçütü

| Model | Oran (küçük iyi) |
|---|---|
| k_32_3 e1900 / TPU / deneme_6 | 0.42 / 0.43 / 0.46 |
| yeni topluluk (3 üye) | 0.47 |
| yeni tek modeller (en iyiler) | 0.49–0.51 |
| 06_modeller / RAW | 0.52 / 0.53 |

60 örnek yakın adaylar için fazla gürültülü (aynı checkpoint 0.37 / 0.51).
6×6 km hızlı vekil yeni (kenar çizen) modelleri iyimser ölçer; seçimde kullanmayın.

## Olumsuz sonuçlar

- minus1_1 + batch 1: tepe 0.52 (vekil) — zero_1'in gerisinde.
- f48_k4 (TPU mimarisi) + batch 1, 6 epoch: vekil 0.34.
- Karma k5/k3 (deneme_6 f32_k5 mimarisi), tam veri: kapı en iyi 0.52 (60 örnek).
- output19_ps (dolu yol etiketi, çok şehir): kapı en iyi 0.65.
- Giriş kazancı/ofseti kalibrasyonu (ilk katmana gömme): iyileştirme yok.
- 60 epoch (3100 örnek): iyi checkpointler çoğalıyor ama tek model 0.49'da (150 örnek) kalıyor.

## Pipeline doğrulaması

Güncellenen pipeline'ın ilk işi (`z1_f32_k3_b1`, TF eğitimi, run
`20260924_065751`) kendi kapısında en iyi 0.51 buldu (RAW 0.53) — PyTorch
koşularıyla (0.49–0.50) seed varyansı içinde uyumlu. Resmi benchmark: global
0.569 (G2B 0.597 / 0.570, B2G 0.619 / 0.488), aynı koşudaki RAW 0.461 —
`deneyler/pipeline_20260924_065751/sonuclar/`.

İki benchmark aynı anda koşturulursa işçi başına ~23 GB commit yüzünden
0xC0000005 ile çökebilir (commit sınırı 86.7 GB); benchmark'ları sırayla koşun.
