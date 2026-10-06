# deneyler/ — modeller ve sonuçlar için tek kök

2026-09-25'ten itibaren eğitilen her model ve her ölçüm burada durur.
Tüm sonuçların tek tablodaki görünümü: **[PANO.md](PANO.md)** (ayrıntılı: `PANO.xlsx`).
Panoyu yenilemek için:

```
python sonuc_panosu.py        # benchmark/ klasöründe; deneyler/ ve outputs/ altını tarar
```

## Düzen

```
deneyler/
  PANO.md, PANO.xlsx           tüm resmi sonuçlar, kapı ölçümleri, uçuş testi (otomatik)
  secili_modeller/             "hangi modeli kullanmalıyım?" -> kısa adlı kopyalar + *.kart.json
  <tarih>_<deney>/             el ile yürütülen bir deney
    BENIOKU.md                 amaç, ayarlar, bulgular
    modeller/                  eğitim koşuları (checkpoint'ler), adaylar, topluluklar
    sonuclar/benchmark/<protokol>_<aday>/   resmi benchmark çıktısı (summary.json, benchmark_results.xlsx)
    sonuclar/kapi/*.json       margin_gate.py ölçümleri
    kayitlar/                  günlükler ve başlatma betikleri
  pipeline_<run_id>/           end_to_end_pipeline.py koşusu (aynı düzen)
    modeller/<iş>/             checkpoint'ler, _epochs/, _benchmark_adayi/
    sonuclar/                  işlerin benchmark çıktıları, _margin_gate/, TUM_SONUCLAR.xlsx
    kayitlar/
```

Kodlar burada değil: deney araçları `benchmark/deney_araclari/`, pipeline
`benchmark/end_to_end_pipeline.py`, pano `benchmark/sonuc_panosu.py`.

## Deneyler

| Klasör | Ne |
|---|---|
| `2026-09-24_zero1_tarif/` | 2022 kazananlarının eğitim tarifinin (batch 1, [0,1] giriş, her epoch karıştırma) bulunup yeniden üretilmesi |
| `pipeline_20260924_065751/` | Güncellenen pipeline'ın doğrulama koşusu (`z1_f32_k3_b1`, TF) |
| `2026-09-25_genis_dogrulama/` | Başarılı modellerin 2022 kazananlarıyla aynı 4000 sorguda doğrulanması |
| `2026-09-25_parametre_arama/` | Başarıyı artırma denemeleri (EMA/SWA, lr takvimi, augmentasyon, kapasite, TTA) |
| `2026-10-02_ucus_uyarlama/` | Gerçek İHA kareleriyle kare modeli uyarlaması (harita = x1 sabit, öğrenci x1'den başlar); Ürgüp'te eğit, Karlık'ta test |
| `2026-10-06_tm_iyilestirme/` | TM uçuş hattı iyileştirmelerinin (GSD interpolasyonu, çok adaylı kaba arama) A/B testi; önceden yazılmış karar kuralı ve geri dönüş planı |

## Eski düzen (taşınmadı, pano yerinde okur)

- `models_pipeline/<run_id>/` + `outputs/end_to_end_pipeline_<run_id>/` — 2026-09-24 öncesi pipeline koşuları
  (git'te izlenen dosyalar; `--run-id` ile devam ettirilirse yine bu yollar kullanılır).
- `outputs/urgup_cevresi_models_4000/` — 2022 modellerinin ve RAW'ın 4000 sorguluk ölçümü (6.5 GB).
- `outputs/zero1_deneyleri/<aday>/` — 2026-09-24 resmi koşularının ham verisi (sorgu karoları, results.jsonl);
  taşıma sırasında klasörler başka bir süreç tarafından kilitliydi. Özetleri
  `2026-09-24_zero1_tarif/sonuclar/benchmark/` altına kopyalandı.
- `outputs/karlik_top_modeller/` — Karlık gerçek İHA uçuş testi (116 kare).
- `models/` — `run_urgup_cevresi_models.py`'nin varsayılan model klasörü (2022 modelleri).

## Git

Ağır ve üretilen dosyalar (`*.npz` checkpoint'ler, sorgu karoları, `results.jsonl`, `*.xlsx`,
`*.tif`) `.gitignore` ile dışarıda; `*.h5` modeller, `summary.json`, kartlar ve `BENIOKU.md`
dosyaları izlenebilir.
