# Jeoreferanslı Ortak-Temsil Benchmarkı

Aynı sinir ağını hem sorgu parçasına hem arama haritasına uygulayıp, metre cinsinden konum başarısını ölçer:

```text
model(query 544×544)  → 16 px kenar kırpma → 512×512 şablon
model(tam arama rasterı) → örtüşmeli mozaik → jeoreferanslı GeoTIFF
şablon → çok adaylı kaba–ince TM_CCOEFF_NORMED → UTM hata (m)
```

Ham eşleştirme `RAW_BASELINE` olarak ayrı tutulur. Bu klasör harici projeye bağlı değildir; `goruntu_islemleri.py` burada kopyalıdır.

## İçindekiler

- [Klasör yapısı](#klasör-yapısı)
- [Ortam](#ortam)
- [Veri ve modeller](#veri-ve-modeller)
- [Ne ölçülür](#ne-ölçülür)
- [Bilimsel tutarlılık](#bilimsel-tutarlılık)
- [Çalıştırma](#çalıştırma)
- [Devam ettirme](#devam-ettirme)
- [Çıktılar](#çıktılar)
- [Metrikler](#metrikler)
- [Önemli CLI bayrakları](#önemli-cli-bayrakları)
- [Testler](#testler)
- [Yorum sınırı](#yorum-sınırı)

## Klasör yapısı

```text
benchmark/
├── geospatial_model_benchmark.py   # ana CLI
├── goruntu_islemleri.py            # karo / inference / mozaik / jeoreferans
├── build_benchmark_excel_openpyxl.py
├── environment.yml
├── DATA.md                         # raster gereksinimleri
├── PROVENANCE.md                   # kod/model kökeni
├── models/                         # .h5 / .keras (Git dışı)
├── tests/
└── outputs/<run_id>/               # koşu çıktıları
```

Varsayılan rasterlar (Git dışı): `urgup_30_cm_yeni_gmaps_utm.tif`, `urgup_bingmap_utm_30_cm.tif`.

## Ortam

```powershell
conda env create -f environment.yml
conda activate visual_navigation_cuda
cd C:\d_surucusu\visual_navigation\benchmark
```

PowerShell’de ortamı etkinleştirip doğrudan `python` kullanın. `conda run` bazı Windows kod sayfalarında Türkçe loglarda kodlama hatası verebilir.

Excel yedek motoru için (ortamda yoksa):

```powershell
python -m pip install openpyxl==3.1.5
```

## Veri ve modeller

| Girdi | Konum / bayrak | Not |
|---|---|---|
| Sorgu raster | `--query-raster` (varsayılan Ürgüp Google) | UTM, metre GSD |
| Arama raster | `--map-raster` (varsayılan Ürgüp Bing) | Aynı CRS / uyumlu çözünürlük |
| Modeller | `models/` veya `--model-dir` | `.h5` / `.keras` / `.hdf5` |

Ayrıntı: [DATA.md](DATA.md), [models/README.md](models/README.md), [PROVENANCE.md](PROVENANCE.md).

## Ne ölçülür

### Sorgu varyantları (varsayılan: `clean,hard_v1`)

| Varyant | Açıklama |
|---|---|
| `clean` | Özgün ortomozaik parçası |
| `hard_v1` | Geometriyi değiştirmeden deterministik İHA kamera bozulmaları |

`hard_v1` senaryo dağılımı: %20 temiz-hafif, %25 pus, %20 hareket bulanıklığı, %15 odak bulanıklığı, %10 düşük kontrast, %10 güçlü sıkıştırma/gürültü. Pozlama, gamma, beyaz dengesi, doygunluk, sensör gürültüsü ve JPEG kalitesi kontrollü değişir; düşük olasılıkla vinyet uygulanır. Parametreler `query_variant_manifest.json` içinde saklanır.

Rotasyon / ölçek / perspektif `hard_v1` içinde yoktur; bunlar NCC’nin geometrik toleransını ölçer, temsil benchmark’ından ayrı tutulur.

### Arama modları (bağımsız; progressive fallback değil)

Sıra: `roi_500m`, `roi_1000m`, `roi_2000m`, `roi_4000m`, `roi_8000m`, `global`.

Örnek: `--search-modes global` veya `--search-modes roi500,roi2000,global`.

### Yönler

Varsayılan `--bidirectional`: Google→Bing, sonra Bing→Google. Tek yön için `--no-bidirectional`.

## Bilimsel tutarlılık

- Sorgular ortak UTM alanından seçilir; 1 km bloklara ayrılır; sabit `--seed` ile bloklu örnekleme.
- Tüm modeller aynı `query_manifest.json` sorgularını kullanır; `clean` / `hard_v1` aynı merkezleri paylaşır.
- Varsayılan: blok başına en fazla 5 sorgu, yön başına 300 sorgu.
- Her model iki tarafta da aynı kanal, normalizasyon, karo ve kırpma ayarını kullanır.
- Top-1 yanında Top-2, peak margin ve PSR kaydedilir; hata GeoTIFF transformundan UTM metreye çevrilir.
- ROI ve global sonuçlar birleştirilmez; ayrı özetlenir.
- Her sorgu sonrası `results.jsonl` checkpoint; her model sonunda özet + Excel güncellenir.
- Kenar tamponu: varsayılan bir tam sorgu karosu (~162 m @ 30 cm GSD); merkezler sınırdan ~244 m içeride. Özel değer: `--query-edge-buffer-m 300`.

## Çalıştırma

### 1. RAW smoke testi

```powershell
python geospatial_model_benchmark.py `
  --no-include-models `
  --max-queries 10 `
  --run-id raw_smoke
```

### 2. Tek model pilotu

```powershell
python geospatial_model_benchmark.py `
  --models "13_05_2023__14_07_model_step_300.h5" `
  --max-queries 25 `
  --run-id pilot_tek_model
```

### 3. Üç model pilotu

```powershell
python geospatial_model_benchmark.py `
  --models "13_05_2023__14_07_model_step_300.h5" `
  --models "GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5" `
  --models "GPU_model_f48_k5_epoch_00500_sigmoid_(2, 2)_.h5" `
  --max-queries 100 `
  --run-id pilot_uc_model
```

### 4. Tüm modeller, 300 sorgu

`--models` yoksa `models/` altındaki tüm uygun dosyalar işlenir.

```powershell
python geospatial_model_benchmark.py `
  --max-queries 300 `
  --samples-per-block 5 `
  --run-id urgup_tum_modeller_300
```

`clean,hard_v1` sorgu yükünü ~2× artırır; model haritası yön×model başına bir kez üretilir. İki yönlü çalışma da maliyeti ~2× yapar.

Arama varsayılanı `--search-workers 8` (güvenli aralık 1–8). İşçiler harita ve piramitleri salt-okunur paylaşır; checkpoint tek koordinatörde yazılır. Seri referans için `--search-workers 1`. Ürgüp’te 6 mod / 48 görev ölçümünde ~2.5× hızlanma görülmüş; konum/NCC/başarı alanları seri ile birebir aynı kalmıştır.

## Devam ettirme

```powershell
python geospatial_model_benchmark.py `
  --resume-run outputs\urgup_tum_modeller_300 `
  --max-queries 300 `
  --samples-per-block 5
```

Tamamlanmış `direction + query_variant + search_mode + model + query_id` kayıtları atlanır. Bitmiş yön/modelde inference ve harita yükleme de atlanır. `resume_signature` bilimsel ayarları kilitler; `--search-workers` imzada değildir.

Farklı `hard_v1` profiliyle üretilmiş bir klasöre devam edilirse benchmark karıştırmamak için yeni `--run-id` ister.

## Çıktılar

Her koşu `outputs/<run_id>/` altındadır:

```text
benchmark.log
run_config.json
results.jsonl / results.csv
summary.json / summary.csv
model_errors.jsonl
benchmark_results.xlsx
excel_validation.json
excel_previews\*.png          # yalnız Artifact motoru
<direction>\queries\          # manifest + ham sorgular
<direction>\models\           # model GeoTIFF + sorgu çıktıları
```

Excel sayfaları: `Özet`, `Model Özeti`, `Ham Sonuçlar`, `Sorgu Manifesti`, `Yapılandırma`, `Hatalar`.

Model sonu checkpointlerinde Excel baştan kurulmaz: `results.jsonl` dosyasının daha önce raporlanmış öneki SHA-256 ile doğrulanır, yalnızca yeni `Ham Sonuçlar` satırları eklenir ve küçük özet/pano sayfaları yenilenir. Eski raporun veri veya şeması kaynakla uyuşmazsa bilimsel içeriği riske atmamak için otomatik olarak tam üretime dönülür. Benchmark finalinde tam üretim ve derin doğrulama korunur.

| Bayrak | Varsayılan | Anlamı |
|---|---|---|
| `--excel-engine` | `auto` | Önce Artifact, yoksa openpyxl |
| `--excel-update` | `model` | Her model sonunda yenile; yalnız sonda için `end` |
| `--strict-excel` | açık | Excel üretilemezse koşuyu başarısız say |

`benchmark_results.xlsx` Excel'de açık/kilitliyse dosya zorlanmaz ve benchmark durmaz. Güncel rapor aynı klasöre `benchmark_results_YYYYMMDD_HHMMSS_kilitli_kopya.xlsx` adıyla yazılır; gerçek son dosya `benchmark.log` ve `excel_latest.json` içinde belirtilir. Sonraki model ana dosyayı yeniden denerken en güncel kopyayı artımlı taban olarak kullanır.

## Metrikler

Ana ölçüt: **`success_30m`** — hata ≤ 30 m olan sorguların tüm sorgulara oranı (red/hata = başarısız).

Sıralama: `success_30m` → `AUC@30m` → başarılıların medyan hatası (`median_error_under_30m`).

| Metrik | Rol |
|---|---|
| `AUC@30m` | 0–30 m başarı CDF alanı (normalize); >30 m katkı yok |
| `mean/median_error_under_30m` | Operasyonel başarıların hassasiyeti |
| `success_5/10/25/50m` | İkincil, geriye dönük karşılaştırma |
| ortalama / medyan / P90 / P95 hata | Tanısal; ana sıralamada değil |

Sorgu başına ayrıca: UTM/piksel hata, Top-1/Top-2 NCC, peak margin, PSR, süreler, doku istatistikleri. `%95` CI için mekânsal blok bootstrap (varsayılan 1000; `--bootstrap-iterations`).

## Önemli CLI bayrakları

```text
--query-raster / --map-raster / --model-dir / --models
--run-id / --resume-run / --output-root
--bidirectional / --no-bidirectional
--include-raw / --no-include-raw
--include-models / --no-include-models
--query-variants clean,hard_v1
--search-modes roi500,roi1000,...,global
--max-queries / --samples-per-block / --block-size-m / --seed
--tile-size 544 / --overlap 32 / --crop-border 16
--search-workers 8 / --batch-size / --pyramid-factors
--normalization minus1_1 / --enhancement none
--excel-engine auto / --excel-update model / --strict-excel
--force-queries / --force-maps / --keep-intermediate / --fail-fast / --verbose
```

Tam liste ve varsayılanlar:

```powershell
python geospatial_model_benchmark.py --help
```

## Testler

```powershell
python -m pytest tests -q
```

## Yorum sınırı

Modeller Ürgüp/Kapadokya verisiyle ilişkili olabilir. Eğitim alanı çakışması kesin dışlanmadıkça sonuçlar **in-domain** benchmark olarak raporlanmalı; dış bölge genellemesi iddia edilmemelidir.
