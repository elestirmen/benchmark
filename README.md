# Jeoreferanslı Ortak-Temsil Benchmarkı

Aynı sinir ağını hem sorgu parçasına hem arama haritasına uygulayıp, metre cinsinden konum başarısını ölçer:

```text
model(query 544×544)  → 16 px kenar kırpma → 512×512 şablon
model(tam arama rasterı; rasterio window/batch streaming) → doğrudan 1-band jeoreferanslı GeoTIFF
şablon → çok adaylı kaba–ince TM_CCOEFF_NORMED → UTM hata (m)
```

Ham eşleştirme `RAW_BASELINE` olarak ayrı tutulur. Model benchmarkları GPU zorunlu çalışır; TensorFlow GPU görmüyorsa CPU'ya sessiz geçiş yapılmaz. Bu klasör harici projeye bağlı değildir; `goruntu_islemleri.py` burada kopyalıdır.

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
├── goruntu_islemleri.py            # ortak model runtime / inference / legacy yardımcıları
├── build_benchmark_excel_openpyxl.py
├── environment.yml                 # Windows / Conda / TensorFlow 2.10 CUDA
├── requirements_wsl.txt            # WSL2 / güncel TensorFlow CUDA
├── gpu_test.py                     # TensorFlow GPU erişim kontrolü
├── DATA.md                         # raster gereksinimleri
├── PROVENANCE.md                   # kod/model kökeni
├── models/                         # .h5 / .keras / .hdf5 (Git dışı)
├── tests/
└── outputs/<run_id>/               # koşu çıktıları
```

Varsayılan rasterlar (Git dışı): `urgup_30_cm_yeni_gmaps_utm.tif`, `urgup_bingmap_utm_30_cm.tif`.

## Ortam

### Windows + Conda

```powershell
conda env create -f environment.yml
conda activate visual_navigation_cuda
cd C:\d_surucusu\visual_navigation\benchmark
python gpu_test.py
```

`environment.yml`, native Windows CUDA desteği için Python 3.10, TensorFlow 2.10.1, CUDA Toolkit 11.2, cuDNN 8.1 ve `.h5` yükleme için h5py 3.10 kurar. Var olan ortamı güncellemek için:

```powershell
conda env update -n visual_navigation_cuda -f environment.yml --prune
```

PowerShell'de ortamı etkinleştirip doğrudan `python` kullanın. `conda run` bazı Windows kod sayfalarında Türkçe loglarda kodlama hatası verebilir.

### WSL2 alternatifi

```bash
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_wsl.txt
python gpu_test.py
```

Model içeren bir koşudan önce `gpu_test.py` çıktısında en az bir GPU görünmelidir. Benchmark, model haritası ve model sorgusu çıkarımlarında GPU'yu zorunlu kılar; GPU yoksa ilgili model `model_errors.jsonl` dosyasına hata olarak yazılır. Yalnız `RAW_BASELINE` çalıştıran `--no-include-models` koşusu model çıkarımı yapmaz.

RTX 50 serisinde native Windows TensorFlow 2.10 süreci her başlatıldığında CUDA bağlamı ve TensorFlow çalışma grafiği yeniden kurulur. Compute Capability 12.0 için eksik hazır kernel'ler ilk model yüklemesi/ilk batch sırasında PTX'ten JIT derlenebilir; bu maliyet yeni Python sürecinde tekrar görülebilir. Derlenen sürücü kernel'leri varsayılan olarak `outputs/cuda_cache/` altında, en çok 4 GiB olacak şekilde süreçler arasında saklanır; mevcut `CUDA_CACHE_PATH` ve `CUDA_CACHE_MAXSIZE` değerleri korunur. Disk cache JIT yükünü azaltır fakat süreç içi CUDA bağlamını, TensorFlow graph hazırlığını ve model-özel autotuning'i ortadan kaldırmaz. Benchmark logu artık model yükleme ile ilk GPU batch süresini ayrı gösterir; uzun koşuyu yeniden başlatmak yerine aynı süreçte sürdürmek bu başlangıç maliyetini yalnız bir kez öder.

Excel yedek motoru olan openpyxl her iki kurulum dosyasında da bulunur. Gerekirse tek başına şu komutla kurulabilir:

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
- Bilimsel semantik v2'de manifest merkezi, seçilen sürekli rastgele nokta değil gerçekten çıkarılan çift-boyutlu karonun geometrik merkezidir. 544→512 simetrik kırpma merkezi değiştirmez.
- Varsayılan: blok başına en fazla 5 sorgu, yön başına 300 sorgu.
- Her model iki tarafta da aynı kanal, normalizasyon, karo ve kırpma ayarını kullanır.
- Query/map rasterları aynı CRS ve GSD'ye, kuzey-yukarı rotasyonsuz affine dönüşüme ve desteklenen `uint8` bantlara sahip olmalıdır; benchmark sessiz reprojection, resampling veya dtype truncation yapmaz.
- Top-1 yanında Top-2, peak margin ve PSR kaydedilir; hata GeoTIFF transformundan UTM metreye çevrilir.
- ROI ve global sonuçlar birleştirilmez; ayrı özetlenir.
- Sonuçlar tek doğruluk kaynağı olan `results.jsonl` dosyasına 100 satır veya en geç 2 saniyelik paketlerle yazılır; her model sonunda tampon kesin olarak boşaltılır ve özet + hafif Excel güncellenir.
- Model sonu özetleri `.summary_state/` altındaki silinebilir artımlı cache ile yalnız yeni JSONL byte aralığından güncellenir. State eksik/bozuk/uyumsuzsa JSONL'den baştan kurulur; normal finalde tam JSONL özetiyle birebir karşılaştırılır.
- Kenar tamponu: varsayılan bir tam sorgu karosu (~162 m @ 30 cm GSD); merkezler sınırdan ~244 m içeride. Özel değer: `--query-edge-buffer-m 300`.

## Çalıştırma

Model içeren 2–4 numaralı örneklerden önce `python gpu_test.py` ile GPU erişimini doğrulayın. Varsayılan koşu hem RAW hem model kanallarını çalıştırır.

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

### 4. Eğitim serisi başına beş checkpoint ile hızlı eleme

`--model-sampling five-point`, aynı klasördeki dosya adlarını `epoch/step` numarasını yok sayarak eğitim serilerine ayırır. Her seri checkpoint numarasına göre sıralanır ve ilk, %25, orta, %75 ve son model seçilir. Böylece aynı fiziksel klasörde bulunan farklı mimariler birbirine karışmaz. Seçim noktası, seri kimliği ve checkpoint numarası `model_catalog.json` içinde saklanır.

Mevcut `models/diger` ağacında 2.273 dosya 126 eğitim serisine ayrılır; beş nokta seçimi 556 checkpoint üretir ve aynı içerikli 11 seçim elendikten sonra 545 benzersiz model çalıştırılır.

Önerilen ilk eleme koşusu 100 deterministik sorgu, yalnız global arama, clean + hard varyantları ve çift yön kullanır:

```powershell
python geospatial_model_benchmark.py `
  --model-dir "models\diger" `
  --model-sampling five-point `
  --max-queries 100 `
  --search-modes global `
  --cleanup-maps `
  --batch-size 64 `
  --run-id urgup_bes_nokta_global_100
```

İlk eleme bittikten sonra global `success_25m`, `AUC@25m` ve `median_error_under_25m` sıralamasındaki finalistler tam protokolle yeniden çalıştırılabilir.

### 5. Tüm modeller, 300 sorgu

Varsayılan `--model-sampling full` davranışında `--models` yoksa `models/` alt klasörleri dahil tüm uygun dosyalar işlenir. Başlangıçta bütün adaylar SHA-256 ile kataloglanır: birebir aynı içerikli kopyalar yalnız bir kez çalıştırılır; aynı dosya adına sahip fakat içeriği farklı modeller göreli klasör yolu ve kısa SHA içeren ayrı `model_id` değerleri alır. Ayrıntılı liste `model_catalog.json`, sayımlar ve katalog özeti `run_config.json` ile başlangıç loguna yazılır. `--max-models`, örnekleme ve SHA tekilleştirmesinden sonraki gerçek model sayısına uygulanır.

```powershell
python geospatial_model_benchmark.py `
  --max-queries 300 `
  --samples-per-block 5 `
  --run-id urgup_tum_modeller_300
```

`clean,hard_v1` sorgu yükünü ~2× artırır; model yön başına bir kez yüklenir ve aynı runtime map, clean ve hard_v1 inference boyunca kullanılır. Aynı `(query_id, ROI modu)` harita piramidi clean/hard_v1 arasında paylaşılır. İki yönlü çalışma maliyeti yaklaşık 2× yapar.

Model haritası varsayılan olarak GeoTIFF pencerelerini doğrudan batch RAM'e okuyup final GeoTIFF'e yazar; binlerce ara map PNG'si oluşturmaz. İnceleme için `--keep-intermediate` verilirse kaynak/prediction karoları ve debug mozaiği ayrıca saklanır.

Arama varsayılanı `--search-workers 8` (güvenli aralık 1–8). İşçiler harita ve piramitleri salt-okunur paylaşır; checkpoint tek koordinatörde yazılır. Seri referans için `--search-workers 1`. Ürgüp’te 6 mod / 48 görev ölçümünde ~2.5× hızlanma görülmüş; konum/NCC/başarı alanları seri ile birebir aynı kalmıştır.

Her model denemesinden sonra logda `MODEL TARAMA SÜRESİ` satırı yazılır. Bu satır gerçek model süresini, son beş başarıyla tamamlanan modelin medyanını ve mevcut yön için kalan yaklaşık süreyi gösterir. Tahmin yön bazlıdır; ilk modelin CUDA/PTX soğuk başlangıcı ve farklı model mimarileri nedeniyle özellikle ilk birkaç modelde değişebilir. Model-sonu Excel süresi ayrıca `ARA EXCEL TAMAMLANDI` satırında raporlanır.

Bir model yüklenemezse veya GPU çıkarımı başarısız olursa hata `model_errors.jsonl` içine kaydedilir ve varsayılan olarak sonraki modele geçilir. İlk model hatasında koşuyu durdurmak için `--fail-fast` kullanın. Model başarıyla yüklenmeden boş model çıktı klasörleri oluşturulmaz.

## Devam ettirme

```powershell
python geospatial_model_benchmark.py `
  --resume-run outputs\urgup_tum_modeller_300 `
  --max-queries 300 `
  --samples-per-block 5
```

Tamamlanmış `direction + query_variant + search_mode + model + query_id` kayıtları atlanır. Bitmiş yön/modelde inference ve harita yükleme de atlanır. `resume_signature` bilimsel ayarları, model kataloğunun göreli yol/SHA özetini ve `SCIENTIFIC_SEMANTICS_VERSION=2` merkez konvansiyonunu kilitler; `--search-workers` imzada değildir.

`--batch-size` bilimsel değil operasyonel bir resume ayarıdır; VRAM durumuna göre devam sırasında değiştirilebilir. İlk ve sonraki batch değerleri `run_config.json` içindeki `operational_history` alanında korunur. Batch değişikliği yalnız henüz tamamlanmamış model çıkarımlarına uygulanır; bitmiş model sonuçları checkpointten yeniden kullanılmaya devam eder.

Semantik v1 veya sürümsüz eski bir sonuç klasörü v2 koduyla resume edilmez; eski/yeni merkez gerçeklerinin karışmaması için açık hata verilir ve yeni `--run-id` gerekir.

Farklı `hard_v1` profiliyle üretilmiş bir klasöre devam edilirse benchmark karıştırmamak için yeni `--run-id` ister.

## Başarılı lineage'lar için 2000 sorguluk epoch sweep

`epoch_sweep_2000.py`, tamamlanmış bir benchmark koşusundaki global clean +
hard_v1 sonuçlarından ilk 10 checkpointi seçer. İki yön tamamsa ikisini eşit
ağırlıkla kullanır; sıralama `success_25m` → `AUC@25m` → `success_5m` → düşük
`median_error_under_25m` şeklindedir. İlk 10 seçildikten sonra training lineage
tekilleştirilir; aynı lineage birden fazla kez seçilmişse sıra 11'den backfill
yapılmaz.

Önce yalnız planı görmek için:

```powershell
python epoch_sweep_2000.py `
  --benchmark-run outputs\<tamamlanmis_run> `
  --model-archive-dir "D:\...\tum_modeller" `
  --dry-run
```

Gerçek çalışma aynı komuttan `--dry-run` çıkarılarak başlatılır:

```powershell
python epoch_sweep_2000.py `
  --benchmark-run outputs\<tamamlanmis_run> `
  --model-archive-dir "D:\...\tum_modeller"
```

Script yalnız `epoch` checkpointlerini recursive tarar; `step_*` ve `batch_*`
dosyalarını dışarıda bırakır. SHA-256 ankrajıyla kaynak training lineage'ı bulur,
aynı adlı fakat farklı koşuların kaynak klasör kimliğini korur ve checkpointleri
`models_epoch_sweep/` altında doğrulanmış manifestlerle toplar. Aynı SHA bir kez
kopyalanıp/çalıştırılır; mantıksal epoch aliasları raporda korunur.

Her yön için sabit seed ile tam 2000 merkez `balanced_exact` blok kotasıyla bir
kez üretilir. Üst düzey `query_manifest_2000.json`, yön bazlı gerçek manifestleri
ve bilimsel SHA-256 parmak izlerini indeksler. Manifest, model inventory veya
bilimsel ayarlar sonuç başladıktan sonra değişirse resume açık hatayla reddedilir.
Core benchmark; RAW, GPU zorunluluğu, `minus1_1`, 544→512 kırpma,
`TM_CCOEFF_NORMED`, UTM hata ve per-query resume semantiğini aynen yürütür.
Ana benchmarktaki model-sınırı raporlama stratejisi de korunur: her model
tamamlandığında artımlı özet yenilenir ve `benchmark_results.xlsx` güncellenir.
Excel dosyası açık/kilitliyse benchmark durmaz; zaman damgalı kilitli kopya
üretilir ve güncel dosya `excel_latest.json` içinde belirtilir.

Ana çıktılar `outputs/epoch_sweep_2000/` altındadır:

```text
selected_top10_models.csv
selected_lineages.csv
query_manifest_2000.json
benchmark_results.xlsx
excel_latest.json
all_epoch_results.csv
best_epoch_per_lineage.csv
raw_baseline_results.csv
epoch_sweep_errors.jsonl
plots\<lineage>_success25.png
```

## Çıktılar

Her koşu `outputs/<run_id>/` altındadır:

```text
benchmark.log
run_config.json
model_catalog.json               # göreli yol + SHA + benzersiz model_id kataloğu
results.jsonl                    # çalışma sırasında paketli checkpoint
results.csv                      # yalnız normal final dışa aktarımı
summary.json / summary.csv
summary_metadata.json
.summary_state\                   # silinebilir artımlı özet cache'i
model_errors.jsonl              # yalnız model hatası oluşursa
benchmark_results.xlsx
excel_validation.json
excel_latest.json               # kilitli Excel kopyası kullanılırsa
excel_previews\*.png          # yalnız Artifact motoru
<direction>\queries\          # manifest + ham sorgular
<direction>\models\           # model GeoTIFF + sorgu çıktıları
```

Paylaşılan CUDA derleme önbelleği koşu klasörünün dışında `outputs/cuda_cache/` altında tutulur.

Model sonu hafif Excel sayfaları: Özet, Model Özeti, Model Kataloğu, Yapılandırma, Hatalar. Model Özeti insan-okur 13 görünür alanlı rapor görünümüdür; model_id artık doğrudan bu sayfada görünür, kanonik model ID, göreli yol ve SHA256 Model Kataloğu sayfasında da korunur. Final Excel bunlara Ham Sonuçlar ve Sorgu Manifesti sayfalarını ekler.

Model sonu checkpointlerinde büyük `results.csv` yeniden yazılmaz ve ham sonuçlar XLSX içine alınmaz. Yalnız tamamlanan sonuçlardan üretilen özet, global sıralama, yapılandırma ve model hataları küçük bir ara rapora yazılır. Benchmark normal tamamlandığında `results.csv` bir kez üretilir; ham sonuçları ve sorgu manifestini içeren tam Excel baştan oluşturulup derin doğrulanır.

| Bayrak | Varsayılan | Anlamı |
|---|---|---|
| `--excel-engine` | `auto` | Önce Artifact, yoksa openpyxl |
| `--excel-update` | `model` | Her model sonunda yenile; yalnız sonda için `end` |
| `--strict-excel` | açık | Excel üretilemezse koşuyu başarısız say |

`benchmark_results.xlsx` Excel'de açık/kilitliyse dosya zorlanmaz ve benchmark durmaz. Güncel rapor aynı klasöre `benchmark_results_YYYYMMDD_HHMMSS_kilitli_kopya.xlsx` adıyla yazılır; gerçek son dosya `benchmark.log` ve `excel_latest.json` içinde belirtilir. Sonraki model ana dosyayı yeniden dener.

## Metrikler

Ana ölçüt: **`success_25m`** — hata ≤ 25 m olan sorguların tüm sorgulara oranı (red/hata = başarısız).

Özet sayfası yalnız `search_mode=global` sonuçlarını gösterir. Sıralama: `success_25m` → `AUC@25m` → başarılıların medyan hatası (`median_error_under_25m`). ROI sonuçları `Model Özeti` sayfasında ayrı tutulur.

| Metrik | Rol |
|---|---|
| `AUC@25m` | 0–25 m başarı CDF alanı (normalize); >25 m katkı yok |
| `mean/median_error_under_25m` | Operasyonel başarıların hassasiyeti |
| `success_5/10/50m` | İkincil, geriye dönük karşılaştırma |
| ortalama / medyan / P90 / P95 hata | Tanısal; ana sıralamada değil |

Sorgu başına ayrıca: UTM/piksel hata, Top-1/Top-2 NCC, peak margin, PSR, süreler, doku istatistikleri. `%95` CI için mekânsal blok bootstrap (varsayılan 1000; `--bootstrap-iterations`).

## Önemli CLI bayrakları

```text
--query-raster / --map-raster / --model-dir / --models / --model-sampling
--max-models / --run-id / --resume-run / --output-root
--bidirectional / --no-bidirectional
--include-raw / --no-include-raw
--include-models / --no-include-models
--query-variants clean,hard_v1
--search-modes roi500,roi1000,...,global
--max-queries / --samples-per-block / --block-size-m / --query-sampling / --seed
--min-query-std / --min-query-entropy / --max-dark-fraction / --query-edge-buffer-m
--tile-size 544 / --overlap 32 / --crop-border 16
--search-workers 8 / --batch-size / --pyramid-factors
--normalization minus1_1 / --enhancement none / --output-value-mode auto
--excel-engine auto / --excel-update model / --excel-report / --results-csv
--strict-excel / --no-strict-excel
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
