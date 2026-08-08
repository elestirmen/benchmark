# Jeoreferanslı Ortak-Temsil Benchmarkı

Bu benchmark, aynı sinir ağını hem anlık sorgu parçasına hem de büyük arama
haritasına uygular:

```text
model(anlık 544x544 parça) -> 16 px kenar kırpma -> 512x512 şablon
model(tam arama rasterı)   -> örtüşmeli mozaik -> jeoreferanslı GeoTIFF
```

Ardından model sorgusu, aynı modelin ürettiği haritada çok adaylı kaba-ince
`TM_CCOEFF_NORMED` ile aranır. Ham görüntü eşleştirmesi `RAW_BASELINE` olarak
ayrı tutulur.

Varsayılan olarak aynı sorgu merkezleri iki ayrı koşulda ölçülür:

- `clean`: özgün, temiz ortomozaik parçası;
- `hard_v1`: geometriyi değiştirmeden deterministik fakat olasılıksal bir İHA
  kamera profili uygulanmış parça. Her sorgu için ana senaryo ayrı seçilir:
  `%20` temiz-hafif, `%25` pus, `%20` hareket bulanıklığı, `%15` odak
  bulanıklığı, `%10` düşük kontrast ve `%10` güçlü sıkıştırma/gürültü.

Ana senaryolar birbirini gereksiz yere yığmaz. Pozlama, gamma, beyaz dengesi,
doygunluk, sensör gürültüsü ve JPEG kalitesi her görüntüde kontrollü olarak
değişir; düşük olasılıkla vinyet uygulanır. Seçilen senaryo ve bütün rastgele
parametreler `query_variant_manifest.json` içinde kaydedilir.

Rotasyon, ölçek ve perspektif `hard_v1` içinde kullanılmaz; bunlar sinir ağı
temsilinden çok NCC eşleştiricisinin geometrik toleransını ölçebileceği için ayrı
bir geometrik deney konusu olarak bırakılmıştır. Yalnız temiz benchmark için
`--query-variants clean`; yalnız zorlaştırılmış kanal için
`--query-variants hard_v1` kullanılabilir.

Varsayılan olarak altı bağımsız benchmark senaryosu şu sırayla raporlanır:
`roi_500m`, `roi_1000m`, `roi_2000m`, `roi_4000m`, `roi_8000m` ve `global`.
Bunlar progressive fallback değildir; her mod aynı sorgular üzerinde bağımsız
olarak ölçülür. Yalnız global arama için `--search-modes global`; farklı bir
liste için örneğin `--search-modes roi500,roi2000,global` kullanılabilir.

## Bilimsel tutarlılık

- Sorgular raster sırasına göre değil, iki rasterın ortak UTM alanından seçilir.
- Ortak alan 1 km bloklara ayrılır; sabit seed ile bloklu örnekleme yapılır.
- Bütün modeller aynı `query_manifest.json` sorgularını kullanır.
- `clean` ve `hard_v1` aynı sorgu merkezlerini kullanır ve Excel'de ayrı
  `query_variant` satırları olarak raporlanır.
- Varsayılan örnekleme her 1 km bloktan en fazla 5 sorgu alır ve yön başına
  toplam 300 sorguda durur.
- Her model iki tarafta da aynı kanal, normalizasyon, karo ve kırpma ayarını kullanır.
- Birinci aday yanında ikinci bağımsız aday, peak margin ve PSR kaydedilir.
- Gerçek konum ve tahmin GeoTIFF transformundan UTM metreye çevrilir.
- Global ve ROI sonuçları aynı başarı sayısında birleştirilmez; ayrı özetlenir.
- Her sorgu tamamlanınca `results.jsonl` checkpointi diske yazılır.
- Her model denemesi tamamlanınca özet tablolar ve Excel çalışma kitabı güncellenir.
- Kesilen bir koşu aynı klasörden devam ettirilebilir.
- Sorgu merkezleri ortak alanın dış kenarlarından alınmaz. Varsayılan ek güvenlik
  tamponu bir tam sorgu karosu genişliğidir; 544 piksel ve yaklaşık 30 cm GSD için
  yaklaşık 162,4 metredir. Parçanın yarı genişliğiyle birlikte merkezler dış
  sınırdan yaklaşık 244 metre içeride başlar. Gerekirse
  `--query-edge-buffer-m 300` gibi açık bir metre değeri verilebilir.

## Ortam

Repo, harici proje koduna ihtiyaç duymadan çalışır. Görüntü bölme, model
inference, birleştirme ve jeoreferanslama işlemleri için kullanılan
`goruntu_islemleri.py` bu depo içinde tutulur. Python ortamı
`environment.yml` ile yeniden oluşturulabilir:

```powershell
conda env create -f environment.yml
```

PowerShell'de ortamı önce etkinleştirin. `conda run` Türkçe canlı logları bazı
Windows kod sayfalarında geri basarken kodlama hatası verebildiği için doğrudan
etkin ortamdan `python` çalıştırılması önerilir.

```powershell
conda activate visual_navigation_cuda
cd C:\d_surucusu\visual_navigation\benchmark
```

## 1. Hızlı RAW smoke testi

Model üretmeden örnekleme, arama ve checkpoint zincirini sınar:

```powershell
python geospatial_model_benchmark.py `
  --no-include-models `
  --max-queries 10 `
  --run-id raw_smoke
```

## 2. Tek model pilotu

```powershell
python geospatial_model_benchmark.py `
  --models "13_05_2023__14_07_model_step_300.h5" `
  --max-queries 25 `
  --run-id pilot_tek_model
```

## 3. Önerilen üç model pilotu

```powershell
python geospatial_model_benchmark.py `
  --models "13_05_2023__14_07_model_step_300.h5" `
  --models "GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5" `
  --models "GPU_model_f48_k5_epoch_00500_sigmoid_(2, 2)_.h5" `
  --max-queries 100 `
  --run-id pilot_uc_model
```

## 4. Tüm modeller, 300 sorgu

`--models` verilmediğinde yerel `models` klasöründeki bütün `.h5/.keras` dosyaları
işlenir.

```powershell
python geospatial_model_benchmark.py `
  --max-queries 300 `
  --samples-per-block 5 `
  --run-id urgup_tum_modeller_300
```

Varsayılan `clean,hard_v1` seçimi sorgu eşleştirme ve sorgu-inference yükünü
yaklaşık iki katına çıkarır; model haritası her model ve yön için yine yalnız bir
kez üretilir.

İki sağlayıcı yönü varsayılan olarak birlikte ölçülür: önce Google sorgusu Bing
haritasında, ardından Bing sorgusu Google haritasında aranır. Yalnızca ilk yönü
çalıştırmak için `--no-bidirectional` kullanın. Varsayılan iki yönlü çalışma,
tek yönlü çalışmaya göre hesaplama ve disk maliyetini yaklaşık iki katına çıkarır.

## Devam ettirme

Aynı bilimsel ayarları tekrar vererek mevcut koşu klasöründen devam edin:

```powershell
python geospatial_model_benchmark.py `
  --resume-run benchmark\outputs\urgup_tum_modeller_300 `
  --max-queries 300 `
  --samples-per-block 5
```

Tamamlanmış `direction + query_variant + search_mode + model + query_id` kayıtları atlanır. Doğrulanmış model
GeoTIFF'leri ve sorgu çıktıları yeniden kullanılır.

`hard_v1` profil tanımı `run_config.json` içinde saklanır. Farklı bir İHA
bozulma profiliyle üretilmiş sonuçların bulunduğu koşu klasörüne devam edilmeye
çalışılırsa benchmark eski ve yeni sonuçları karıştırmak yerine yeni bir
`--run-id` ister.

## Canlı bilgilendirme

Konsol ve `benchmark.log` şunları ayrıntılı olarak bildirir:

- raster CRS, çözünürlük, boyut ve ortak sınır;
- örnekleme seed'i, blok düzeni ve kabul edilen sorgu sayısı;
- TensorFlow sürümü, CUDA derleme durumu ve görülen GPU'lar;
- her modelin yükleme/inference/birleştirme/jeoreferanslama aşaması;
- sorgu bazında hata, NCC skoru, durum, yüzde ilerleme ve ETA;
- her model sonunda ara Excel başlangıç/bitiş durumu ve raporlama süresi;
- atlanan checkpointler ve silinen yeniden üretilebilir ara çıktılar;
- model bazında hata tracebackleri.

## Çıktılar

Her koşu `benchmark\outputs\<run_id>\` altında tutulur:

```text
benchmark.log                 ayrıntılı çalışma günlüğü
run_config.json               bütün parametreler ve ortam bilgisi
results.jsonl                 sorgu bazında checkpoint kaynağı
results.csv                   Excel/dış analiz için düz ham tablo
summary.json / summary.csv    model/yön özetleri
model_errors.jsonl            model düzeyindeki hatalar
benchmark_results.xlsx        nihai çalışma kitabı
excel_validation.json         XLSX bütünlük, sayfa, formül ve grafik doğrulaması
excel_previews\*.png          yalnız Artifact motorunda üretilen görsel QA çıktıları
<direction>\queries\          sabit sorgu manifesti ve ham sorgular
<direction>\models\           model GeoTIFF'leri ve sorgu çıktıları
```

Excel çalışma kitabı şu sayfaları içerir:

- `Özet`
- `Model Özeti`
- `Ham Sonuçlar`
- `Sorgu Manifesti`
- `Yapılandırma`
- `Hatalar`

Excel raporlayıcı varsayılan `--excel-engine auto` modunda önce
`@oai/artifact-tool` motorunu dener. Motor çalışma ortamında yoksa kullanıcı
onaylı `openpyxl` yedeğine otomatik geçer; seçilen motor ve bütün doğrulama
aşamaları canlı logda belirtilir. Yalnız belirli bir motoru zorlamak için
`--excel-engine artifact` veya `--excel-engine openpyxl` kullanılabilir.

`visual_navigation_cuda` ortamında yedek motor bağımlılığı bir kez kurulmalıdır:

```powershell
python -m pip install openpyxl==3.1.5
```

Üretim sonunda XLSX ZIP bütünlüğü, gerekli sayfalar, tablo/grafik nesneleri ve
formül hata sabitleri denetlenir; sonuç `excel_validation.json` dosyasına yazılır.
İki motor da kullanılamazsa JSONL/CSV checkpointleri korunur ve varsayılan
`--strict-excel` koşuyu raporlama aşamasında başarısız işaretler.

Varsayılan `--excel-update model`, her model başarılı veya başarısız biçimde
sona erdiğinde `summary.json`, `summary.csv` ve `benchmark_results.xlsx`
dosyalarını yeniler. Ara Excel yazımı başarısız olursa (örneğin dosya Excel'de
açık ve Windows tarafından kilitliyse) benchmark sonraki modele devam eder;
koşu sonundaki nihai Excel üretiminde `--strict-excel` yine uygulanır. Yalnız
koşu sonunda Excel üretmek için `--excel-update end` kullanılabilir.

## Temel metrikler

Ana benchmark ölçütü `success_30m` değeridir: 30 metre veya daha düşük hatalı
konumlamaların bütün sorgulara oranı. Reddedilen ve hata veren sorgular bu
oranda başarısız sayılır. Model sıralaması önce `success_30m`, sonra `AUC@30m`,
sonra yalnız 30 m içinde başarılı sonuçların medyan hatasıyla yapılır.

`AUC@30m`, 0-30 m hata aralığındaki başarı CDF alanının normalize edilmiş
özetidir; 30 m üzerindeki sonuçlar sıfır katkı yapar. `mean_error_under_30m` ve
`median_error_under_30m` yalnız operasyonel olarak başarılı eşleşmelerin
hassasiyetini gösterir. Bütün sonuçlar üzerindeki ortalama, medyan ve P90/P95
hata değerleri tanısal amaçla korunur fakat ana model sıralamasında kullanılmaz.
Eski `success_5m`, `success_10m`, `success_25m` ve `success_50m` oranları da
geriye dönük karşılaştırma için korunur; bunlar yalnızca geçerli eşleşmeler
üzerinden hesaplanan ikincil göstergelerdir.

Her sorgu için ayrıca UTM/piksel hatası, Top-1/Top-2 NCC, peak margin, PSR,
5/10/25/30/50 metre başarı bayrakları, inference/search süreleri ve sorgu doku
istatistikleri kaydedilir. 30 m başarı oranı ve tanısal medyan hata için mekânsal
blok bootstrap yöntemiyle %95 güven aralıkları hesaplanır (varsayılan 1.000
tekrar; `--bootstrap-iterations` ile değiştirilebilir).

## Önemli yorum sınırı

Bu modeller Ürgüp/Kapadokya verileriyle ilişkili olabilir. Eğitim alanı
çakışması kesin olarak dışlanmadıkça sonuçlar `in-domain` benchmark olarak
raporlanmalı; dış bölge genellemesi iddia edilmemelidir.
