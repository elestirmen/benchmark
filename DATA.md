# Raster verileri

Varsayılan benchmark çifti depo kökünde şu dosya adlarını bekler:

- `urgup_30_cm_yeni_gmaps_utm.tif`
- `urgup_bingmap_utm_30_cm.tif`

Bu büyük GeoTIFF dosyaları Git deposuna dahil edilmez. İki rasterın da
`EPSG:32636` (WGS 84 / UTM zone 36N) olması ve metre cinsinden aynı piksel
çözünürlüğünü kullanması gerekir. Rasterlar kuzey-yukarı, rotation/skew içermeyen
affine dönüşüme; 1 veya en az 3 adet `uint8` banda sahip olmalıdır. Benchmark
sessiz resampling, reprojection veya `uint16`/float → `uint8` dönüşümü yapmaz.
Başka rasterlar `--query-raster` ve `--map-raster` seçenekleriyle verilebilir.
