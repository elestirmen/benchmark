# Raster verileri

Varsayılan benchmark çifti depo kökünde şu dosya adlarını bekler:

- `urgup_30_cm_yeni_gmaps_utm.tif`
- `urgup_bingmap_utm_30_cm.tif`

Bu büyük GeoTIFF dosyaları Git deposuna dahil edilmez. İki rasterın da
`EPSG:32636` (WGS 84 / UTM zone 36N) olması ve metre cinsinden uyumlu piksel
çözünürlüğü kullanması gerekir. Başka rasterlar `--query-raster` ve
`--map-raster` seçenekleriyle verilebilir.

