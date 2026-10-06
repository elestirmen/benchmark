# deney_araclari/ — el ile yürütülen deneylerin araçları

Deneylerin verisi (modeller, sonuçlar, günlükler) burada değil, `benchmark/deneyler/<deney>/`
altında durur (bkz. `deneyler/BENIOKU.md`). Tüm sonuçların tek görünümü: `deneyler/PANO.md`
(`python sonuc_panosu.py`).

| Dosya | Ne yapar | Ortam |
|---|---|---|
| `torch_flat_egitim.py` | `autoencoder_unified.py` `flat_bottleneck` eğitiminin PyTorch eşdeğeri (aynı veri, ön işleme, TF `same` dolgusu, BCE-logit, Adam 1e-3/eps 1e-7, Keras başlatması). Seçenekler: `--outer-kernel 5` (deneme_6 karma çekirdek), `--lr-schedule "1e-3:6,3e-4:3"`, `--ema 0.9998` (her kayıtta `ema_XXXXX.npz` da), `--aug 0.5` (fotometrik İHA bozulmaları), `--init-npz` (ince ayar). | `adaf` (torch 2.8 cu128) |
| `agirlik_ortalama.py` | Aynı koşunun birkaç checkpoint'inin ağırlık ortalaması (SWA) → tek model `.npz`. | `adaf` |
| `torch_to_keras.py` | `.npz` → benchmark'ın yüklediği Keras `.h5`; her dışa aktarımı PyTorch çıktısıyla karşılaştırır (fark ~1e-6). `epoch_`/`ema_`/`swa_` önekleri ada yansır. | `visual_navigation_cuda` |
| `kapi.sh` | `kapi.sh <örnek> <json_çıktı> <koşu>:<önek> ...` — koşunun tüm `.npz`'lerini dönüştürür ve `margin_gate.py` ile ölçer. Varsayılan 4 grup (G2B/B2G × temiz/hard_v1); `KAPI_GRUPLAR=1` eski tek grup. | Git Bash |
| `topluluk.py` | Birkaç modelin sigmoid çıktısını ortalayan tek `.h5`. | `visual_navigation_cuda` |
| `yansima_toplulugu.py` | Yansıma/döndürme TTA'sı: çekirdekleri çevrilmiş kopyaların ortalaması, tek `.h5` (k3'te birebir; k4'te 1 px kayma). | `visual_navigation_cuda` |
| `rafa_ekle.py` | Bir modeli `deneyler/secili_modeller/` rafına kısa adla kopyalar, kartını yazar (pano kısa adı kullanır). | herhangi |
| `kapi_ozet.py` | Kapı JSON'larını tek sıralı tabloda gösterir (4 grubun ayrı oranlarıyla). | herhangi |
| `kapi_kalibrasyon.py` | Kapı oranlarının resmi sonuçları protokol başına ne kadar iyi sıraladığını (Spearman) ölçer; 4 grup vs eski tek grup. | `visual_navigation_cuda` |
| `sicaklik.py` | Sigmoid çıkışını T ile ölçekler (deney aracı; iyileştirme vermedi). | `visual_navigation_cuda` |
| `ucus_testi.py` | Gerçek İHA uçuş testi (template-matching BENCHMARK modu): `harita` (benchmark karo planıyla model haritası), `test` (kare başına hata; **GPS merkezi yedeği kapalı**), `kimlik` (RAW karşılığı). `--protokol F1/F2`. A/B: `--yaw-kaynagi GIMBAL`, `--gsd-interp AREA`, `--coklu-aday K` (varsayılanlar önceki davranış). | `visual_navigation_cuda` |
| `ab_karsilastir.py` | Uçuş testi A/B kararı: iki ayar kolunu kare bazında eşleştirir (McNemar), önceden yazılmış KABUL/RED/BELİRSİZ kuralını uygular; `--birebir` varsayılan davranışın değişmediğini kanıtlar. | herhangi |
| `exif_tani.py` | Rota karelerinin EXIF/XMP tanısı: gimbal ve uçuş yawı farkı, irtifa tutarlılığı (GPU ve DEM gerekmez). | herhangi (PIL) |
| `raster_utm.py` | RGB raster'ı benchmark ızgarasına (UTM, 0.298582 m/px, alfa bandı) yeniden örnekler; `--referans` ile iki kaynağı aynı ızgaraya oturtur. | `visual_navigation_cuda` |
| `hizli_vekil.py` | 2 × 6×6 km hızlı vekil benchmark. **Yeni (kenar çizen) modelleri iyimser ölçer**; seçimde kapıyı kullanın. | `visual_navigation_cuda` (GPU) |
| `degerlendir.sh` | Dönüştür + vekil. | Git Bash |
| `giris_kalibre.py` | İlk katmana giriş kazancı/ofseti gömer (deney aracı; iyileştirme vermedi). | `visual_navigation_cuda` |
| `norm_probe.py`, `viz_outputs.py` | Kazananların eğitim normalizasyonu probu ve çıktı görselleştirme. | `visual_navigation_cuda` |
| `egitim_surucusu.py` | `autoencoder_unified.py`'yi pipeline'ın kod üreticisiyle tek iş olarak koşturur. | `visual_navigation_cuda` |

Ortam notları: TF ortamında Git Bash'ten çalıştırırken `$ENV/Library/bin` PATH'te olmalı;
GPU'yu TF'ten gizlemek için `CUDA_VISIBLE_DEVICES=-1` kullanın (boş değer gizlemez).
İki resmi benchmark aynı anda koşturulmamalı (işçi başına ~23 GB commit).
