# Geniş doğrulama (2026-09-25)

Soru: 1000 sorguluk resmi benchmark'ta RAW'ı ve 2022 kazananlarının bir kısmını geçen yeni
modeller, daha geniş bir kümede de gerçekten başarılı mı?

## 1. Ürgüp 4000 sorgu — 2022 kazananlarıyla birebir aynı sorgular

- Protokol `run_urgup_cevresi_models.py --max-queries 4000` (seed 42, balanced_exact,
  6 arama modu × clean/hard_v1 × iki yön). Sorgu manifesti 2022 koşusuyla
  (`outputs/urgup_cevresi_models_4000`) **4000/4000 merkezde birebir aynı** (kontrol edildi).
- RAW ve 2022 modelleri bu sorgularda zaten ölçülmüş olduğu için yeniden koşturulmadı
  (`--no-include-raw`); kıyas doğrudan aynı sorgu kümesinde.
- Başlatma: `kayitlar/baslat_urgup4000.sh <aday>` (`modeller/<aday>/` içindeki .h5).
- Çıktı: `sonuclar/benchmark/urgup4000_<aday>/`.

Sonuç (2026-09-25 10:52 → 21:46; ileri yön 372 dk, ters 268 dk), global success_25m:

| Model | Global | Δ RAW | G2B temiz / hard | B2G temiz / hard | roi 500 m | roi 2000 m |
|---|---:|---:|---|---|---:|---:|
| **x1 = 2022 TPU cb62 + k_32_3 + deneme6 topluluğu** (2026-09-26 05:21 → 13:57) | **0.718** | **+0.241** | 0.734 / 0.710 | 0.762 / 0.668 | 0.870 | 0.818 |
| 2022 TPU cb62 (serinin en iyisi; 10 checkpoint 0.631–0.659) | 0.659 | +0.182 | 0.678 / 0.656 | 0.703 / 0.598 | 0.848 | 0.781 |
| **yeni topluluk3** | **0.625** | **+0.148** | 0.666 / 0.633 | 0.666 / 0.536 | 0.799 | 0.734 |
| 2022 k_32_3 e1900 | 0.620 | +0.143 | 0.652 / 0.609 | 0.694 / 0.523 | 0.824 | 0.748 |
| 2022 TPU e1100 | 0.614 | +0.137 | 0.637 / 0.602 | 0.658 / 0.559 | 0.812 | 0.742 |
| 2022 deneme6 | 0.601 | +0.124 | 0.641 / 0.604 | 0.630 / 0.531 | 0.789 | 0.714 |
| 2022 06_modeller | 0.504 | +0.027 | 0.535 / 0.558 | 0.495 / 0.429 | 0.735 | 0.645 |
| RAW | 0.477 | — | 0.616 / 0.583 | 0.367 / 0.342 | 0.678 | 0.595 |

**x1 en iyi tekil modeli (TPU cb62) aynı 4000 sorguda +0.059 geçiyor** (1000 sorguda 0.719, 4000'de 0.718); her grup ve modda önde.

**Doğrulandı:** 1000 sorgudaki fark (+0.155) 4000'de korunuyor (+0.148). Yeni topluluk TPU
serisi dışındaki tüm 2022 modellerini geçiyor; TPU serisinin arkasında. Zayıf noktalar: B2G hard
(0.536 / TPU 0.598) ve yerel modlar (roi 500 m 0.799 / TPU 0.848).

## 2. Karlık gerçek İHA uçuş testi — farklı bölge, gerçek kamera

- 198 karenin 191'i işlenir (güncel kod; 2026-09-12 testinde 116) (`template-matching/guzergahlar/guzergah_5_to_karain`, DJI), EXIF GPS merkezli
  5000 px pencere (`BENCHMARK=True`, her kare bağımsız). Hata = TM konumu ile EXIF GPS arası.
- **Protokol F1:** harita ve kareler resmi uydu benchmark'ıyla aynı ön işlemeyle
  (`minus1_1`, enhancement `none`); harita benchmark'ın `build_model_map`'i ile
  (544 karo, 32 bindirme). Kaynak: `E:\tez__ayna\python_calismalar\AutoEncoder_pix2pix\karlik_30_cm_bingmap_utm.tif`.
- RAW karşılığı: `RAW_kimlik.h5` = sigmoid(1.5·x) tek 1×1 conv (modelsiz eşleştirmeye monoton yakınsama).
- Araç: `deney_araclari/ucus_testi.py` (`harita`, `test`, `kimlik`); kuyruk `kayitlar/ucus_kuyrugu.sh`.
- Çıktı: `sonuclar/ucus/karlik_f1/<model>.json` (kare başına hata ve TM skoru).

### Ölçüm hatası bulundu ve düzeltildi: kesişimsiz kare = GPS merkezi

template-matching ana betiğinde `NO_INTERSECTION_USE_SEARCH_CENTER=True` (varsayılan):
üç şablonun eşleşmesi kesişmeyince tahmin **arama merkezi** yapılır. BENCHMARK modunda
arama merkezi EXIF GPS'in kendisi olduğu için bulunamayan kareler **0 m hata** alır
(ilk koşuda RAW'da 191 karenin 32'si). `ucus_testi.py` bu yedeği kapatır
(`False`: tekil şablonun gerçek eşleşmesi); geçersiz ilk iki sonuç
`sonuclar/ucus/_gecersiz_gps_merkezi_yedegi/` altında (RAW 0.953 → düzeltilmiş 0.801).
Aynı sızıntı ana betiğin kendi benchmark raporlarını da iyimser gösterir.

### F1 sonuçları (191 kare, düzeltilmiş)

| Model | ≤70 m | ≤50 m | ≤25 m | Medyan | Kesişimsiz kare |
|---|---:|---:|---:|---:|---:|
| RAW (kimlik) | 0.801 | 0.785 | 0.660 | 11 m | 30 |
| yeni topluluk3 | 0.785 | 0.723 | 0.550 | 16 m | 34 |
| 2022 k_32_3 e1900 | 0.775 | 0.770 | 0.613 | 16 m | 26 |
| yeni tam f32k3 e5 | 0.775 | 0.717 | 0.555 | 16 m | – |
| 2022 TPU e1100 | 0.613 | 0.581 | 0.393 | 43 m | 55 |
| 2022 TPU cb62 | 0.534 | 0.492 | 0.325 | 50 m | 62 |
| yeni pipeline z1 e9 | 0.508 | 0.440 | 0.267 | 63 m | 82 |
| 2022 deneme6 f32_k5 (tamamlama turu) | **0.843** | **0.838** | **0.738** | 12 m | 22 |
| x1 = TPU cb62 + k_32_3 e1900 + deneme6 topluluğu (2026-10-01) | 0.812 | 0.806 | 0.670 | 15 m | 27 |

Gerçek İHA karelerinde resmi ön işlemeyle (`minus1_1`) yalnız **deneme6** RAW'ı geçiyor; yeni
modeller RAW düzeyinde ya da altında, uydu benchmark'ının şampiyonu TPU cb62 sondan ikinci.
Uydu sıralaması uçuş görevine aktarılmıyor.

**x1 (2026-10-01, `kayitlar/ucus_x1.sh`):** uydu şampiyonu topluluk uçuşta RAW düzeyinde, deneme6'nın
altında. Kare bazında eşleştirilmiş McNemar (191 kare): RAW'a karşı ≤25 m 36/34 kare (p = 0.91),
≤70 m 22/20 (p = 0.88), yani fark yok; deneme6'ya karşı ≤25 m 10/23 (p = 0.035) kötü;
k_32_3'e karşı ≤25 m 18/7 (p = 0.043), TPU cb62'ye karşı 68/2 (p < 0.001) iyi. Topluluk zayıf
TPU üyesini toparlıyor ama en iyi üyesi deneme6'yı uçuşta seyreltiyor. (Bu yargı tek rotaya
dayanıyordu; 14 rotada tersine döndü, bkz. "Tüm güzergâhlar".)

### Tüm güzergâhlar (2026-10-01, F1, 5 temel model)

`kayitlar/ucus_rotalar.sh`; özet `deney_araclari/ucus_ozet.py` → `sonuclar/ucus/ROTALAR_OZET.md`
(rota tablosu, bölge toplamları, eşleştirilmiş McNemar). Sonuçlar `sonuclar/ucus/rota_<rota>/`.

- **Ürgüp merkezi** (11 rota, 1579 ortak kare): harita `benchmark/urgup_bingmap_utm_30_cm.tif`
  (`ucus_testi.py harita --kaynak ... --etiket urgup`), DEM `ana_harita_urgup_30_cm_utm_elevation.tif`.
  Kasaba + çevresi; eğitim bölgesiyle ilişkili olabilir (in-domain).
- **Karlık** (3 rota, 465 ortak kare): guzergah_5 (`karlik_f1/`), 6, 7; Karlık haritası ve DEM.
- **Dışarıda:** guzergah_8_koy_vadi — FC2204 (4.4 mm, 77–200 m irtifa): 30 cm'ye ölçeklenen kare
  TM patch'inden küçük, 435 karenin hepsi "merkezi dışarıda" → 0 kare (modelden bağımsız).
  guzergah_9_koy_dag — karelerin çoğu Karlık haritası/DEM'i dışında, koşulmadı.
- **GPS kayması:** guzergah_1_donus ve guzergah_2_tezde_ucus_1'de doğru eşleşen karelerin hatası
  sabit ~35 m (23–40 m); ≤25 m bu iki rotada anlamsız, ≤50/70 m geçerli. Model sıralamasını değiştirmiyor.

| Model | Ürgüp ≤25 / ≤70 m | Karlık ≤25 / ≤70 m | Tümü (2044 kare) ≤25 / ≤70 m | Medyan |
|---|---|---|---|---:|
| **x1 topluluğu** | **0.677 / 0.876** | **0.617 / 0.778** | **0.663 / 0.854** | 18 m |
| 2022 k_32_3 e1900 | 0.657 / 0.858 | 0.527 / 0.677 | 0.628 / 0.817 | 20 m |
| 2022 deneme6 | 0.618 / 0.835 | 0.559 / 0.763 | 0.605 / 0.819 | 20 m |
| RAW (kimlik) | 0.560 / 0.764 | 0.471 / 0.673 | 0.540 / 0.744 | 23 m |
| 2022 TPU cb62 | 0.582 / 0.809 | 0.323 / 0.514 | 0.523 / 0.742 | 23 m |

- **x1 iki bölgede de birinci.** Tümünde McNemar ≤25 m: RAW'a karşı 408/156, k_32_3'e karşı 135/62,
  deneme6'ya karşı 227/107 (hepsi p < 0.001). Karlık'ta deneme6'ya karşı ≤25 m 81/54 (p = 0.025),
  ≤70 m 45/38 (p = 0.51, fark yok).
- guzergah_5'te deneme6'nın x1'e üstünlüğü (0.738 / 0.670) rotaya özgü: guzergah_6'da 0.446 / 0.662,
  guzergah_7'de 0.424 / 0.507. **Tek rotalık uçuş testiyle model seçilmemeli.**
- TPU cb62 Ürgüp'te RAW'dan iyi (≤70 m 222/152, p < 0.001), Karlık'ta çok kötü (≤70 m 0.514): uydu
  şampiyonu tekil model eğitim bölgesi dışına genellemiyor; x1 topluluğu genelliyor.
- Modellerin RAW'a kazancı uçuşta uydudakinden küçük ama gerçek: x1 tümünde +0.123 (≤25 m).

### F2 — modellerin eğitim ön işlemesi (`zero_1`, harita ve kare)

| Model | F1 ≤70 m | F2 ≤70 m | F2 ≤25 m | F2 medyan |
|---|---:|---:|---:|---:|
| yeni topluluk3 | 0.785 | 0.513 | 0.325 | 52 m |
| yeni tam e5 | 0.775 | 0.414 | 0.215 | 184 m |
| 2022 k_32_3 e1900 | 0.775 | 0.545 | 0.325 | 49 m |
| 2022 TPU cb62 | 0.534 | 0.325 | 0.162 | 362 m |

Hepsi F1'den çok kötü: uçuştaki zayıflık ön işleme uyumsuzluğundan değil. Tersine "uyumsuz"
`minus1_1` (zero_1 ile eğitilmiş modele 2× kontrast) hem uyduda hem uçuşta daha iyi; çıktıyı
doygunlaştırıyor ve eşleştirme bundan yararlanıyor.

### 2026-09-12 testiyle (outputs/karlik_top_modeller) neden doğrudan kıyaslanmıyor

1. **Hata hesabı değişti.** Aynı model/haritayla (TPU bdb7) eski protokolü tekrar koşturduğumda
   kare başına **TM skorları birebir aynı** çıktı (0.113, 0.132, 0.102, …) ama hatalar
   farklı (ör. 44→31 m, 51→5 m; ilk 18 karede medyan 12 m). template-matching ana betiği
   2026-09-16'da güncellendi; aynı tepe için artık farklı (çoğunlukla daha küçük) hata
   hesaplıyor. Eski yüzdeler bugünkü kodla kıyaslanamaz. `sonuclar/ucus/tekrar_bdb7/`.
2. **Ön işleme sözleşmeleri modelden modele farklıydı** (`top_modeller/*.pre.json`,
   bazıları "eğitim scriptinden elle çıkarıldı"): 2022 modellerine `minus1_1 + hist_eq_pil`
   verilmiş. Bern verisinde BCE probu (`deney_araclari/norm_probe.py`, 48 karo): TPU cb62
   ham/zero_1 **0.424**, ham/minus1_1 1.823, eşitlenmiş/zero_1 0.740,
   eşitlenmiş/minus1_1 **2.567**; k_32_3 e1900: 0.405 / 1.249 / 0.580 / 1.610. Kazananlar
   eşitlenmemiş [0,1] girişle eğitilmiş; o testte en uyumsuz ön işlemeyi almışlar. Ancak
   TPU'nun uçuştaki zayıflığı ön işlemeden bağımsız: eşitlemesiz F1'de 0.534, eğitim ön
   işlemesi F2'de 0.325 (70 m).

## 3. Adana — farklı bölge (kentsel), aynı uydu çapraz-kaynak protokolü

- Kaynak: `E:\tez__ayna\python_calismalar\AutoEncoder_pix2pix\adana_{gmaps,bingmaps}.tif` (Web Mercator,
  0.597 m/px ≈ 0.48 m zemin) → `deney_araclari/raster_utm.py` ile UTM 36N, 0.298582 m/px + alfa
  (`veri/`, ~4.4 × 4.5 km). Google–Bing kayması ~2 m (9 bölgede NCC; hizalama gerekmedi).
- 100 sorgu (balanced_exact, seed 42, blok başına 6), modlar roi500 / roi1000 / global, clean + hard_v1,
  iki yön; başlatma `kayitlar/baslat_adana.sh`; çıktı `sonuclar/benchmark/adana100_modeller/`.

| Model | Global (4 grup) | G2B temiz / hard | B2G temiz / hard |
|---|---:|---|---|
| RAW | **0.972** | 0.990 / 0.980 | 0.960 / 0.960 |
| 2022 TPU cb62 | **0.972** | 0.990 / 0.960 | 0.970 / 0.970 |
| x1 (2022 topluluğu) | 0.955 | 0.960 / 0.940 | 0.970 / 0.950 |
| 2022 deneme6 | 0.932 | 0.960 / 0.920 | 0.920 / 0.930 |
| 2022 k_32_3 e1900 | 0.925 | 0.950 / 0.870 | 0.930 / 0.950 |
| yeni tta8_tam_e5 | 0.922 | 0.950 / 0.870 | 0.940 / 0.930 |
| yeni pipeline z1 e9 | 0.917 | 0.950 / 0.890 | 0.910 / 0.920 |
| yeni tam e5 | 0.897 | 0.930 / 0.840 | 0.920 / 0.900 |
| yeni topluluk3 | 0.880 | 0.880 / 0.840 | 0.920 / 0.880 |

Tavan etkisi: kentsel Adana çiftinde iki kaynak çok benzer ve alan küçük, RAW %97. Hiçbir model
RAW'ı geçmiyor (TPU eşit); yeni modeller eğitim dışı bu bölgede 5–9 puan kaybediyor (TTA burada da
yardımcı: 0.897 → 0.922). 100 sorguda bir grubun standart hatası ~%3.

## Genel sonuç (üç doğrulama)

Yeni modellerin RAW'a karşı kazancı **Kapadokya uydu çapraz-kaynak görevine özgü** (RAW'ın zayıf
olduğu B2G ağırlıklı): Ürgüp 4000 sorguda doğrulandı (+0.148, TPU dışındaki 2022 modellerinden
iyi). Farklı bölgede (Adana, kolay kentsel) ve gerçek İHA karelerinde (Karlık) RAW'ı geçen yeni
model yok; 2022 modellerinde de tablo karışık (TPU Adana'da RAW'a eşit ama uçuşta zayıf,
deneme6 uçuşta RAW'dan iyi). README'deki "bölge dışı genelleme iddia edilmemeli" uyarısını
deneysel olarak destekliyor.
