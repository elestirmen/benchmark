# Gerçek İHA kareleriyle kare modeli uyarlaması (2026-10-02)

Soru: x1'i (2022 TPU cb62 + k_32_3 e1900 + deneme6 topluluğu, 14 rotalık uçuş testinin en iyisi)
örnek alarak gerçek uçuşta daha iyi bir model eğitilebilir mi?

## Yaklaşım: harita sabit, yalnız kare modeli eğitilir

TM, model haritasını hazır bir dosyadan okur, İHA karesini ayrı bir modelden geçirir. Bu deneyde
harita **x1'in uydu haritası olarak kalır** (uçuş öncesi, yerde, bir kez üretilir); eğitilen yalnızca
gerçek İHA karesini işleyen model. Öğrenci x1'in ağırlıklarıyla başlar (0. adım = x1) ve İHA
karesinden x1 haritasında doğru konumda tepe veren bir çıktı üretmeyi öğrenir.

- **Kayıp:** TM'nin ölçütü TM_CCOEFF_NORMED; öğrenci çıktısı (512 px şablon) ile 1280 px x1 harita
  kesiti arasındaki NCC haritası FFT ile hesaplanır; softmax(NCC/τ, τ = 0.02) üzerinde doğru konumun
  ±2 px komşuluğunun olasılığı büyütülür (doğru konumun ±10 px halkası negatiflerden çıkarılır).
- **Veri** (`kayitlar/yakala.sh`, `ucus_testi.py test --yakala`): x1 + x1 haritasıyla 14 rotada kare
  başına model girdisi (544 px, 3 ölçek), merkez ölçek şablonu, GPS merkezli 1280 px harita kesiti,
  eşleşme konumu. `veri/<rota>/kare_XXXX.npz` (~7 GB, git dışı).
- **Etiket (doğru konum):** x1 o karede doğruysa (hata < 25 m; sabit GPS kaymalı iki rotada < 45 m)
  x1'in eşleşmesi, değilse aynı karede doğru olan başka bir modelin (14 rota koşusunun kayıtlarındaki
  max_loc2; arama penceresi tüm modellerde aynı) eşleşmesi. Eğitim: 1039 etiketli kare (%85 x1).
- **Bölme:** eğitim 9 Ürgüp rotası; doğrulama (yalnız checkpoint seçimi) 6_tezde_ucus_4 +
  guzergah_4_tezde_ucus_3; **test Karlık 5/6/7** (eğitimde hiç yok; x1'in en zayıf bölgesi).
- Araçlar: `deney_araclari/keras_to_npz.py` (Keras → torch, 1e-6 farkla doğrulandı),
  `ucus_uyarlama_egitim.py`, `npz_to_topluluk.py` (torch → tek Keras .h5; x1 gidiş-dönüş farkı 0.0),
  `ucus_uyarlama_olc.py` (GPS vekili), `kayitlar/aday_test.sh` (dışa aktar + TM testi).

## Ölçüt tuzağı: etiketli vekil öğretmen lehine yanlı

İlk koşularda (a1, b*, c*) etiketli doğrulama başarısı 0.882 → 0.81'e "düştü"; eğitime dair her
değişiklik (kayıp türü, etiket kümesi, Adam/SGD, tam toplu adım) aynı düşüşü verdi. Neden: etiketlerin
%85'i x1'in kendi eşleşmesi, o karelerde x1 tanım gereği %100 başarılı; her değişiklik orada yalnız
kaybedebilir. **Modelden bağımsız GPS vekili** (kesitteki NCC tepesi EXIF GPS'e < 25 m mi; tüm
kareler, etiketsiz) ile aynı checkpoint'ler x1'den iyiydi (x1 0.822; a1 e1/e2 0.838). Seçim bundan
sonra GPS vekiliyle yapıldı (`VAL_GPS`). Not: tam toplu teşhiste kaybı birkaç öğrenilemez zor kare
(kayıp 5–9; çoğu yalnız RAW'ın bulduğu kareler) domine ediyor.

## Tarama D (GPS vekiliyle seçim, 6 epoch)

| Koşu | Ayar | en iyi VAL_GPS (x1 0.822) |
|---|---|---:|
| d1 | Adam 3e-5, pos terimi, halka yok | 0.841 (1000. adım) |
| **d2** | **Adam 3e-5, pos yok, ±10 px halka** | **0.844 (1000. adım)** |
| d3 | Adam 1e-4 | (koşuyor) |
| d4 | d2 + fotometrik bozulma 0.5 | (koşuyor) |

Kazanç ilk ~1000 adımda (1 epoch) geliyor, sonra düz (0.83–0.84); daha uzun eğitim katkı vermiyor.

## Sonuç: TM uçuş testi (kare modeli = öğrenci, harita = x1)

**Birincil aday Karlık testinden önce sabitlendi:** `d2_s1000` = `modeller/egitim/d2_lr3e5/step_001000`,
Keras: `modeller/aday/d2_s1000.h5`. Sonuçlar `sonuclar/ucus/d2_s1000/<rota>.json`.

### Doğrulama rotaları (Ürgüp; eğitim rotalarıyla aynı kasaba, mekânsal örtüşme var)

| Rota | x1 ≤25 / ≤70 m | öğrenci ≤25 / ≤70 m | ≤25 m yalnız öğrenci / yalnız x1 |
|---|---|---|---|
| 6_tezde_ucus_4 (167) | 0.850 / 0.922 | **0.916 / 0.970** | 15 / 4 (p = 0.019) |
| guzergah_4_tezde_ucus_3 (210) | 0.552 / 0.838 | **0.605 / 0.914** | 16 / 5 (p = 0.027) |
| toplam | | | **31 / 9 (p = 0.0007)** |

### Test: Karlık (465 ortak kare, eğitimde hiç görülmedi)

| Model | ≤25 m | ≤50 m | ≤70 m | medyan |
|---|---:|---:|---:|---:|
| **öğrenci d2_s1000** | **0.639** | **0.789** | **0.804** | 16 m |
| x1 | 0.617 | 0.768 | 0.778 | 16 m |
| 2022 deneme6 | 0.559 | 0.714 | 0.763 | 18 m |
| 2022 k_32_3 e1900 | 0.527 | 0.669 | 0.677 | 18 m |
| RAW | 0.471 | 0.591 | 0.673 | 35 m |

Rota bazında ≤25 m: Karain 0.670 → **0.728**, Tağar 0.662 → 0.654, Gürlevek 0.507 → 0.507.
McNemar öğrenci–x1: ≤25 m 33 / 23 (p = 0.23), ≤70 m 22 / 10 (p = 0.050). Öğrenci deneme6'yı
(≤25 m 83 / 46, p = 0.001), k_32_3 ve RAW'ı (p < 0.001) anlamlı geçiyor.

**Değerlendirme:** Uyarlama x1'i hiçbir yerde kötüleştirmedi. Eğitildiği bölgede (Ürgüp, yeni
rotalar) kazanç büyük ve anlamlı (≤25 m +5–7 puan); **yeni bölgede (Karlık) kazanç küçük
(+2.2 / +2.6 puan) ve 25 m'de anlamlı değil**, 70 m'de sınırda. Kazancın çoğu yerel görünüş
uyumundan (aynı kasaba, aynı mevsim/ışık) geliyor; bölgeden bağımsız genelleme zayıf ama yönü olumlu.

## Kullanım

Uçuşta: harita = x1 uydu haritası (`topluluk_x1_2022_geo.tif` / `_urgup_geo.tif`), kare modeli =
`modeller/aday/d2_s1000.h5`; ön işleme F1 (minus1_1, iyileştirme yok). Kare başına maliyet x1 ile aynı
(aynı mimari). Öğrenci x1 haritası **olmadan** (kendi haritasıyla) anlamlı değildir.

## Açık işler

- Ters kat: Karlık rotalarıyla eğit → Ürgüp'te test (Karlık verisi `veri/` içinde yakalı).
- Tüm rotalarla eğitilmiş son model (dağıtım için) + yeni bir bölgede gerçek uçuş testi.
- Zor kareler için etiket: GPS kaymalı rotalarda ve x1'in hiç bulamadığı karelerde yerel arama.
