# Parametre araması (2026-09-25)

Amaç: 2026-09-24'te bulunan tarifin (f32_k3 flat_bottleneck, zero_1, BCE, Adam 1e-3, batch 1,
her epoch karıştırma; bkz. `../2026-09-24_zero1_tarif/BENIOKU.md`) başarısını artırmak.

## Seçim ölçütü: 4 gruplu kapı

`margin_gate.py --bidirectional --variants clean,hard_v1`: resmi global skorun dört grubu
(G2B/B2G × temiz/hard_v1) için yanlış tepe / gerçek tepe oranı, ortalama. Eski kapı yalnız
G2B temiz ölçüyordu; kalan açık ise en çok B2G hard_v1'de. NCC artık tek matris çarpımıyla
(150 örnekte eski ikili döngüyle aynı değer: RAW 0.5307). 4 gruplu RAW barajı **0.585**
(g2b temiz 0.531, g2b hard 0.559, b2g temiz 0.622, b2g hard 0.626).

Kalibrasyon (resmi sonucu bilinen 39 model): `sonuclar/kapi/kapi4_kalibrasyon_150.json`.

## Denenenler

| Aile | Ne | Nerede |
|---|---|---|
| SWA | Aynı koşunun ardışık epoch ağırlıklarının ortalaması (eğitimsiz) | `modeller/swa/` |
| TTA | Yansıma/döndürme ortalaması, çekirdek dönüşümüyle tek .h5 (eğitimsiz) | `modeller/tta/` |
| Kuşaklar arası topluluk | 2022 kazananları ± yeni modeller | `modeller/topluluklar/` |
| EMA | Ağırlık EMA'sı (0.9998/adım); her kayıtta ham + EMA checkpoint | `modeller/egitim/r1_s11_ema` |
| LR takvimi | R1 e6'dan 3e-4 → 1e-4 ince ayar | `modeller/egitim/r2_s11_lrd` |
| Augmentasyon | Fotometrik İHA bozulmaları (%50 örnek) | `modeller/egitim/r3_s11_aug` |
| Kapasite | f48_k3 | `modeller/egitim/r4_s11_f48` |
| İnce ayar | En iyi 3 checkpoint'e lr düşürmeli ince ayar + EMA | `modeller/egitim/ft_*` |
| Çıktı sıcaklığı | sigmoid(T·z), T = 0.5 … 8 (eğitimsiz) | `modeller/sicaklik/` |
| Seed piyangosu | Yalın tarif, seed 12 / 13 / 14, her epoch kapıdan | `modeller/egitim/p_s*` |
| Resmi koşular | x1 ve tta8_tam_e5 (1000 sorgu) | `sonuclar/benchmark/urgup1000_*` |

Eğitim: `kayitlar/egit.sh <ad> [torch_flat_egitim.py argümanları]`. Kuyruklar (sırayla koşan,
benchmark'ın yanında tek iş kuralıyla): `kuyruk.sh` (R2–R4), `kapi_sirali.sh` (G1–G3),
`kapi_g4_sicaklik.sh`, `ince_ayar_kuyrugu.sh` (G6), `kapi_g8_tta_k4.sh`, `resmi_kuyruk.sh`
(`resmi_adaylar.txt`), `sabah_kuyrugu.sh` (G9 + x1 4000 sorgu). R5 (seed 12 + EMA) bellek
sıkışması yüzünden durduruldu; yerine piyango koştu.

## Bulgular

### Kalibrasyon (4 grup vs eski tek grup kapı)

Spearman(kapı oranı, resmi global − RAW), resmi sonucu bilinen modeller:

| Protokol | Model | 4 grup | eski (G2B temiz) |
|---|---:|---:|---:|
| Ürgüp 4000 sorgu | 19 | **−0.918** | −0.847 |
| Ürgüp 1000 sorgu | 11 | **−0.909** | −0.718 |
| eski küçük raster 300 sorgu | 16 | −0.691 | −0.647 |
| eski küçük raster 2000 sorgu | 20 | −0.677 | −0.656 |

`end_to_end_pipeline.py` artık 4 gruplu kapıyla seçiyor (`MARGIN_GATE_VARIANTS`, `MARGIN_GATE_BIDIRECTIONAL`).
4 grupta: 2022 TPU serisi 0.42–0.46, k_32_3 e1900 0.442, deneme_6 0.480, **yeni topluluk3 0.495**,
yeni tam e5 0.517, yeni pipeline e9 0.534, RAW 0.584. Yeni modellerin en zayıf grubu B2G hard_v1
(topluluk3 0.539, TPU cb62 0.453).

### G1 — EMA ve lr takvimi (seed 11)

- Sabit lr 1e-3'te **EMA zarar veriyor** (R1 e4: ham 0.570 / EMA 0.659; e6: 0.551 / 0.585).
  Muhtemel neden: ortalama ağırlıklar çıktıyı yumuşatıyor; eşleştirme keskin, doygun çıktıdan yararlanıyor.
- **lr düşürmeli ince ayar (R1 e6'dan 3e-4×3 + 1e-4×3) + EMA** tüm checkpoint'leri 0.533–0.547'de
  topluyor (en iyi 0.533; aynı koşunun en iyi ham checkpoint'i 0.551): ~0.02 kazanç + TPU serisi
  gibi kararlılık. Düşük lr aşamasında EMA yardımcı.
- Seed farkı bu ayarların etkisinden büyük (seed 11 en iyi 0.551, seed 42 tam e5 0.517).
- Sonraki adım: aynı ince ayar topluluk3 üyelerine (`kayitlar/ince_ayar_kuyrugu.sh`).

### G2 — fotometrik augmentasyon ve kapasite (seed 11)

- **Augmentasyon (R3, %50 örnek):** en iyi 0.549 (augmentasyonsuz R1 0.551) — genel etki yok;
  G2B hard biraz iyileşiyor (0.547 → 0.531) ama B2G temiz kötüleşiyor (0.518 → 0.541); erken
  epochlar çok gürültülü (e4 0.846).
- **f48_k3 (R4), batch 1, lr 1e-3:** kararsız (e2 0.586, e4 1.016 çökme, e6 0.828). Büyük model
  bu lr'de eğitilemiyor (2026-09-24 f48_k4 b1 denemesiyle tutarlı).

### G3 — SWA, TTA, kuşaklar arası topluluk (eğitimsiz)

| Aday | Kapı |
|---|---:|
| **x1 = 2022 TPU cb62 + k_32_3 e1900 + deneme6** (çıktı ortalaması) | **0.406** |
| x3 = TPU + k_32_3 + yeni 3 üye | 0.419 |
| x2 = TPU + yeni 3 üye | 0.442 |
| tta8_tam_e5 (8 dihedral dönüşüm) | 0.482 |
| tta4_tam_e5 (id, yatay/dikey çevirme, 180°) | 0.487 |
| tta4_topluluk3 (3 üye × 4) | 0.488 |
| (tek) tam e5 / UZUN e27 / TAM_s2 e7 | 0.517 / 0.520 / 0.541 |
| SWA (9 aday) | 0.555 – 0.820 |

- **TTA yeni soyda en iyi iyileştirme** (0.517 → 0.482; B2G hard 0.553 → 0.513); topluluk ile
  üst üste binmiyor (ikisi de aynı gürültüyü azaltıyor).
- **2022 kazananlarının topluluğu şimdiye kadarki en iyi kapı değeri** (tek TPU 0.422–0.440);
  yeni modelleri eklemek seyreltiyor.
- **SWA zarar veriyor** (yüksek lr'deki epochların ağırlık ortalaması).
- k4 modellerde TTA: TF `same` dolgusu yüzünden çevrilmiş kopya 1 px kayıyor
  (fliplr (0,1), flipud (1,0), rot180 (1,1)); `yansima_toplulugu.py` kaymayı ölçüp
  ZeroPadding2D + Cropping2D ile düzeltiyor (12 px içeride fark ~1e-6).
- **G8: TTA 2022 modellerine zarar veriyor:** TPU cb62 0.440 → TTA4 **0.518**; x1 0.406 → TTA4
  **0.469**. Dönüşüm birebir doğrulandığı için gerçek etki: 2022 modellerinin doygun, ikili
  çıktısı dört çevrilmiş tahminin ortalamasında bulanıklaşıyor; yeni modellerin kenar benzeri
  çıktısında ise ortalama gürültüyü azaltıyor. TTA yalnız yeni soyda kullanılmalı.

### G6 — en iyi checkpoint'lere lr düşürmeli ince ayar (3e-4×3 + 1e-4×3, EMA)

| Başlangıç (kapı) | İnce ayar sonrası en iyi | en kötü |
|---|---:|---:|
| tam e5 (0.517) | 0.569 | 0.611 |
| TAM_s2 e7 (0.541) | 0.556 | 0.984 (e6 çöküş) |
| UZUN e27 (0.520) | 0.536 | 0.561 |

Üçü de başladığı checkpoint'ten **kötü**; val_loss ise iyileşiyor (tam: 0.3787 → 0.3759). G1'deki
iyileşme (R1 e6 0.551 → 0.533) zayıf bir başlangıçtandı: ince ayar modeli ~0.53–0.60 "tipik"
düzeyine çekiyor. **Ortak açıklama (G1, G3 SWA, G4, G6):** BCE'yi daha da düşürmek (yakınsama,
ağırlık ortalaması, düşük lr) çıktıyı BCE-optimal yumuşak çözüme götürüyor ve eşleştirmeyi
bozuyor; iyi eşleştiriciler batch-1'in gürültülü, tam yakınsamamış durumları. Tekil model için en iyi
strateji: yalın tarif (batch 1, lr 1e-3) + her epoch kapıyla seçim (seed piyangosu) + TTA.

### Seed piyangosu (yalın tarif, 12 epoch, her epoch kapıdan)

En iyi epoch: seed 12 0.540, seed 13 0.561, seed 14 0.570 (seed 11 0.551). Yalın tarifin
tipik en iyisi 0.54–0.57; seed 42 tam e5 (0.517) ve UZUN e27 (0.520) şanslı istisnalar.
Birkaç seed daha denemek TPU düzeyine (0.42–0.44) yaklaştırmıyor.

### Resmi 1000 sorgu: x1 (2022 kazananlarının topluluğu)

| Model | Global | Δ RAW | G2B temiz / hard | B2G temiz / hard | roi 500 m | roi 2000 m |
|---|---:|---:|---|---|---:|---:|
| **x1 = TPU cb62 + k_32_3 e1900 + deneme6** | **0.719** | **+0.258** | 0.711 / 0.699 | 0.779 / 0.687 | 0.858 | 0.808 |
| yeni topluluk3 | 0.615 | +0.154 | 0.659 / 0.625 | 0.656 / 0.522 | 0.793 | 0.722 |
| yeni tam e5 | 0.564 | +0.103 | 0.594 / 0.572 | 0.604 / 0.485 | 0.788 | 0.705 |
| RAW | 0.461 | — | 0.581 / 0.553 | 0.363 / 0.346 | 0.682 | 0.581 |

Karşılaştırma için en iyi tekil model TPU cb62 4000 sorguda +0.182 (RAW 0.477). x1 bu farkı
~0.08 artırıyor; kapı bunu önceden gösterdi (0.406, açık ara en iyi). Üyeler bu bölgenin
benchmark'ında kazanan oldukları için seçilmişti; 1000 sorguluk küme farklı bir örneklem
(4000'le yalnızca 2 ortak merkez). **Kesin kıyas (2026-09-26, 2022 ile aynı 4000 sorgu): x1 0.718**
(RAW 0.477, +0.241; TPU cb62 0.659, bdb7 0.657) — G2B 0.734 / 0.710, B2G 0.762 / 0.668, roi 500 m 0.870;
her grup ve modda en iyi tekil modelin önünde (`../2026-09-25_genis_dogrulama/sonuclar/benchmark/urgup4000_x1_2022_topluluk/`).
Rafta: `deneyler/secili_modeller/topluluk_x1_2022.h5`.

### G9 — x1 topluluğunun varyantları (kapı; x1 = 0.406)

| Varyant | Kapı |
|---|---:|
| x6 = TPU bdb7 + k_32_3 + deneme6 | 0.405 |
| x4 = TPU cb62 + TPU bdb7 + k_32_3 + deneme6 | 0.406 |
| x7 = TPU cb62 + k_32_3 | 0.410 |
| x10 = x1 + tta8_tam_e5 | 0.411 |
| x8 = TPU cb62 + deneme6 | 0.421 |
| x9 = k_32_3 + deneme6 | 0.428 |

Üç farklı soydan üç üye en iyi nokta; aynı seriden ikinci TPU ya da TTA'lı yeni model eklemek
kazandırmıyor. Kazancın çoğu TPU + k_32_3 ikilisinden. (`topluluk.py` artık üyelerin iç katman
adlarına önek koyuyor: aynı seriden iki üye h5 kaydında çakışıyordu.)

### Resmi 1000 sorgu: TTA (yeni soy)

| Model | Global | Δ RAW | G2B temiz / hard | B2G temiz / hard | roi 500 m | roi 2000 m |
|---|---:|---:|---|---|---:|---:|
| **tta8_tam_e5** (tam e5 × 8 dihedral dönüşüm) | **0.673** | **+0.212** | 0.698 / 0.675 | 0.712 / 0.606 | 0.833 | 0.768 |
| tam e5 (aynı model, TTA'sız) | 0.564 | +0.103 | 0.594 / 0.572 | 0.604 / 0.485 | 0.788 | 0.705 |

TTA tek başına **+0.109** kazandırıyor; tek bir yeni model TTA ile yeni topluluğu (0.615) ve en iyi
tekil 2022 modelinin RAW farkını (TPU cb62 +0.182) geçiyor. En zayıf grup B2G hard 0.485 → 0.606.
`end_to_end_pipeline.py` artık kapı şampiyonunun TTA8'li sürümünü de kapıdan geçiriyor ve daha
iyiyse benchmark'a onu sokuyor (`BENCHMARK_TTA_TRANSFORMS`, `consider_tta_champion`).

### G4 — çıktı sıcaklığı (sigmoid(T·z), eğitimsiz)

tam e5: T=0.5 0.522, **T=1 0.517**, T=2 0.532, T=4 0.707, T=8 1.104; topluluk3 T=3 0.542 (T=1 0.495).
Sonradan keskinleştirme zarar veriyor; EMA'nın zararı yalnız çıktı keskinliğiyle açıklanamaz.

### U1c — TPU tipi uzun büyük-batch eğitim (f48_k4, batch 16, lr 1e-3, bf16, 300 epoch)

| Epoch | 10 | 25 | 50 | 75 | 100 | 150 | 200 | 250 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_loss (EMA) | 0.476 | 0.332 | 0.321 | 0.317 | 0.315 | 0.313 | 0.312 | 0.310 | 0.309 |
| Kapı (ham) | **0.580** | 0.656 | 0.632 | 0.675 | 0.738 | 0.853 | 0.964 | 1.180 | 1.239 |

BCE hiçbir batch-1 modelin ulaşmadığı düzeye iniyor (0.31; batch 1 ~0.37–0.39), kapı ise tekdüze
kötüleşiyor; en iyi checkpoint ancak RAW düzeyinde (0.584). **Uzun eğitim hipotezi çürüdü** ve
günün ana bulgusunun en açık kanıtı: harita etiketini piksel düzeyinde daha iyi üretmek kaynaklar
arası eşleştirmeyi bozuyor. 2022 TPU serisinin kalitesi bu tür eğitimden gelmiyor (veri, kayıp ya
da ön işleme farklı olmalı; eğitim betiği arşivde yok).

