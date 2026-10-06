# TM iyileştirmeleri: A/B testi ve geri dönüş planı (2026-10-06)

Soru: template-matching'in uçuş hattında iki ucuz değişiklik gerçek İHA karelerinde
başarımı artırıyor mu?

1. **GSD küçültmesinde interpolasyon.** Kare, harita çözünürlüğüne (29.85 cm/px) inerken
   `cv2.INTER_NEAREST` ile küçültülüyordu. Mavic 2 Pro 100 m'de yaklaşık 2.4 cm/px çeker,
   yani kare yaklaşık 12 kat küçülür; en yakın komşu bu oranda örtüşme (aliasing) üretir.
   Referans harita ise alan ortalamalı bir ortomozaiktir ve benchmark piramidi `INTER_AREA`
   kullanır. Hipotez: "uydu sıralaması uçuşa aktarılmıyor" bulgusunun bir kısmı bu farktan.
2. **Çok adaylı kaba arama.** Benchmark top-30 aday + NMS kullanıyor; uçuş testinde
   `PYRAMID_MULTI_CANDIDATE` kapalıydı (tek kaba tepe). Kendine benzer arazide yanlış
   kaba tohuma kilitlenmeyi azaltabilir.

## Koddaki değişiklikler (varsayılanlar eski davranışı korur)

| Depo | Değişiklik | Varsayılan |
|---|---|---|
| template-matching | `RUN_CFG["GSD_RESIZE_INTERP"]`: `NEAREST` / `AREA` / `LINEAR` / `CUBIC` / `AUTO`; merkez, sol-üst ve sağ-alt ölçeklerinin dördü de buna bağlı | `NEAREST` (önceki davranış, sonuçlar birebir aynı) |
| template-matching | `BENCHMARK=True` iken "kesişim yok" ve "düşük skor" arama merkezi yedekleri kendiliğinden kapanır (arama merkezi = EXIF GPS, yani GPS sızıntısı). `BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK=True` eski davranışı geri getirir | kapalı (sızıntı yok) |
| benchmark | `ucus_testi.py test --gsd-interp AREA`, `--coklu-aday K` | `NEAREST`, `0` (önceki davranış) |
| benchmark | `ucus_testi.py --eski-protokol` artık `BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK=True` verir (2026-09-12 testinin birebir tekrarı korunur) | |
| benchmark | `deney_araclari/ab_karsilastir.py`: kare bazında eşleştirilmiş karar aracı + `--birebir` kontrolü | |

Not: `ucus_testi.py` yeni protokolde arama merkezi yedeğini zaten kapatıyordu; bu yüzden
PANO'daki uçuş sonuçlarının hiçbiri koruma değişikliğinden etkilenmez. Etkilenen tek şey,
ana betiğin `BENCHMARK=True` ile doğrudan çalıştırılmasıdır: o raporlar artık bulunamayan
kareye 0 m hata yazmaz (daha düşük ama dürüst sayılar).

## Kollar

| Kol | Ayar | Amaç |
|---|---|---|
| A (referans) | `NEAREST`, tek aday | Mevcut sonuçlar: `../2026-09-25_genis_dogrulama/sonuclar/ucus/` (2026-10-01) |
| `kontrol` | `NEAREST`, tek aday; 2 rota | Kod değişikliğinden sonra varsayılanın eski sonuçları **birebir** ürettiğinin kanıtı |
| `gsd_area` | `AREA` | Hipotez 1 |
| `coklu5` | `NEAREST`, top-5 | Hipotez 2 |
| `gsd_area_coklu5` | `AREA` + top-5 | Birlikte etkisi |

Modeller: `topluluk_x1_2022` (birincil, karar bu modelden) ve `RAW_kimlik` (yorum için).
Rotalar: 11 Ürgüp merkezi + Karlık 5/6/7 (14 rota, yaklaşık 2040 kare). guzergah_8 (kare
yamadan küçük) ve guzergah_9 (harita dışında) bu testte yok. Haritalar değişmez (aynı
`ucus_haritalari`), yalnız karenin işlenişi değişir.

## Karar kuralı (koşudan önce yazıldı)

Yalnız birincil model; ortak kareler; eşleştirilmiş McNemar, α = 0.05. GPS'i sabit ~35 m
kayık iki rota (`guzergah_1_donus`, `guzergah_2_tezde_ucus_1`) ≤25 m hesabında dışarıda,
≤70 m'de içeride.

- **KABUL:** tüm rotalarda ≤25 m farkı > 0 ve p < 0.05, **ve** hiçbir bölgede (tüm,
  Ürgüp merkezi, Karlık) ≤25 m ya da ≤70 m'de anlamlı düşüş yok.
- **RED:** herhangi bir bölgede ≤25 m ya da ≤70 m'de anlamlı düşüş.
- **BELİRSİZ:** diğerleri.

Birden fazla kol KABUL alırsa birincil ölçütte (tüm rotalar ≤25 m) en yüksek olan seçilir.
Kural `ab_karsilastir.py` içinde kodlu; sonuç `sonuclar/KARAR_<kol>.md`.

## Geri dönüş planı

1. **Şu an hiçbir varsayılan değişmedi.** Yeni kod yalnız bayrak açılınca devreye girer;
   `kontrol` kolu bunu ölçerek doğrular. Başarım bu adımda düşemez.
2. **RED veya BELİRSİZ:** hiçbir şey yapılmaz. `GSD_RESIZE_INTERP` `NEAREST` olarak kalır,
   bayrak kodda etkisiz durur.
3. **KABUL:** varsayılan **ayrı bir commit** ile değiştirilir (template-matching
   `RUN_CFG`). Sonradan başka bir testte (yeni bölge, ardışık mod, d2 öğrencisi) düşüş
   görülürse: `git revert <o commit>` ya da `RUN_CFG["GSD_RESIZE_INTERP"] = "NEAREST"`.
4. **Toptan geri dönüş:** üç depoda da `yedek/2026-10-06-oncesi` etiketi ve dokunulmamış
   `main` var; iyileştirmeler `iyilestirme/2026-10-06` dalında. `git switch main` her şeyi
   önceki hale getirir. Dalın ilk commit'i kullanıcının daha önce commit edilmemiş işidir;
   o iş kaybolmasın diye dal silinmemeli (gerekirse yalnız o commit main'e alınır).
5. **Koruma (GPS sızıntısı) değişikliği** bir başarım değil ölçüm düzeltmesidir; geri almak
   için `BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK=True`.

## Çalıştırma

Git Bash'te, GPU boşken (başka benchmark koşmuyorken):

```
bash /c/d_surucusu/visual_navigation/benchmark/deneyler/2026-10-06_tm_iyilestirme/kayitlar/ab_kos.sh
```

Yalnız bazı kollar: `ab_kos.sh kontrol gsd_area`. Devam ettirilebilir (sonucu olan iş atlanır).
Tahmini süre: kontrol ~10 dk, her tam kol ~1 saat (x1 ~32 dk + RAW ~23 dk), toplam ~3.5 saat.
İlerleme: `kayitlar/ab_kos.log`. Kontrol tutmazsa betik durur. Fark birkaç kareyse GPU
belirsizliği olabilir: `ab_kos.sh referans` ile referansı yeniden koşup
`A_KOK=<bu klasör>/sonuclar/ucus/referans ab_kos.sh gsd_area ...` ile devam edin.

## Açık işler (bu testin sonucuna bağlı)

- `AREA` kabul edilirse `d2_s1000` öğrencisi `NEAREST` girdilerle eğitildiği için uyumsuz
  kalır: verinin `--gsd-interp AREA` ile yeniden yakalanması ve ince ayar gerekir.
- Ardışık (BENCHMARK=False) modda aynı karşılaştırma.

## Sonuçlar

(koşudan sonra `sonuclar/KARAR_*.md` özetleri buraya)
