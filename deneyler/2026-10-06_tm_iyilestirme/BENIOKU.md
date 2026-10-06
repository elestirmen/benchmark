# TM iyileştirmeleri: A/B testi ve geri dönüş planı (2026-10-06)

Soru: template-matching'in uçuş hattında üç ucuz değişiklik gerçek İHA karelerinde
başarımı artırıyor mu?

1. **Yaw kaynağı (en güçlü aday).** TM kareyi kuzeye döndürürken DJI MakerNote
   `FlightDegree`'yi (gövde yawı) kullanıyordu; nadir karenin yönünü ise gimbal (kamera)
   yawı belirler. EXIF tanısı (`sonuclar/exif_tani.md`, GPU gerektirmez,
   `deney_araclari/exif_tani.py`): 16 rotada gimbal eksi uçuş yawı rota içinde neredeyse
   sabit, **-1.5 ile -13 derece** arasında; 11 rotada karelerin çoğu 5 dereceden fazla
   sapıyor. 4. rotanın (4_tezde_ucus_8) 2026-06-11'de elle bulunan -8.5 derecelik profil
   ofseti, ölçülen farkla (-8.3) örtüşüyor: o düzeltme aslında bu hatayı yakalamıştı.
   512 px'lik şablonda 8 derecelik dönüklük köşelerde yaklaşık 50 px kayma demek.
2. **GSD küçültmesinde interpolasyon.** Kare, harita çözünürlüğüne (29.85 cm/px) inerken
   `cv2.INTER_NEAREST` ile küçültülüyordu. Bu rotalar kalkışa göre ~300-500 m'de uçtuğu
   için kare ~2-4 kat küçülüyor (daha alçak uçuşta oran büyür); en yakın komşu piksel
   atlayarak örtüşme (aliasing) üretir. Referans harita alan ortalamalı bir ortomozaiktir,
   benchmark piramidi de `INTER_AREA` kullanır.
3. **Çok adaylı kaba arama.** Benchmark top-30 aday + NMS kullanıyor; uçuş testinde
   `PYRAMID_MULTI_CANDIDATE` kapalıydı (tek kaba tepe). Kendine benzer arazide yanlış
   kaba tohuma kilitlenmeyi azaltabilir.

Aynı tanının yan bulgusu: rota içinde std(GPSAltitude - RelativeAltitude) 0.3-5.9 m, yani
irtifadan gelen ölçek gürültüsü %2'nin altında. İrtifa kaynağını değiştirmek öncelikli değil.

## Koddaki değişiklikler (varsayılanlar eski davranışı korur)

| Depo | Değişiklik | Varsayılan |
|---|---|---|
| template-matching | `RUN_CFG["YAW_SOURCE"]`: `FLIGHT` / `GIMBAL`; `GIMBAL` iken rota profillerindeki `YAW_OFFSET_DEG` uygulanmaz (çift düzeltme olmasın) | `FLIGHT` (önceki davranış) |
| template-matching | `RUN_CFG["GSD_RESIZE_INTERP"]`: `NEAREST` / `AREA` / `LINEAR` / `CUBIC` / `AUTO`; merkez, sol-üst ve sağ-alt ölçeklerinin dördü de buna bağlı | `NEAREST` (önceki davranış) |
| template-matching | `BENCHMARK=True` iken "kesişim yok" ve "düşük skor" arama merkezi yedekleri kendiliğinden kapanır (arama merkezi = EXIF GPS, yani GPS sızıntısı). `BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK=True` eski davranışı geri getirir | kapalı (sızıntı yok) |
| benchmark | `ucus_testi.py test --yaw-kaynagi GIMBAL`, `--gsd-interp AREA`, `--coklu-aday K` | `FLIGHT`, `NEAREST`, `0` (önceki davranış) |
| benchmark | `ucus_testi.py --eski-protokol` artık `BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK=True` verir (2026-09-12 testinin birebir tekrarı korunur) | |
| benchmark | `deney_araclari/ab_karsilastir.py`: kare bazında eşleştirilmiş karar aracı + `--birebir` kontrolü; `deney_araclari/exif_tani.py` | |

Not: `ucus_testi.py` yeni protokolde arama merkezi yedeğini zaten kapatıyordu; bu yüzden
PANO'daki uçuş sonuçlarının hiçbiri koruma değişikliğinden etkilenmez. Etkilenen tek şey,
ana betiğin `BENCHMARK=True` ile doğrudan çalıştırılmasıdır: o raporlar artık bulunamayan
kareye 0 m hata yazmaz (daha düşük ama dürüst sayılar).

## Kollar

| Kol | Ayar | Amaç |
|---|---|---|
| A (referans) | `FLIGHT`, `NEAREST`, tek aday | Mevcut sonuçlar: `../2026-09-25_genis_dogrulama/sonuclar/ucus/` (2026-10-01) |
| `kontrol` | A ile aynı; 2 rota | Kod değişikliğinden sonra varsayılanın eski sonuçları **birebir** ürettiğinin kanıtı |
| `gimbal_yaw` | `GIMBAL` | Hipotez 1 |
| `gsd_area` | `AREA` | Hipotez 2 |
| `coklu5` | top-5 aday | Hipotez 3 |
| `birlesik` | `GIMBAL` + `AREA` + top-5 | Birlikte etkisi |

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

Birden fazla tekil kol KABUL alırsa varsayılana yalnız KABUL alanlar girer; `birlesik` kol
da KABUL ise ve en iyi tekil koldan iyiyse hepsi birlikte girer. Kural `ab_karsilastir.py`
içinde kodlu; sonuç `sonuclar/KARAR_<kol>.md`.

## Geri dönüş planı

1. **Şu an hiçbir varsayılan değişmedi.** Yeni kod yalnız bayrak açılınca devreye girer;
   `kontrol` kolu bunu ölçerek doğrular. Başarım bu adımda düşemez.
2. **RED veya BELİRSİZ:** hiçbir şey yapılmaz; ilgili bayrak eski değerinde kalır ve kodda
   etkisiz durur.
3. **KABUL:** varsayılan **ayrı bir commit** ile değiştirilir (template-matching `RUN_CFG`).
   Sonradan başka bir testte (yeni bölge, ardışık mod, d2 öğrencisi) düşüş görülürse:
   `git revert <o commit>` ya da `RUN_CFG`'de eski değere (`FLIGHT` / `NEAREST` / `False`) dönüş.
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

Yalnız bazı kollar: `ab_kos.sh kontrol gimbal_yaw`. Devam ettirilebilir (sonucu olan iş
atlanır). Tahmini süre: kontrol ~10 dk, her tam kol ~1 saat (x1 ~32 dk + RAW ~23 dk), beş
kol toplam ~4-4.5 saat. Öncelik sırası `kontrol`, `gimbal_yaw`; zaman kısıtlıysa önce bu
ikisini koşun. İlerleme: `kayitlar/ab_kos.log`. Kontrol tutmazsa betik durur. Fark birkaç
kareyse GPU belirsizliği olabilir: `ab_kos.sh referans` ile referansı yeniden koşup
`A_KOK=<bu klasör>/sonuclar/ucus/referans ab_kos.sh gimbal_yaw ...` ile devam edin.

## Açık işler (bu testin sonucuna bağlı)

- Yaw ya da interpolasyon değişirse `d2_s1000` öğrencisi eski girdilerle eğitildiği için
  uyumsuz kalır: verinin yeni ayarla yeniden yakalanması ve ince ayar gerekir.
- `GIMBAL` kabul edilirse 4. rotanın `YAW_OFFSET_DEG` profili zaten devre dışı kalır;
  kalıcı olarak silinmesi ayrı bir karar.
- Ardışık (BENCHMARK=False) modda aynı karşılaştırma.

## Sonuçlar

(koşudan sonra `sonuclar/KARAR_*.md` özetleri buraya)
