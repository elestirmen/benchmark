#!/usr/bin/env bash
# 4000 sorguluk dogrulama bitince agir isleri SIRAYLA calistirir (resmi benchmark'in yaninda tek is).
# 2026-09-25 bulgusu: lr dusurme / EMA / SWA / ince ayar BCE'yi dusurup eslestirmeyi bozuyor;
# iyi modeller batch-1'in gurultulu durumlari. Bu yuzden R5c yerine yalin tarifle seed piyangosu
# (her epoch kapidan gecer), sonra TPU tipi uzun egitim (u1b).
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
A=$BENCH/deneyler/2026-09-25_parametre_arama
L=$G/kayitlar/gece_kuyrugu.log
log() { echo "$(date '+%F %T') $*" >> "$L"; }
until grep -q "CIKIS_KODU" "$G/kayitlar/urgup4000_topluluk3.log" 2>/dev/null; do sleep 120; done
log "benchmark bitti: $(grep CIKIS_KODU "$G/kayitlar/urgup4000_topluluk3.log")"
until grep -q "BITTI g8_tta_k4" "$A/kayitlar/ince_ayar.log" 2>/dev/null; do sleep 60; done
log "BASLA ucus tamamlama + F2"
bash "$G/kayitlar/ucus_eksikleri_tamamla.sh"
bash "$G/kayitlar/ucus_f2_kuyrugu.sh"
log "BITTI ucus"
for s in 12 13 14; do
  log "BASLA piyango s$s"
  bash "$A/kayitlar/egit.sh" p_s$s --seed $s --epochs 12 > "$A/kayitlar/p_s$s.log" 2>&1
  bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_piyango_s${s}_150.json" "$A/modeller/egitim/p_s$s:P$s" >> "$L" 2>&1
  log "BITTI piyango s$s kapi_cikis=$?"
done
log "BASLA u1b (TPU tipi uzun)"
bash "$A/kayitlar/egit.sh" u1b_tpu_f48k4_b16 --seed 21 --filters 48 --kernel 4 --batch 16 --lr 1e-3 --epochs 600 --ema 0.999 --amp \
  --save-epochs 10,25,50,75,100,150,200,250,300,400,500,600 > "$A/kayitlar/u1b_tpu_f48k4_b16.log" 2>&1
log "BITTI u1b $(tail -1 "$A/kayitlar/u1b_tpu_f48k4_b16.log")"
log "GECE KUYRUGU BITTI"
