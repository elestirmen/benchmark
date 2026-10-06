#!/usr/bin/env bash
# Gece kuyrugu (devam): s14 egitimi bitince Adana sagalamlik kontrolu, s14 kapisi, TPU tipi uzun egitim.
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
A=$BENCH/deneyler/2026-09-25_parametre_arama
L=$G/kayitlar/gece_kuyrugu.log
log() { echo "$(date '+%F %T') $*" >> "$L"; }
until grep -q "CIKIS_KODU" "$A/kayitlar/p_s14.log" 2>/dev/null; do sleep 60; done
log "BITTI egitim s14 $(tail -1 "$A/kayitlar/p_s14.log")"
log "BASLA adana"
bash "$G/kayitlar/baslat_adana.sh" > "$G/kayitlar/adana100_modeller.log" 2>&1
log "BITTI adana $(tail -1 "$G/kayitlar/adana100_modeller.log")"
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_piyango_s14_150.json" "$A/modeller/egitim/p_s14:P14" >> "$L" 2>&1
log "BITTI piyango s14 kapi_cikis=$?"
log "BASLA u1b (TPU tipi uzun)"
bash "$A/kayitlar/egit.sh" u1b_tpu_f48k4_b16 --seed 21 --filters 48 --kernel 4 --batch 16 --lr 1e-3 --epochs 600 --ema 0.999 --amp \
  --save-epochs 10,25,50,75,100,150,200,250,300,400,500,600 > "$A/kayitlar/u1b_tpu_f48k4_b16.log" 2>&1
log "BITTI u1b $(tail -1 "$A/kayitlar/u1b_tpu_f48k4_b16.log")"
log "GECE KUYRUGU BITTI"
