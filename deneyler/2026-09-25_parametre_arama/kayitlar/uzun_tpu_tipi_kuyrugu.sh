#!/usr/bin/env bash
# TPU tipi uzun egitim hipotezi (2022 TPU f48_k4 serisi ~1100 epoch): buyuk batch + cok epoch.
# Makine bosken calisir; bitince tum checkpoint'ler (ham + EMA) 4 gruplu kapidan gecer.
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
L=$A/kayitlar/uzun_tpu_tipi.log
echo "$(date '+%F %T') BASLA u1c" >> "$L"
bash "$A/kayitlar/egit.sh" u1c_tpu_f48k4_b16 --seed 21 --filters 48 --kernel 4 --batch 16 --lr 1e-3 --epochs 300 --ema 0.999 --amp \
  --save-epochs 10,25,50,75,100,150,200,250,300 > "$A/kayitlar/u1c_tpu_f48k4_b16.log" 2>&1
echo "$(date '+%F %T') BITTI u1c $(tail -1 "$A/kayitlar/u1c_tpu_f48k4_b16.log")" >> "$L"
KAPI_GPU=1 bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_u1c_150.json" "$A/modeller/egitim/u1c_tpu_f48k4_b16:U1c" >> "$L" 2>&1
echo "$(date '+%F %T') BITTI kapi u1c cikis=$?" >> "$L"
