#!/usr/bin/env bash
# k4 (TPU) TTA'si 1 px kayma duzeltmesiyle kuruldu; x1 (2022 toplulugu) TTA'si ile birlikte olc.
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
L=$A/kayitlar/ince_ayar.log
until grep -q "INCE AYAR KUYRUGU BITTI" "$L" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') BASLA g8_tta_k4" >> "$L"
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_g8_tta_k4_150.json" \
  "$A/modeller/tta/tta4_2022_tpu_cb62.h5" "$A/modeller/tta/tta4_x1_2022_tpu_k323_d6.h5" >> "$L" 2>&1
echo "$(date '+%F %T') BITTI g8_tta_k4 cikis=$?" >> "$L"
