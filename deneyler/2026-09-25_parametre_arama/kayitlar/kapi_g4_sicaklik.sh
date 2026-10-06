#!/usr/bin/env bash
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
K=$A/kayitlar
until grep -q "KAPI SIRALI BITTI" $K/kapi_sirali.log 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') BASLA g4_sicaklik" >> $K/kapi_sirali.log
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_g4_sicaklik_150.json" "$A"/modeller/sicaklik/*.h5 >> $K/kapi_sirali.log 2>&1
echo "$(date '+%F %T') BITTI g4_sicaklik cikis=$?" >> $K/kapi_sirali.log
