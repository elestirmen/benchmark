#!/usr/bin/env bash
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
until grep -q "BITTI kapi topluluk varyantlari" "$A/kayitlar/sabah_kuyrugu.log" 2>/dev/null; do sleep 60; done
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_g9b_x4_150.json" "$A/modeller/topluluklar/x4_cb62_bdb7_k323_d6.h5" > "$A/kayitlar/kapi_x4.log" 2>&1
echo "BITTI kapi x4 cikis=$?" >> "$A/kayitlar/kapi_x4.log"
