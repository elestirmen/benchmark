#!/usr/bin/env bash
# Resmi benchmark kuyrugu: onceki 4000'luk kosu bitince adaylari SIRAYLA kosturur
# (iki benchmark ayni anda ~23 GB commit x2 ile 0xC0000005 cokmesi veriyordu).
# Kullanim: benchmark_kuyrugu.sh <aday1> [<aday2> ...]   (modeller/<aday>/ icinde tek .h5)
G=/c/d_surucusu/visual_navigation/benchmark/deneyler/2026-09-25_genis_dogrulama
K=$G/kayitlar
until grep -q "CIKIS_KODU" "$K/urgup4000_topluluk3.log" 2>/dev/null; do sleep 120; done
for aday in "$@"; do
  echo "$(date '+%F %T') BASLA urgup4000_$aday" >> "$K/benchmark_kuyrugu.log"
  bash "$K/baslat_urgup4000.sh" "$aday" > "$K/urgup4000_$aday.log" 2>&1
  echo "$(date '+%F %T') BITTI urgup4000_$aday $(tail -1 "$K/urgup4000_$aday.log")" >> "$K/benchmark_kuyrugu.log"
done
