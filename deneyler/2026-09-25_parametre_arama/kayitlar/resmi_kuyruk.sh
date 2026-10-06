#!/usr/bin/env bash
# Resmi 1000 sorguluk kosular: ucus testleri bitince (bellek) resmi_adaylar.txt'deki adaylari SIRAYLA.
# Liste dosyasi satir satir okunur; kosu surerken sona aday eklenebilir.
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
L=$A/kayitlar/resmi_kuyruk.log
until grep -q "BITTI ucus" "$G/kayitlar/gece_kuyrugu.log" 2>/dev/null; do sleep 120; done
while read -r aday; do
  [ -z "$aday" ] && continue
  [ -f "$A/sonuclar/benchmark/urgup1000_$aday/summary_metadata.json" ] && { echo "$(date '+%F %T') ATLA $aday (var)" >> "$L"; continue; }
  echo "$(date '+%F %T') BASLA urgup1000_$aday" >> "$L"
  bash "$A/kayitlar/baslat_urgup1000.sh" "$aday" > "$A/kayitlar/urgup1000_$aday.log" 2>&1
  echo "$(date '+%F %T') BITTI urgup1000_$aday $(tail -1 "$A/kayitlar/urgup1000_$aday.log")" >> "$L"
done < "$A/kayitlar/resmi_adaylar.txt"
echo "$(date '+%F %T') RESMI KUYRUK BITTI" >> "$L"
