#!/usr/bin/env bash
# TM iyilestirme A/B kosusu (2026-10-06). Ayrinti ve karar kurali: ../BENIOKU.md
# Kullanim (Git Bash):
#   bash /c/d_surucusu/visual_navigation/benchmark/deneyler/2026-10-06_tm_iyilestirme/kayitlar/ab_kos.sh [kol ...]
# Kollar (varsayilan: hepsi, bu sirayla): kontrol gsd_area coklu5 gsd_area_coklu5
#   referans: mevcut varsayilani 14 rotada bastan kosar (kontrol GPU belirsizligi yuzunden
#   tutmazsa); sonra A_KOK=.../sonuclar/ucus/referans ile A/B'yi buna karsi yapin.
# Devam ettirilebilir: sonucu olan isler atlanir. Baska bir benchmark ile ayni anda kosturmayin.
set -u
BENCH=/c/d_surucusu/visual_navigation/benchmark
TM=/c/d_surucusu/visual_navigation/template-matching
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
D=$BENCH/deneyler/2026-10-06_tm_iyilestirme
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$D/kayitlar/ab_kos.log
LD=$D/kayitlar/ab_kos
HD=$G/modeller/ucus_haritalari
A=${A_KOK:-$G/sonuclar/ucus}   # referans kol: mevcut varsayilan (NEAREST, tek aday), 2026-10-01 kosusu
mkdir -p "$LD"
cd "$BENCH" || exit 1
w() { cygpath -w "$1"; }

MODELS=(
  "$BENCH/deneyler/secili_modeller/topluluk_x1_2022.h5"   # birincil (karar bu modelden)
  "$G/modeller/ucus/RAW_kimlik.h5"                         # yorum icin: modelsiz eslestirme
)
URGUP_DEM=$TM/ana_harita_urgup_30_cm_utm_elevation.tif
KARLIK_DEM=$TM/karlik_30_cm_bingmap_utm_elevation.tif
URGUP_ROUTES=(1_tezde_ucus_5 2_tezde_ucus_6 3_tezde_ucus_7 4_tezde_ucus_8 5_tezde_ucus_9 6_tezde_ucus_4
  guzergah_1 guzergah_1_donus guzergah_2_tezde_ucus_1 guzergah_3_tezde_ucus_2 guzergah_4_tezde_ucus_3)
KARLIK_ROUTES=(guzergah_5_to_karain guzergah_6_to_tagar guzergah_7_to_gurlevek)
KONTROL_ROUTES=(6_tezde_ucus_4 guzergah_6_to_tagar)

kol_args() {
  case $1 in
    kontrol|referans) echo "" ;;
    gsd_area) echo "--gsd-interp AREA" ;;
    coklu5) echo "--coklu-aday 5" ;;
    gsd_area_coklu5) echo "--gsd-interp AREA --coklu-aday 5" ;;
    *) return 1 ;;
  esac
}

run_one() {  # $1 kol, $2 rota
  local kol=$1 rota=$2 suffix dem
  case " ${KARLIK_ROUTES[*]} " in
    *" $rota "*) suffix=_geo.tif; dem=$KARLIK_DEM ;;
    *) suffix=_urgup_geo.tif; dem=$URGUP_DEM ;;
  esac
  local out_dir=$D/sonuclar/ucus/$kol/rota_$rota
  mkdir -p "$out_dir"
  for m in "${MODELS[@]}"; do
    local stem map out log
    stem=$(basename "$m" .h5); map=$HD/${stem}${suffix}; out=$out_dir/${stem}.json; log=$LD/${kol}_${rota}_${stem}.log
    [ -f "$out" ] && continue
    [ -f "$map" ] || { echo "$(date '+%F %T') HARITA YOK $kol $rota $stem" >> "$L"; continue; }
    echo "$(date '+%F %T') TEST $kol $rota $stem" >> "$L"
    # shellcheck disable=SC2046
    "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(w "$m")" --harita "$(w "$map")" \
      --anlik "$(w "$TM/guzergahlar/$rota")" --dem "$(w "$dem")" --out "$(w "$out")" $(kol_args "$kol") > "$log" 2>&1
    grep "^UCUS" "$log" | sed "s/^/$kol $rota /" >> "$L" || echo "$(date '+%F %T') COKTU $kol $rota $stem" >> "$L"
  done
}

KOLLAR=("$@")
[ ${#KOLLAR[@]} -eq 0 ] && KOLLAR=(kontrol gsd_area coklu5 gsd_area_coklu5)
echo "$(date '+%F %T') AB BASLADI: ${KOLLAR[*]} | TM $(git -C "$TM" rev-parse --short HEAD) | benchmark $(git -C "$BENCH" rev-parse --short HEAD)" >> "$L"
for kol in "${KOLLAR[@]}"; do
  kol_args "$kol" > /dev/null || { echo "bilinmeyen kol: $kol" | tee -a "$L"; exit 2; }
  if [ "$kol" = kontrol ]; then
    for r in "${KONTROL_ROUTES[@]}"; do run_one kontrol "$r"; done
    if ! "$ENV/python.exe" deney_araclari/ab_karsilastir.py --a "$(w "$A")" --b "$(w "$D/sonuclar/ucus/kontrol")" --birebir >> "$L" 2>&1; then
      echo "$(date '+%F %T') KONTROL BASARISIZ: varsayilan ayarlar eski sonuclari birebir uretmedi; A/B durduruldu" >> "$L"
      exit 1
    fi
    continue
  fi
  for r in "${URGUP_ROUTES[@]}" "${KARLIK_ROUTES[@]}"; do run_one "$kol" "$r"; done
  "$ENV/python.exe" deney_araclari/ab_karsilastir.py --a "$(w "$A")" --b "$(w "$D/sonuclar/ucus/$kol")" --b-adi "$kol" \
    --rapor "$(w "$D/sonuclar/KARAR_$kol.md")" > "$LD/karar_$kol.log" 2>&1
  grep "^KARAR" "$LD/karar_$kol.log" >> "$L" || echo "$(date '+%F %T') KARAR HESAPLANAMADI $kol" >> "$L"
done
echo "$(date '+%F %T') AB BITTI" >> "$L"
