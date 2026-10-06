#!/usr/bin/env bash
# Tum guzergahlarda ucus testi (F1: minus1_1 + none), 5 temel model.
# Urgup merkezi rotalari (11): urgup_bingmap_utm_30_cm.tif + urgup UTM DEM (egitim bolgesiyle iliskili olabilir).
# Karlik rotalari (6, 7, 8): karlik haritasi + karlik DEM. guzergah_5 karlik_f1'de var; guzergah_9 haritanin disinda.
# Rota sirasi: her rota tum modellerde bitince sonrakine gecer. Sonucu olan isler atlanir (devam ettirilebilir).
BENCH=/c/d_surucusu/visual_navigation/benchmark
TM=/c/d_surucusu/visual_navigation/template-matching
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$G/kayitlar/ucus_rotalar.log
LD=$G/kayitlar/ucus_rotalar
HD=$G/modeller/ucus_haritalari
mkdir -p "$LD"
cd "$BENCH" || exit 1
w() { cygpath -w "$1"; }

MODELS=(
  "$BENCH/deneyler/secili_modeller/topluluk_x1_2022.h5"
  "$BENCH/deneyler/secili_modeller/2022_deneme6_f32k5.h5"
  "$G/modeller/ucus/RAW_kimlik.h5"
  "$BENCH/deneyler/secili_modeller/2022_k32_3_e1900.h5"
  "$BENCH/deneyler/secili_modeller/2022_tpu_f48k4_cb62.h5"
)
URGUP_SRC=$BENCH/urgup_bingmap_utm_30_cm.tif
URGUP_DEM=$TM/ana_harita_urgup_30_cm_utm_elevation.tif
KARLIK_DEM=$TM/karlik_30_cm_bingmap_utm_elevation.tif
URGUP_ROUTES=(1_tezde_ucus_5 2_tezde_ucus_6 3_tezde_ucus_7 4_tezde_ucus_8 5_tezde_ucus_9 6_tezde_ucus_4
  guzergah_1 guzergah_1_donus guzergah_2_tezde_ucus_1 guzergah_3_tezde_ucus_2 guzergah_4_tezde_ucus_3)
KARLIK_ROUTES=(guzergah_6_to_tagar guzergah_7_to_gurlevek guzergah_8_koy_vadi)

echo "$(date '+%F %T') ROTALAR BASLADI" >> "$L"
for m in "${MODELS[@]}"; do
  stem=$(basename "$m" .h5)
  [ -f "$HD/${stem}_urgup_geo.tif" ] && continue
  echo "$(date '+%F %T') HARITA urgup $stem" >> "$L"
  "$ENV/python.exe" -u deney_araclari/ucus_testi.py harita --model "$(w "$m")" --out-dir "$(w "$HD")" \
    --kaynak "$(w "$URGUP_SRC")" --etiket urgup > "$LD/harita_urgup_${stem}.log" 2>&1
  grep "^HARITA" "$LD/harita_urgup_${stem}.log" >> "$L" || echo "$(date '+%F %T') HARITA COKTU urgup $stem" >> "$L"
done

run_route() {  # $1 rota, $2 harita eki (_urgup_geo.tif | _geo.tif), $3 DEM
  local rota=$1 suffix=$2 dem=$3 out_dir=$G/sonuclar/ucus/rota_$1
  mkdir -p "$out_dir"
  for m in "${MODELS[@]}"; do
    local stem; stem=$(basename "$m" .h5)
    local map=$HD/${stem}${suffix} out=$out_dir/${stem}.json log=$LD/${rota}_${stem}.log
    [ -f "$out" ] && continue
    [ -f "$map" ] || { echo "$(date '+%F %T') HARITA YOK $rota $stem" >> "$L"; continue; }
    echo "$(date '+%F %T') TEST $rota $stem" >> "$L"
    "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(w "$m")" --harita "$(w "$map")" \
      --anlik "$(w "$TM/guzergahlar/$rota")" --dem "$(w "$dem")" --out "$(w "$out")" > "$log" 2>&1
    grep "^UCUS" "$log" | sed "s/^/$rota /" >> "$L" || echo "$(date '+%F %T') COKTU $rota $stem" >> "$L"
  done
}
for r in "${URGUP_ROUTES[@]}"; do run_route "$r" _urgup_geo.tif "$URGUP_DEM"; done
for r in "${KARLIK_ROUTES[@]}"; do run_route "$r" _geo.tif "$KARLIK_DEM"; done
echo "$(date '+%F %T') ROTALAR BITTI" >> "$L"
