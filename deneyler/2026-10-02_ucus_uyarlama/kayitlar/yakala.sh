#!/usr/bin/env bash
# x1 (kare modeli) + x1 haritasi ile tum rotalarda kare basina egitim verisi yakalar.
# Cikti: veri/<rota>/kare_XXXX.npz (+ veri/<rota>.json: ayni kosunun ucus sonucu).
BENCH=/c/d_surucusu/visual_navigation/benchmark
TM=/c/d_surucusu/visual_navigation/template-matching
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
D=$BENCH/deneyler/2026-10-02_ucus_uyarlama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$D/kayitlar/yakala.log
cd "$BENCH" || exit 1
w() { cygpath -w "$1"; }
M=$BENCH/deneyler/secili_modeller/topluluk_x1_2022.h5
URGUP=(1_tezde_ucus_5 2_tezde_ucus_6 3_tezde_ucus_7 4_tezde_ucus_8 5_tezde_ucus_9 6_tezde_ucus_4
  guzergah_1 guzergah_1_donus guzergah_2_tezde_ucus_1 guzergah_3_tezde_ucus_2 guzergah_4_tezde_ucus_3)
KARLIK=(guzergah_5_to_karain guzergah_6_to_tagar guzergah_7_to_gurlevek)
yakala() {  # $1 rota $2 harita $3 DEM
  [ -f "$D/veri/$1.json" ] && return
  echo "$(date '+%F %T') YAKALA $1" >> "$L"
  "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(w "$M")" --harita "$(w "$2")" \
    --anlik "$(w "$TM/guzergahlar/$1")" --dem "$(w "$3")" --out "$(w "$D/veri/$1.json")" \
    --yakala "$(w "$D/veri/$1")" > "$D/kayitlar/yakala_$1.log" 2>&1
  grep "^UCUS" "$D/kayitlar/yakala_$1.log" | sed "s/^/$1 /" >> "$L" || echo "$(date '+%F %T') COKTU $1" >> "$L"
}
for r in "${URGUP[@]}"; do yakala "$r" "$G/modeller/ucus_haritalari/topluluk_x1_2022_urgup_geo.tif" "$TM/ana_harita_urgup_30_cm_utm_elevation.tif"; done
for r in "${KARLIK[@]}"; do yakala "$r" "$G/modeller/ucus_haritalari/topluluk_x1_2022_geo.tif" "$TM/karlik_30_cm_bingmap_utm_elevation.tif"; done
echo "$(date '+%F %T') YAKALA BITTI" >> "$L"
