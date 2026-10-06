#!/usr/bin/env bash
# Kullanim: aday_test.sh <ad> <checkpoint_onek> <rota>...
# Ogrenciyi Keras topluluguna cevirir (modeller/aday/<ad>.h5) ve verilen rotalarda TM ucus testi yapar:
# kare modeli = aday, harita = x1'in uydu haritasi (Urgup ya da Karlik), DEM rotanin bolgesine gore.
# Sonuc: sonuclar/ucus/<ad>/<rota>.json ; kayit: kayitlar/aday_<ad>_<rota>.log
NAME=$1; PREFIX=$2; shift 2
BENCH=/c/d_surucusu/visual_navigation/benchmark
TM=/c/d_surucusu/visual_navigation/template-matching
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
D=$BENCH/deneyler/2026-10-02_ucus_uyarlama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$D/kayitlar/aday_test.log
w() { cygpath -w "$1"; }
H5=$D/modeller/aday/$NAME.h5
if [ ! -f "$H5" ]; then
  (cd "$BENCH/deney_araclari" && "$ENV/python.exe" npz_to_topluluk.py "$PREFIX"_uye*.npz --out "$(w "$H5")") \
    > "$D/kayitlar/aday_${NAME}_disa_aktar.log" 2>&1 || { echo "$(date '+%F %T') DISA AKTARMA COKTU $NAME" >> "$L"; exit 1; }
fi
cd "$BENCH" || exit 1
for r in "$@"; do
  case $r in
    guzergah_5_to_karain|guzergah_6_to_tagar|guzergah_7_to_gurlevek)
      map=$G/modeller/ucus_haritalari/topluluk_x1_2022_geo.tif; dem=$TM/karlik_30_cm_bingmap_utm_elevation.tif ;;
    *) map=$G/modeller/ucus_haritalari/topluluk_x1_2022_urgup_geo.tif; dem=$TM/ana_harita_urgup_30_cm_utm_elevation.tif ;;
  esac
  out=$D/sonuclar/ucus/$NAME/$r.json
  [ -f "$out" ] && continue
  echo "$(date '+%F %T') TEST $NAME $r" >> "$L"
  "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(w "$H5")" --harita "$(w "$map")" \
    --anlik "$(w "$TM/guzergahlar/$r")" --dem "$(w "$dem")" --out "$(w "$out")" > "$D/kayitlar/aday_${NAME}_$r.log" 2>&1
  grep "^UCUS" "$D/kayitlar/aday_${NAME}_$r.log" | sed "s/^/$NAME $r /" >> "$L" || echo "$(date '+%F %T') COKTU $NAME $r" >> "$L"
done
