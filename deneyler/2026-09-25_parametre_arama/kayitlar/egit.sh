#!/usr/bin/env bash
# Kullanim: egit.sh <kosu_adi> [torch_flat_egitim.py argumanlari...]
# Cikti: deneyler/2026-09-25_parametre_arama/modeller/egitim/<kosu_adi>/ ; kayit: kayitlar/<kosu_adi>.log
NAME="$1"; shift
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
ENVT=/c/Users/ertugrul/anaconda3/envs/adaf
cd "$BENCH/deney_araclari" || exit 1
"$ENVT/python.exe" -u torch_flat_egitim.py --out-dir "$(cygpath -w "$A/modeller/egitim/$NAME")" \
  --filters 32 --kernel 3 --batch 1 --norm zero_1 --save-epochs 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,24 "$@"
echo "CIKIS_KODU=$?"
