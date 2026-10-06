#!/usr/bin/env bash
# 4000 sorguluk Urgup protokolu (2022 kazananlariyla ayni sorgular: seed 42, balanced_exact).
# RAW bu sorgularda outputs/urgup_cevresi_models_4000 icinde zaten olculdu -> --no-include-raw.
# Kullanim: baslat_urgup4000.sh <aday_adi>   (modeller/<aday_adi>/ icindeki .h5 dosyalari)
set -u
ADAY="$1"
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache'
export CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
cd "$BENCH" || exit 1
"$ENV/python.exe" -u run_urgup_cevresi_models.py \
  --model-dir "$(cygpath -w "$G/modeller/$ADAY")" \
  --output-dir "$(cygpath -w "$G/sonuclar/benchmark/urgup4000_$ADAY")" \
  --max-queries 4000 --no-include-raw
echo "CIKIS_KODU=$?"
