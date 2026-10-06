#!/usr/bin/env bash
# 1000 sorguluk Urgup protokolu (2026-09-24 yeni modelleriyle ayni sorgular: seed 42, balanced_exact).
# RAW bu sorgularda olculdu (0.461) -> --no-include-raw.
# Kullanim: baslat_urgup1000.sh <aday>   (modeller/adaylar/<aday>/ icindeki .h5)
set -u
ADAY="$1"
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache'
export CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
cd "$BENCH" || exit 1
"$ENV/python.exe" -u run_urgup_cevresi_models.py \
  --model-dir "$(cygpath -w "$A/modeller/adaylar/$ADAY")" \
  --output-dir "$(cygpath -w "$A/sonuclar/benchmark/urgup1000_$ADAY")" \
  --max-queries 1000 --no-include-raw --batch-size 8
echo "CIKIS_KODU=$?"
