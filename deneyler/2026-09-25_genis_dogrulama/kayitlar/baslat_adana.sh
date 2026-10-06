#!/usr/bin/env bash
# Adana capraz-kaynak benchmark'i (Google sorgu / Bing harita, iki yon): Urgup ve Bern disi bolge.
# Rasterlar Web Mercator 0.597 m/px (~0.48 m zemin) -> UTM 36N 0.298582 m/px (raster_utm.py);
# iki kaynak arasi kayma ~2 m (9 bolgede NCC). Alan ~4.4 x 4.5 km: global ~ Urgup roi 2000 m.
set -u
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache'
export CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
cd "$BENCH" || exit 1
W() { cygpath -w "$1"; }
"$ENV/python.exe" -u geospatial_model_benchmark.py \
  --query-raster "$(W $G/veri/adana_gmap_utm_30cm.tif)" --map-raster "$(W $G/veri/adana_bingmap_utm_30cm.tif)" \
  --model-dir "$(W $G/modeller/adana_modelleri)" --model-sampling full \
  --include-raw --include-models --bidirectional --query-variants clean,hard_v1 \
  --search-modes roi500,roi1000,global \
  --tile-size 544 --overlap 32 --crop-border 16 --block-size-m 1000 --samples-per-block 6 \
  --max-queries 100 --query-sampling balanced_exact --seed 42 \
  --min-query-std 12 --min-query-entropy 4 --max-dark-fraction 0.20 \
  --pyramid-factors 16,8,4,2,1 --top-k 30 --refine-radius-px 160 --nms-radius-px 256 \
  --batch-size 8 --search-workers 8 --bootstrap-iterations 1000 \
  --normalization minus1_1 --output-value-mode auto --enhancement none \
  --excel-engine openpyxl --excel-update model --excel-report --no-results-csv \
  --output-root "$(W $G/sonuclar/benchmark)" --run-id adana100_modeller --cleanup-maps --isolate-models
echo "CIKIS_KODU=$?"
