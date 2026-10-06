#!/usr/bin/env bash
# 4 gruplu kapinin (G2B/B2G x clean/hard_v1) resmi sonuclarla uyumunu olcer.
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
BENCH=/c/d_surucusu/visual_navigation/benchmark
cd "$BENCH" || exit 1
ARGS=(--model-dir models)
for f in models_pipeline/20260909_223425/*/_benchmark_adayi/*.h5 \
         deneyler/2026-09-24_zero1_tarif/modeller/adaylar/*/*.h5 \
         deneyler/pipeline_20260924_065751/modeller/z1_f32_k3_b1/_benchmark_adayi/*.h5; do
  ARGS+=(--model "$f")
done
"$ENV/python.exe" -u margin_gate.py --samples 150 --aoi train --normalization minus1_1 \
  --bidirectional --variants clean,hard_v1 \
  --json-out deneyler/2026-09-25_parametre_arama/sonuclar/kapi/kapi4_kalibrasyon_150.json "${ARGS[@]}"
echo "CIKIS_KODU=$?"
