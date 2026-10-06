#!/usr/bin/env bash
# Karlik ucus testi (guzergah_5_to_karain), protokol F1: harita ve kareler minus1_1 + none.
# Benchmark harita donusumu bitince baslar (GPU cakismasini azaltmak icin).
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$G/kayitlar/ucus_kuyrugu.log
until grep -q "(%99)" "$G/kayitlar/urgup4000_topluluk3.log" 2>/dev/null; do sleep 60; done; sleep 180
cd "$BENCH" || exit 1
[ -f "$G/modeller/ucus/RAW_kimlik.h5" ] || "$ENV/python.exe" deney_araclari/ucus_testi.py kimlik --out "$(cygpath -w "$G/modeller/ucus/RAW_kimlik.h5")" >> "$L" 2>&1
MODELS=(
  "$G/modeller/ucus/RAW_kimlik.h5"
  "$BENCH/deneyler/secili_modeller/yeni_topluluk3.h5"
  "$BENCH/deneyler/secili_modeller/2022_tpu_f48k4_cb62.h5"
  "$BENCH/deneyler/secili_modeller/yeni_pipeline_z1_f32k3_e9.h5"
  "$BENCH/deneyler/secili_modeller/2022_k32_3_e1900.h5"
  "$BENCH/deneyler/secili_modeller/yeni_tam_f32k3_e5.h5"
  "$BENCH/deneyler/secili_modeller/2022_deneme6_f32k5.h5"
  "$BENCH/deneyler/secili_modeller/2022_tpu_f48k4_e1100.h5"
)
for m in "${MODELS[@]}"; do
  stem=$(basename "$m" .h5)
  map="$G/modeller/ucus_haritalari/${stem}_geo.tif"
  echo "$(date '+%F %T') HARITA $stem" >> "$L"
  [ -f "$map" ] || "$ENV/python.exe" -u deney_araclari/ucus_testi.py harita --model "$(cygpath -w "$m")" --out-dir "$(cygpath -w "$G/modeller/ucus_haritalari")" >> "$L" 2>&1
  echo "$(date '+%F %T') TEST $stem" >> "$L"
  "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(cygpath -w "$m")" --harita "$(cygpath -w "$map")" \
    --out "$(cygpath -w "$G/sonuclar/ucus/karlik_f1/${stem}.json")" > "$G/kayitlar/ucus_${stem}.log" 2>&1
  grep "^UCUS" "$G/kayitlar/ucus_${stem}.log" >> "$L"
done
echo "$(date '+%F %T') UCUS KUYRUGU BITTI" >> "$L"
