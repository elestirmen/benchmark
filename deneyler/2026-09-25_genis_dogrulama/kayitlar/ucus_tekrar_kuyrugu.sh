#!/usr/bin/env bash
# Kesisimsiz karelerde GPS merkezi yedegi kapatildiktan sonra, haritasi hazir modellerin
# uçuş testlerini yeniden kosturur (ilk kuyruk bittikten sonra).
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$G/kayitlar/ucus_kuyrugu.log
until grep -q "UCUS KUYRUGU BITTI" "$L" 2>/dev/null; do sleep 60; done
cd "$BENCH" || exit 1
for m in "$G/modeller/ucus/RAW_kimlik.h5" "$BENCH/deneyler/secili_modeller/yeni_topluluk3.h5" "$BENCH/deneyler/secili_modeller/2022_tpu_f48k4_cb62.h5" \
         "$BENCH/deneyler/secili_modeller/yeni_pipeline_z1_f32k3_e9.h5" "$BENCH/deneyler/secili_modeller/2022_k32_3_e1900.h5"; do
  stem=$(basename "$m" .h5); map="$G/modeller/ucus_haritalari/${stem}_geo.tif"
  out="$G/sonuclar/ucus/karlik_f1/${stem}.json"
  [ -f "$out" ] && continue
  echo "$(date '+%F %T') TEST $stem (tekrar)" >> "$L"
  "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(cygpath -w "$m")" --harita "$(cygpath -w "$map")" \
    --out "$(cygpath -w "$out")" > "$G/kayitlar/ucus_${stem}.log" 2>&1
  grep "^UCUS" "$G/kayitlar/ucus_${stem}.log" >> "$L"
done
echo "$(date '+%F %T') UCUS TEKRAR BITTI" >> "$L"
