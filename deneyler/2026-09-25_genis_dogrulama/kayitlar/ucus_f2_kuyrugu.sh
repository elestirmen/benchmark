#!/usr/bin/env bash
# Ucus testi F2: harita ve kareler modellerin EGITIM on islemesiyle (zero_1 + none).
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$G/kayitlar/ucus_kuyrugu.log
until grep -q "UCUS TAMAMLAMA BITTI" "$L" 2>/dev/null; do sleep 60; done
cd "$BENCH" || exit 1
for stem in yeni_topluluk3 yeni_tam_f32k3_e5 2022_tpu_f48k4_cb62 2022_k32_3_e1900; do
  m="$BENCH/deneyler/secili_modeller/${stem}.h5"
  map="$G/modeller/ucus_haritalari/${stem}_geo_z1.tif"
  out="$G/sonuclar/ucus/karlik_f2/${stem}.json"
  [ -f "$out" ] && continue
  echo "$(date '+%F %T') F2 HARITA $stem" >> "$L"
  [ -f "$map" ] || "$ENV/python.exe" -u deney_araclari/ucus_testi.py harita --protokol F2 --model "$(cygpath -w "$m")" --out-dir "$(cygpath -w "$G/modeller/ucus_haritalari")" >> "$L" 2>&1
  for attempt in 1 2; do
    [ -f "$out" ] && break
    echo "$(date '+%F %T') F2 TEST $stem ($attempt)" >> "$L"
    "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --protokol F2 --model "$(cygpath -w "$m")" --harita "$(cygpath -w "$map")" \
      --out "$(cygpath -w "$out")" > "$G/kayitlar/ucus_f2_${stem}.log" 2>&1
    grep "^UCUS" "$G/kayitlar/ucus_f2_${stem}.log" | sed 's/^/F2 /' >> "$L" || echo "$(date '+%F %T') F2 COKTU $stem" >> "$L"
  done
done
echo "$(date '+%F %T') UCUS F2 BITTI" >> "$L"
