#!/usr/bin/env bash
# Karlik ucus testi, protokol F1 (minus1_1 + none): x1 = 2022 TPU cb62 + k_32_3 e1900 + deneme6 toplulugu.
# Uydu benchmark'inin sampiyonu (Urgup 4000: 0.718); gercek IHA karelerinde de onde mi?
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$G/kayitlar/ucus_kuyrugu.log
cd "$BENCH" || exit 1
m="$BENCH/deneyler/secili_modeller/topluluk_x1_2022.h5"
stem=$(basename "$m" .h5)
map="$G/modeller/ucus_haritalari/${stem}_geo.tif"
echo "$(date '+%F %T') HARITA $stem" >> "$L"
[ -f "$map" ] || "$ENV/python.exe" -u deney_araclari/ucus_testi.py harita --model "$(cygpath -w "$m")" --out-dir "$(cygpath -w "$G/modeller/ucus_haritalari")" >> "$L" 2>&1
[ -f "$map" ] || { echo "$(date '+%F %T') HARITA COKTU $stem" >> "$L"; exit 1; }
echo "$(date '+%F %T') TEST $stem" >> "$L"
"$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(cygpath -w "$m")" --harita "$(cygpath -w "$map")" \
  --out "$(cygpath -w "$G/sonuclar/ucus/karlik_f1/${stem}.json")" > "$G/kayitlar/ucus_${stem}.log" 2>&1
grep "^UCUS" "$G/kayitlar/ucus_${stem}.log" >> "$L" || echo "$(date '+%F %T') COKTU $stem" >> "$L"
echo "$(date '+%F %T') UCUS X1 BITTI" >> "$L"
