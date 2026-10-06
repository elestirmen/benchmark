#!/usr/bin/env bash
# Haritasi hazir olup sonucu eksik (cokmus) ucus testlerini tamamlar; en fazla 2 deneme.
BENCH=/c/d_surucusu/visual_navigation/benchmark
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache' CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
L=$G/kayitlar/ucus_kuyrugu.log
until grep -q "UCUS TEKRAR BITTI" "$L" 2>/dev/null; do sleep 60; done
cd "$BENCH" || exit 1
for attempt in 1 2; do
  for map in "$G"/modeller/ucus_haritalari/*_geo.tif; do
    stem=$(basename "$map" _geo.tif)
    out="$G/sonuclar/ucus/karlik_f1/${stem}.json"
    [ -f "$out" ] && continue
    m="$BENCH/deneyler/secili_modeller/${stem}.h5"; [ -f "$m" ] || m="$G/modeller/ucus/${stem}.h5"
    [ -f "$m" ] || m="$(ls "$BENCH"/deneyler/2026-09-25_parametre_arama/modeller/*/"${stem}.h5" 2>/dev/null | head -1)"
    [ -f "$m" ] || { echo "$(date '+%F %T') MODEL YOK $stem" >> "$L"; continue; }
    echo "$(date '+%F %T') TEST $stem (tamamlama $attempt)" >> "$L"
    "$ENV/python.exe" -u deney_araclari/ucus_testi.py test --model "$(cygpath -w "$m")" --harita "$(cygpath -w "$map")" \
      --out "$(cygpath -w "$out")" > "$G/kayitlar/ucus_${stem}.log" 2>&1
    grep "^UCUS" "$G/kayitlar/ucus_${stem}.log" >> "$L" || echo "$(date '+%F %T') COKTU $stem" >> "$L"
  done
done
echo "$(date '+%F %T') UCUS TAMAMLAMA BITTI" >> "$L"
