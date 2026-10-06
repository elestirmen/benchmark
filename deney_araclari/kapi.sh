#!/usr/bin/env bash
# Kullanim: kapi.sh <ornek> <json_cikti> <kosu_klasoru>:<onek> [<kosu_klasoru>:<onek> ...]
#
# Her kosunun *.npz checkpoint'lerini (epoch_/ema_/swa_) <kosu>/keras/ altina
# .h5 olarak cevirir (varsa atlar) ve hepsini margin_gate.py ile birlikte olcer
# (benchmark normalizasyonu minus1_1, egitim bolgesi).
#
# KAPI_GRUPLAR=4 (varsayilan): G2B/B2G x clean/hard_v1 ortalamasi = resmi global
#                              skorun dort grubu.  KAPI_GRUPLAR=1: eski tek grup (G2B clean).
# Arguman olarak dogrudan .h5 dosyasi da verilebilir (onek gerekmez).
# KAPI_GPU=1: TF GPU'da calisir (varsayilan CPU; paralel benchmark GPU bellegini tutarken guvenli).
set -o pipefail  # margin_gate cokerse (sessiz 0xC0000005 dahil) cikis kodu sifirdan farkli olsun
SP="$(cd "$(dirname "$0")" && pwd)"
ENV="/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda"
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
export CUDA_CACHE_PATH='C:\d_surucusu\visual_navigation\benchmark\outputs\cuda_cache'
export CUDA_CACHE_MAXSIZE=4294967296 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
SAMPLES="$1"; JSON_OUT="$2"; shift 2
ARGS=()
for spec in "$@"; do
  if [[ "$spec" == *.h5 ]]; then ARGS+=(--model "$(cygpath -w "$spec")"); continue; fi
  RUN=$(cygpath -u "${spec%:*}"); PREFIX="${spec##*:}"
  NPZ=()
  for f in "$RUN"/*.npz; do [ -f "$f" ] && NPZ+=("$(cygpath -w "$f")"); done
  if [ ${#NPZ[@]} -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=-1 "$ENV/python.exe" "$SP/torch_to_keras.py" "${NPZ[@]}" --skip-existing \
      --out-dir "$(cygpath -w "$RUN/keras")" --prefix "$PREFIX" --tol 3e-3 2>&1 | grep -E "BASARISIZ|Error"
  fi
  for f in "$RUN/keras/"*.h5; do [ -f "$f" ] && ARGS+=(--model "$(cygpath -w "$f")"); done
done
GROUP_ARGS=(--bidirectional --variants clean,hard_v1)
[ "${KAPI_GRUPLAR:-4}" = "1" ] && GROUP_ARGS=()
DEVICE_ARGS=()
if [ "${KAPI_GPU:-0}" = "1" ]; then DEVICE_ARGS=(--require-gpu); else export CUDA_VISIBLE_DEVICES=-1; DEVICE_ARGS=(--threads "${KAPI_THREADS:-6}"); fi
cd /c/d_surucusu/visual_navigation/benchmark || exit 1
"$ENV/python.exe" -u margin_gate.py --samples "$SAMPLES" --aoi train --normalization minus1_1 \
  "${DEVICE_ARGS[@]}" "${GROUP_ARGS[@]}" --json-out "$(cygpath -w "$JSON_OUT")" "${ARGS[@]}" 2>&1 \
  | tee "${JSON_OUT%.json}.log" \
  | grep -E --line-buffered "GECTI|ELENDI|RAW baraji|HATA|Error|Traceback"
