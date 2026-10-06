#!/usr/bin/env bash
# Usage: degerlendir.sh <run_dir(windows path ok)> <prefix> <out_json_basename> <epochs...>
# Converts torch checkpoints of the given epochs to Keras .h5 (verified) and
# evaluates them with the fast proxy. Cached proxy results are skipped.
SP="$(cd "$(dirname "$0")" && pwd)"
SPW="$(cygpath -w "$SP/olcumler")"
ENV="/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda"
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
RUN="$1"; PREFIX="$2"; OUT="$3"; shift 3
RUNU=$(cygpath -u "$RUN")
NPZ=()
for e in "$@"; do
  f="$RUNU/epoch_$(printf '%05d' "$e").npz"
  [ -f "$f" ] && NPZ+=("$(cygpath -w "$f")")
done
[ ${#NPZ[@]} -eq 0 ] && { echo "checkpoint yok"; exit 1; }
CUDA_VISIBLE_DEVICES=-1 "$ENV/python.exe" "$SP/torch_to_keras.py" "${NPZ[@]}" --out-dir "$(cygpath -w "$RUNU/keras")" --prefix "$PREFIX" --tol "${TOL:-2e-4}" 2>&1 | grep -v "WARNING\|GDAL"
ARGS=()
for e in "$@"; do
  f=$(ls "$RUNU/keras/"*"epoch_$(printf '%05d' "$e")_"*.h5 2>/dev/null | head -1)
  [ -n "$f" ] && ARGS+=(--model "$(cygpath -w "$f")")
done
cd /c/d_surucusu/visual_navigation/benchmark || exit 1
"$ENV/python.exe" -u "$SP/hizli_vekil.py" --out "$SPW\\$OUT" --regions train --queries-per-region 100 --no-raw "${ARGS[@]}" 2>&1 \
  | grep --line-buffered -v "^WARNING\|jit-compiled\|TensorFlow was not built\|GDAL_DATA\|oneDNN\|rebuild TensorFlow"
