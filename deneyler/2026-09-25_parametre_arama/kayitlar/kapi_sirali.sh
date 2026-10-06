#!/usr/bin/env bash
# Benchmark surerken tek hafif is: kapi olcumleri sirayla (CPU, 6 is parcacigi), her grup ayri surec.
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
Z=$BENCH/deneyler/2026-09-24_zero1_tarif/modeller
K=$A/kayitlar
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
E=$A/modeller/egitim
log() { echo "$(date '+%F %T') $*" >> $K/kapi_sirali.log; }
convert() {  # <kosu> <onek>
  local npz=(); for f in "$E/$1"/*.npz; do [ -f "$f" ] && npz+=("$(cygpath -w "$f")"); done
  CUDA_VISIBLE_DEVICES=-1 "$ENV/python.exe" $BENCH/deney_araclari/torch_to_keras.py "${npz[@]}" --skip-existing \
    --out-dir "$(cygpath -w "$E/$1/keras")" --prefix "$2" --tol 3e-3 2>&1 | grep -E "BASARISIZ|Error" >> $K/kapi_sirali.log
}
pick() {  # <kosu> <epoch listesi> -> ham ve ema h5 yollari
  for e in $2; do for f in "$E/$1/keras/"*"_epoch_$(printf %05d $e)_"*.h5; do [ -f "$f" ] && echo "$f"; done; done
}
gate() {  # <ad> <h5...>
  local name=$1; shift
  log "BASLA $name ($# model)"
  bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_${name}_150.json" "$@" >> $K/kapi_sirali.log 2>&1
  log "BITTI $name cikis=$?"
}
convert r2_s11_lrd R2; convert r3_s11_aug R3; convert r4_s11_f48 R4
mapfile -t G1 < <(pick r1_s11_ema "4 6 8 10 12"; pick r2_s11_lrd "1 2 3 4 5 6")
gate g1_ema_lrd "${G1[@]}"
mapfile -t G2 < <(pick r3_s11_aug "4 6 8 10 12"; pick r4_s11_f48 "2 4 6")
gate g2_aug_f48 "${G2[@]}"
for r in tam_f32_k3_elu_z1_b1_s1:SWA_s1 tam_f32_k3_elu_z1_b1_s2:SWA_s2 tarama_f32_k3_elu_z1_b1_tamveri:SWA_tam uzun_f32_k3_elu_z1_b1_3100:SWA_uzun karma_f32_k53_z1_b1_tam:SWA_karma; do :; done
G3=( "$A"/modeller/swa/*/keras/*.h5 "$A"/modeller/tta/*.h5 "$A"/modeller/topluluklar/*.h5
     "$Z/egitim/tam_f32_k3_elu_z1_b1_s2/keras/TAM_s2_model_f32_k3_epoch_00007_helu_osigmoid_(1, 1)_.h5"
     "$Z/egitim/uzun_f32_k3_elu_z1_b1_3100/keras/UZUN_model_f32_k3_epoch_00027_helu_osigmoid_(1, 1)_.h5" )
gate g3_swa_tta_topluluk "${G3[@]}"
log "KAPI SIRALI BITTI"
