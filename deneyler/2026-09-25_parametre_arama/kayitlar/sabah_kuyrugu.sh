#!/usr/bin/env bash
# Resmi 1000'lik kuyruk bitince: x1 toplulugunun varyantlari (kapi), sonra x1'in 4000 sorguluk kosusu.
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
G=$BENCH/deneyler/2026-09-25_genis_dogrulama
S=$BENCH/deneyler/secili_modeller
L=$A/kayitlar/sabah_kuyrugu.log
ENV=/c/Users/ertugrul/anaconda3/envs/visual_navigation_cuda
export PATH="$ENV:$ENV/Library/bin:$ENV/Library/usr/bin:$ENV/Scripts:$ENV/bin:$PATH"
log() { echo "$(date '+%F %T') $*" >> "$L"; }
W() { cygpath -w "$1"; }
until grep -q "RESMI KUYRUK BITTI" "$A/kayitlar/resmi_kuyruk.log" 2>/dev/null; do sleep 120; done
BDB7="$BENCH/models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5"
T=$A/modeller/topluluklar
cd $BENCH/deney_araclari || exit 1
mk() { local out=$1; shift; [ -f "$T/$out.h5" ] || CUDA_VISIBLE_DEVICES=-1 "$ENV/python.exe" topluluk.py "$@" --out "$(W $T/$out.h5)" >> "$L" 2>&1; }
mk x4_cb62_bdb7_k323_d6 "$(W $S/2022_tpu_f48k4_cb62.h5)" "$(W $BDB7)" "$(W $S/2022_k32_3_e1900.h5)" "$(W $S/2022_deneme6_f32k5.h5)"
mk x6_bdb7_k323_d6 "$(W $BDB7)" "$(W $S/2022_k32_3_e1900.h5)" "$(W $S/2022_deneme6_f32k5.h5)"
mk x7_cb62_k323 "$(W $S/2022_tpu_f48k4_cb62.h5)" "$(W $S/2022_k32_3_e1900.h5)"
mk x8_cb62_d6 "$(W $S/2022_tpu_f48k4_cb62.h5)" "$(W $S/2022_deneme6_f32k5.h5)"
mk x9_k323_d6 "$(W $S/2022_k32_3_e1900.h5)" "$(W $S/2022_deneme6_f32k5.h5)"
mk x10_x1_tta8tam "$(W $S/2022_tpu_f48k4_cb62.h5)" "$(W $S/2022_k32_3_e1900.h5)" "$(W $S/2022_deneme6_f32k5.h5)" "$(W $A/modeller/tta/tta8_tam_e5.h5)"
log "BASLA kapi topluluk varyantlari"
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_g9_topluluk_varyant_150.json" \
  $T/x4_cb62_bdb7_k323_d6.h5 $T/x6_bdb7_k323_d6.h5 $T/x7_cb62_k323.h5 $T/x8_cb62_d6.h5 $T/x9_k323_d6.h5 $T/x10_x1_tta8tam.h5 >> "$L" 2>&1
log "BITTI kapi topluluk varyantlari cikis=$?"
log "BASLA urgup4000_x1_2022_topluluk"
bash "$G/kayitlar/baslat_urgup4000.sh" x1_2022_topluluk > "$G/kayitlar/urgup4000_x1_2022_topluluk.log" 2>&1
log "BITTI urgup4000_x1_2022_topluluk $(tail -1 "$G/kayitlar/urgup4000_x1_2022_topluluk.log")"
log "SABAH KUYRUGU BITTI"
