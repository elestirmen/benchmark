#!/usr/bin/env bash
# Kapi kuyrugu (4 grup, 150 ornek, CPU): kalibrasyon bitince egitimsiz adaylar, sonra egitim kosulari.
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
Z=$BENCH/deneyler/2026-09-24_zero1_tarif/modeller
K=$A/kayitlar
[ -f "$A/sonuclar/kapi/kapi4_kalibrasyon_150.json" ] || exit 1
echo "$(date '+%F %T') BASLA egitimsiz" >> $K/kapi_kuyrugu.log
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_egitimsiz_150.json" \
  $A/modeller/swa/tam_f32_k3_elu_z1_b1_s1:SWA_s1 $A/modeller/swa/tam_f32_k3_elu_z1_b1_s2:SWA_s2 \
  $A/modeller/swa/tarama_f32_k3_elu_z1_b1_tamveri:SWA_tam $A/modeller/swa/uzun_f32_k3_elu_z1_b1_3100:SWA_uzun \
  $A/modeller/swa/karma_f32_k53_z1_b1_tam:SWA_karma \
  $A/modeller/tta/*.h5 $A/modeller/topluluklar/*.h5 \
  "$Z/egitim/tam_f32_k3_elu_z1_b1_s2/keras/TAM_s2_model_f32_k3_epoch_00007_helu_osigmoid_(1, 1)_.h5" \
  "$Z/egitim/uzun_f32_k3_elu_z1_b1_3100/keras/UZUN_model_f32_k3_epoch_00027_helu_osigmoid_(1, 1)_.h5" \
  >> $K/kapi_kuyrugu.log 2>&1; echo "cikis=$? $(date +%T)" >> $K/kapi_kuyrugu.log
echo "$(date '+%F %T') BITTI egitimsiz" >> $K/kapi_kuyrugu.log
until grep -q "CIKIS_KODU" "$K/r1_s11_ema.log" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') BASLA r1" >> $K/kapi_kuyrugu.log
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_r1_150.json" $A/modeller/egitim/r1_s11_ema:R1 >> $K/kapi_kuyrugu.log 2>&1; echo "cikis=$? $(date +%T)" >> $K/kapi_kuyrugu.log
echo "$(date '+%F %T') BITTI r1" >> $K/kapi_kuyrugu.log
