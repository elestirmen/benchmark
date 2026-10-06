#!/usr/bin/env bash
# G1 bulgusu: lr dusurmeli ince ayar + EMA, ayni kosunun en iyi ham checkpoint'ini ~0.02 gecti ve
# checkpoint'leri kararli yapti. Ayni ince ayar topluluk3 uyelerine (en iyi mevcut checkpoint'ler).
BENCH=/c/d_surucusu/visual_navigation/benchmark
A=$BENCH/deneyler/2026-09-25_parametre_arama
Z=$BENCH/deneyler/2026-09-24_zero1_tarif/modeller/egitim
K=$A/kayitlar
L=$K/ince_ayar.log
log() { echo "$(date '+%F %T') $*" >> "$L"; }
until grep -q "BITTI g4_sicaklik" $K/kapi_sirali.log 2>/dev/null; do sleep 60; done
ft() {  # <ad> <npz>
  log "BASLA $1"
  bash $K/egit.sh "$1" --seed 5 --init-npz "$(cygpath -w "$2")" --lr-schedule "3e-4:3,1e-4:3" --ema 0.9998 > "$K/$1.log" 2>&1
  log "BITTI $1 $(tail -1 "$K/$1.log")"
}
ft ft_tam_e5 "$Z/tarama_f32_k3_elu_z1_b1_tamveri/epoch_00005.npz"
ft ft_s2_e7 "$Z/tam_f32_k3_elu_z1_b1_s2/epoch_00007.npz"
ft ft_uzun_e27 "$Z/uzun_f32_k3_elu_z1_b1_3100/epoch_00027.npz"
log "BASLA kapi ince ayar"
bash $BENCH/deney_araclari/kapi.sh 150 "$A/sonuclar/kapi/kapi4_g6_ince_ayar_150.json" \
  $A/modeller/egitim/ft_tam_e5:FT_tam $A/modeller/egitim/ft_s2_e7:FT_s2 $A/modeller/egitim/ft_uzun_e27:FT_uzun >> "$L" 2>&1
log "BITTI kapi ince ayar cikis=$?"
log "INCE AYAR KUYRUGU BITTI"
