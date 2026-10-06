#!/usr/bin/env bash
# Tarama D: secim GPS vekiliyle (VAL_GPS; etiketli basari x1 lehine yanli cikti). 6 epoch,
# 500 adimda bir + epoch sonu dogrulama. Iki kosu paralel.
BENCH=/c/d_surucusu/visual_navigation/benchmark
D=$BENCH/deneyler/2026-10-02_ucus_uyarlama
R=C:/d_surucusu/visual_navigation/benchmark/deneyler/2026-09-25_genis_dogrulama/kayitlar/ucus_rotalar
ENVT=/c/Users/ertugrul/anaconda3/envs/adaf
cd "$BENCH/deney_araclari" || exit 1
O=$D/modeller/ogretmen
EGITIM=(1_tezde_ucus_5 2_tezde_ucus_6 3_tezde_ucus_7 4_tezde_ucus_8 5_tezde_ucus_9
  guzergah_1 guzergah_1_donus guzergah_2_tezde_ucus_1 guzergah_3_tezde_ucus_2)
DOGRULAMA=(6_tezde_ucus_4 guzergah_4_tezde_ucus_3)
kos() {
  local name=$1; shift
  "$ENVT/python.exe" -u ucus_uyarlama_egitim.py --veri "$D/veri" --out-dir "$D/modeller/egitim/$name" \
    --ogretmen "$O/x1_uye0.npz" "$O/x1_uye1.npz" "$O/x1_uye2.npz" \
    --egitim "${EGITIM[@]}" --dogrulama "${DOGRULAMA[@]}" --epochs 6 --eval-every 500 \
    --diger-kayitlar "$R/{rota}_2022_*.log" "$R/{rota}_RAW_kimlik.log" "$@" > "$D/kayitlar/$name.log" 2>&1
  echo "$(date '+%F %T') BITTI $name" >> "$D/kayitlar/tarama_d.log"
}
( kos d1_lr3e5_pos --lr 3e-5 --pos-weight 1 --ignore 0; kos d3_lr1e4 --lr 1e-4 --pos-weight 0 --ignore 10 ) &
( kos d2_lr3e5 --lr 3e-5 --pos-weight 0 --ignore 10; kos d4_lr3e5_aug --lr 3e-5 --pos-weight 0 --ignore 10 --aug 0.5 ) &
wait
echo "$(date '+%F %T') TARAMA D BITTI" >> "$D/kayitlar/tarama_d.log"
