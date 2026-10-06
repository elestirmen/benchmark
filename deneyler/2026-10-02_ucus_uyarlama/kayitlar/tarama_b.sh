#!/usr/bin/env bash
# Tarama B: a1 (Adam 3e-5 + pos terimi) dogrulamayi bozdu (0.882 -> 0.810). Dusuk lr, pos terimi yok,
# cikti/agirlik capasi; 2 epoch, 200 adimda bir dogrulama. Iki kosu paralel.
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
    --egitim "${EGITIM[@]}" --dogrulama "${DOGRULAMA[@]}" --epochs 2 --eval-every 200 --pos-weight 0 \
    --diger-kayitlar "$R/{rota}_2022_*.log" "$R/{rota}_RAW_kimlik.log" "$@" > "$D/kayitlar/$name.log" 2>&1
  echo "$(date '+%F %T') BITTI $name" >> "$D/kayitlar/tarama_b.log"
}
( kos b1_lr3e6 --lr 3e-6; kos b3_lr1e5_capa --lr 1e-5 --anchor 1.0 ) &
( kos b2_lr1e5 --lr 1e-5; kos b4_lr3e5_l2sp --lr 3e-5 --l2sp 10 ) &
wait
echo "$(date '+%F %T') TARAMA B BITTI" >> "$D/kayitlar/tarama_b.log"
