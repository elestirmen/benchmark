#!/usr/bin/env bash
# Kat A: Urgup rotalariyla egit (9 egitim + 2 dogrulama), Karlik rotalarinda (5, 6, 7) test.
# Yakalama bitince baslar. Her kosu: modeller/egitim/<ad>/ ; kayit kayitlar/<ad>.log
BENCH=/c/d_surucusu/visual_navigation/benchmark
D=$BENCH/deneyler/2026-10-02_ucus_uyarlama
R=C:/d_surucusu/visual_navigation/benchmark/deneyler/2026-09-25_genis_dogrulama/kayitlar/ucus_rotalar
ENVT=/c/Users/ertugrul/anaconda3/envs/adaf
until grep -q "YAKALA BITTI" "$D/kayitlar/yakala.log" 2>/dev/null; do sleep 60; done
cd "$BENCH/deney_araclari" || exit 1
O=$D/modeller/ogretmen
EGITIM=(1_tezde_ucus_5 2_tezde_ucus_6 3_tezde_ucus_7 4_tezde_ucus_8 5_tezde_ucus_9
  guzergah_1 guzergah_1_donus guzergah_2_tezde_ucus_1 guzergah_3_tezde_ucus_2)
DOGRULAMA=(6_tezde_ucus_4 guzergah_4_tezde_ucus_3)
kos() {  # $1 ad, sonra ek argumanlar
  local name=$1; shift
  [ -f "$D/modeller/egitim/$name/_log.jsonl" ] && grep -q '"epoch": 8' "$D/modeller/egitim/$name/_log.jsonl" && return
  "$ENVT/python.exe" -u ucus_uyarlama_egitim.py --veri "$D/veri" --out-dir "$D/modeller/egitim/$name" \
    --ogretmen "$O/x1_uye0.npz" "$O/x1_uye1.npz" "$O/x1_uye2.npz" \
    --egitim "${EGITIM[@]}" --dogrulama "${DOGRULAMA[@]}" --epochs 8 \
    --diger-kayitlar "$R/{rota}_2022_*.log" "$R/{rota}_RAW_kimlik.log" "$@" > "$D/kayitlar/$name.log" 2>&1
  echo "$(date '+%F %T') BITTI $name" >> "$D/kayitlar/egit_kat_a.log"
}
kos a1_lr3e5
kos a2_lr1e4 --lr 1e-4
kos a3_lr3e5_aug --aug 0.5
echo "$(date '+%F %T') KAT A EGITIM BITTI" >> "$D/kayitlar/egit_kat_a.log"
