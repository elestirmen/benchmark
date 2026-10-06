#!/usr/bin/env bash
# Egitim kuyrugu: R1 bitince sirayla calisir (GPU'da ayni anda tek egitim).
A=/c/d_surucusu/visual_navigation/benchmark/deneyler/2026-09-25_parametre_arama
K=$A/kayitlar
until grep -q "CIKIS_KODU" "$K/r1_s11_ema.log" 2>/dev/null; do sleep 30; done
run() { name=$1; shift; echo "$(date '+%F %T') BASLA $name $*" >> "$K/kuyruk.log"; bash "$K/egit.sh" "$name" "$@" > "$K/$name.log" 2>&1; echo "$(date '+%F %T') BITTI $name $(tail -1 "$K/$name.log")" >> "$K/kuyruk.log"; }
run r2_s11_lrd --seed 11 --init-npz "$(cygpath -w "$A/modeller/egitim/r1_s11_ema/epoch_00006.npz")" --lr-schedule "3e-4:3,1e-4:3" --ema 0.9998
run r3_s11_aug --seed 11 --epochs 12 --ema 0.9998 --aug 0.5
run r4_s11_f48 --seed 11 --epochs 12 --ema 0.9998 --filters 48
run r5_s12_ema --seed 12 --epochs 12 --ema 0.9998
echo "$(date '+%F %T') KUYRUK BITTI" >> "$K/kuyruk.log"
