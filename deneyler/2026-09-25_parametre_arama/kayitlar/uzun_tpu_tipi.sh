#!/usr/bin/env bash
# TPU tipi uzun egitim hipotezi: 2022 TPU f48_k4 serisi (Urgup 4000'de 0.63-0.66, 10 checkpoint
# tutarli) ~1100 epoch egitilmis; batch-1 tarifimiz 12 epoch. Burada buyuk batch + cok epoch:
# f48_k4, batch 16, Adam 1e-3, zero_1, bf16 autocast, EMA 0.999 (adim), seyrek checkpoint.
A=/c/d_surucusu/visual_navigation/benchmark/deneyler/2026-09-25_parametre_arama
bash $A/kayitlar/egit.sh u1_tpu_f48k4_b16 --seed 21 --filters 48 --kernel 4 --batch 16 --lr 1e-3 \
  --epochs 1500 --ema 0.999 --amp \
  --save-epochs 10,25,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1200,1500
