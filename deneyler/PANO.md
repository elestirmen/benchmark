# Sonuç panosu

Otomatik üretildi: `python sonuc_panosu.py` (2026-11-09 05:31). Elle düzenlemeyin; ayrıntılı tüm satırlar `PANO.xlsx` içindedir.

Ölçüt: global arama `success_25m`, dört grubun ortalaması (G2B/B2G × temiz/hard_v1). Δ RAW: aynı protokolün (aynı sorgu kümesi) RAW'ına göre fark. Kapı: `margin_gate.py` yanlış tepe oranı (150 örnek, eğitim bölgesi; küçük iyi).

## urgup_cevresi 4000 sorgu

RAW global: **0.477** · 22 sonuç

| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | topluluk_x1_2022 | 0.718 | +0.241 | 0.734 | 0.710 | 0.762 | 0.668 | 0.870 | 0.818 | 0.406 | `deneyler/secili_modeller/topluluk_x1_2022.h5` |
| 2 | 2022_tpu_f48k4_cb62 | 0.659 | +0.182 | 0.678 | 0.656 | 0.703 | 0.598 | 0.848 | 0.781 | 0.440 | `deneyler/secili_modeller/2022_tpu_f48k4_cb62.h5` |
| 3 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__bdb7daa618c0 | 0.657 | +0.180 | 0.690 | 0.659 | 0.699 | 0.581 | 0.846 | 0.782 | 0.422 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5` |
| 4 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__0ba75572577e | 0.656 | +0.179 | 0.680 | 0.662 | 0.702 | 0.582 | 0.847 | 0.783 | 0.426 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__0ba75572577e.h5` |
| 5 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__be1ed72dac32 | 0.650 | +0.174 | 0.691 | 0.665 | 0.684 | 0.561 | 0.846 | 0.779 | 0.432 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__be1ed72dac32.h5` |
| 6 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__f5ec282aa346 | 0.647 | +0.170 | 0.682 | 0.649 | 0.688 | 0.568 | 0.843 | 0.776 | 0.430 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__f5ec282aa346.h5` |
| 7 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__47491621799b | 0.642 | +0.165 | 0.683 | 0.652 | 0.681 | 0.553 | 0.837 | 0.766 | 0.436 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__47491621799b.h5` |
| 8 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__3ca910618354 | 0.640 | +0.163 | 0.671 | 0.646 | 0.685 | 0.555 | 0.834 | 0.764 | 0.434 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__3ca910618354.h5` |
| 9 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__d75d34b45400 | 0.637 | +0.160 | 0.684 | 0.651 | 0.671 | 0.541 | 0.838 | 0.763 | 0.444 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__d75d34b45400.h5` |
| 10 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__83d14ff0bb51 | 0.633 | +0.156 | 0.675 | 0.646 | 0.667 | 0.542 | 0.840 | 0.768 | 0.435 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__83d14ff0bb51.h5` |
| 11 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__540ee65bcc35 | 0.631 | +0.154 | 0.655 | 0.624 | 0.675 | 0.569 | 0.828 | 0.758 | 0.458 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__540ee65bcc35.h5` |
| 12 | yeni_topluluk3 | 0.625 | +0.148 | 0.666 | 0.633 | 0.666 | 0.536 | 0.799 | 0.734 | 0.495 | `deneyler/secili_modeller/yeni_topluluk3.h5` |
| 13 | 2022_k32_3_e1900 | 0.620 | +0.143 | 0.652 | 0.609 | 0.694 | 0.523 | 0.824 | 0.748 | 0.442 | `deneyler/secili_modeller/2022_k32_3_e1900.h5` |
| 14 | 2022_tpu_f48k4_e1100 | 0.614 | +0.137 | 0.637 | 0.602 | 0.658 | 0.559 | 0.812 | 0.742 | 0.460 | `deneyler/secili_modeller/2022_tpu_f48k4_e1100.h5` |
| 15 | 2022_deneme6_f32k5 | 0.601 | +0.124 | 0.641 | 0.604 | 0.630 | 0.531 | 0.789 | 0.714 | 0.480 | `deneyler/secili_modeller/2022_deneme6_f32k5.h5` |
| 16 | 02_modeller___15_10_2022__16_32_gpu_model_f32_k5_…__82b000f07266 | 0.514 | +0.037 | 0.533 | 0.566 | 0.514 | 0.443 | 0.763 | 0.667 | 0.477 | `models/02_modeller___15_10_2022__16_32_gpu_model_f32_k5_epoch__epoch__sigmoi__1__82b000f07266.h5` |
| 17 | 2022_06_f40k3 | 0.504 | +0.027 | 0.535 | 0.558 | 0.495 | 0.429 | 0.735 | 0.645 | 0.536 | `deneyler/secili_modeller/2022_06_f40k3.h5` |
| 18 | model_k_32_3_epoch_02000_sigmoid_(1, 1)_ | 0.503 | +0.026 | 0.595 | 0.545 | 0.550 | 0.322 | 0.779 | 0.676 | 0.502 | `models/model_k_32_3_epoch_02000_sigmoid_(1, 1)_.h5` |
| 19 | tpu_model_f48_k7_epoch_00100_sigmoid_(1, 1)_ | 0.503 | +0.026 | 0.625 | 0.604 | 0.469 | 0.312 | 0.765 | 0.662 | 0.519 | `models/tpu_model_f48_k7_epoch_00100_sigmoid_(1, 1)_.h5` |
| 20 | **RAW (model yok)** | 0.477 | — | 0.616 | 0.583 | 0.367 | 0.342 | 0.678 | 0.595 | — | — |
| 21 | 01_deneme_6__gpu_model_f32_k5_epoch__epoch__sigmo…__f910fdaa1ad7 | 0.455 | -0.021 | 0.506 | 0.524 | 0.438 | 0.353 | 0.726 | 0.624 | 0.517 | `models/01_deneme_6__gpu_model_f32_k5_epoch__epoch__sigmoid__1__1__07_10_2022__5__f910fdaa1ad7.h5` |
| 22 | GPU_model_f32_k6_epoch_00004_sigmoid_(1, 1)_ | 0.432 | -0.045 | 0.551 | 0.533 | 0.363 | 0.279 | 0.696 | 0.587 | 0.545 | `models/GPU_model_f32_k6_epoch_00004_sigmoid_(1, 1)_.h5` |

## urgup_cevresi 1000 sorgu

RAW global: **0.461** · 36 sonuç

| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | topluluk_x1_2022 | 0.719 | +0.258 | 0.711 | 0.699 | 0.779 | 0.687 | 0.858 | 0.808 | 0.406 | `deneyler/secili_modeller/topluluk_x1_2022.h5` |
| 2 | yeni_tta8_tam_f32k3_e5 | 0.673 | +0.212 | 0.698 | 0.675 | 0.712 | 0.606 | 0.833 | 0.768 | 0.482 | `deneyler/secili_modeller/yeni_tta8_tam_f32k3_e5.h5` |
| 3 | yeni_topluluk3 | 0.615 | +0.155 | 0.659 | 0.625 | 0.656 | 0.522 | 0.793 | 0.722 | 0.495 | `deneyler/secili_modeller/yeni_topluluk3.h5` |
| 4 | yeni_pipeline_z1_f32k3_e9 | 0.569 | +0.108 | 0.597 | 0.570 | 0.619 | 0.488 | 0.803 | 0.721 | 0.534 | `deneyler/secili_modeller/yeni_pipeline_z1_f32k3_e9.h5` |
| 5 | yeni_tam_f32k3_e5 | 0.564 | +0.103 | 0.594 | 0.572 | 0.604 | 0.485 | 0.788 | 0.705 | 0.517 | `deneyler/secili_modeller/yeni_tam_f32k3_e5.h5` |
| 6 | TARAMA_b1_model_f32_k3_epoch_00003_helu_osigmoid_(1, 1)_ | 0.478 | +0.018 | 0.494 | 0.491 | 0.509 | 0.420 | 0.744 | 0.650 | 0.552 | `deneyler/2026-09-24_zero1_tarif/modeller/adaylar/b1_f32k3_e3/TARAMA_b1_model_f32_k3_epoch_00003_helu_osigmoid_(1, 1)_.h5` |
| 7 | **RAW (model yok)** | 0.461 | — | 0.581 | 0.553 | 0.363 | 0.346 | 0.682 | 0.581 | — | — |
| 8 | _JOB_flat_f24_k4_10_09_2026__14_37_model_f24_k4_e…igmoid_(1, 1)_ | 0.378 | -0.083 | 0.380 | 0.481 | 0.369 | 0.282 | 0.615 | 0.513 | 0.716 | `models_pipeline/20260909_223425/flat_f24_k4/_JOB_flat_f24_k4_10_09_2026__14_37_model_f24_k4_epoch_00008_helu_osigmoid_(1, 1)_.h5` |
| 9 | _JOB_flat_f48_k4_sigmoid_repro_09_09_2026__22_37_…igmoid_(1, 1)_ | 0.355 | -0.106 | 0.390 | 0.445 | 0.318 | 0.265 | 0.606 | 0.490 | 0.723 | `models_pipeline/20260909_223425/flat_f48_k4_sigmoid_repro/_JOB_flat_f48_k4_sigmoid_repro_09_09_2026__22_37_model_f48_k4_epoch_00003_helu_osigmoid_(1, 1)_.h5` |
| 10 | _JOB_z1_f32_k3_b1_recon_08_11_2026__22_22_model_f…igmoid_(1, 1)_ | 0.317 | -0.144 | 0.401 | 0.355 | 0.269 | 0.242 | 0.566 | 0.432 | 0.675 | `deneyler/pipeline_20261108_222031/modeller/z1_f32_k3_b1_recon/_JOB_z1_f32_k3_b1_recon_08_11_2026__22_22_model_f32_k3_epoch_00006_helu_osigmoid_(1, 1)_.h5` |
| 11 | _JOB_flat_f64_k4_10_09_2026__21_28_model_f64_k4_e…igmoid_(1, 1)_ | 0.275 | -0.186 | 0.295 | 0.377 | 0.276 | 0.151 | 0.514 | 0.391 | 0.858 | `models_pipeline/20260909_223425/flat_f64_k4/_JOB_flat_f64_k4_10_09_2026__21_28_model_f64_k4_epoch_00006_helu_osigmoid_(1, 1)_.h5` |
| 12 | _JOB_bce_b16_10_09_2026__07_13_best_ssim_model_f48_k4 | 0.255 | -0.206 | 0.327 | 0.369 | 0.182 | 0.143 | 0.496 | 0.379 | 0.771 | `models_pipeline/20260909_223425/bce_b16/_JOB_bce_b16_10_09_2026__07_13_best_ssim_model_f48_k4.h5` |
| 13 | _JOB_bce_b8_10_09_2026__03_50_best_ssim_model_f48_k4 | 0.253 | -0.208 | 0.347 | 0.320 | 0.201 | 0.143 | 0.489 | 0.382 | 0.757 | `models_pipeline/20260909_223425/bce_b8/_JOB_bce_b8_10_09_2026__03_50_best_ssim_model_f48_k4.h5` |
| 14 | _JOB_bce_b32_10_09_2026__11_08_best_ssim_model_f48_k4 | 0.244 | -0.217 | 0.340 | 0.357 | 0.149 | 0.128 | 0.471 | 0.356 | 0.758 | `models_pipeline/20260909_223425/bce_b32/_JOB_bce_b32_10_09_2026__11_08_best_ssim_model_f48_k4.h5` |
| 15 | _JOB_flat_f32_k4_10_09_2026__19_36_son_model | 0.199 | -0.262 | 0.208 | 0.300 | 0.194 | 0.094 | 0.424 | 0.316 | 1.013 | `models_pipeline/20260909_223425/flat_f32_k4/_JOB_flat_f32_k4_10_09_2026__19_36_son_model.h5` |
| 16 | _JOB_flat_f24_k4_03_09_2026__04_30_best_val_loss_model_f24_k4 | 0.178 | -0.283 | 0.195 | 0.205 | 0.189 | 0.122 | 0.396 | 0.286 | — | `models_pipeline/20260830_021548/flat_f24_k4/_JOB_flat_f24_k4_03_09_2026__04_30_best_val_loss_model_f24_k4.h5` |
| 17 | _JOB_flat_f24_k4_03_09_2026__08_53_son_model | 0.178 | -0.283 | 0.195 | 0.205 | 0.189 | 0.122 | 0.396 | 0.286 | — | `models_pipeline/20260830_021548/flat_f24_k4/_JOB_flat_f24_k4_03_09_2026__08_53_son_model.h5` |
| 18 | _JOB_flat_f24_k4_03_09_2026__04_30_best_mae_model_f24_k4 | 0.167 | -0.294 | 0.198 | 0.194 | 0.165 | 0.110 | 0.381 | 0.275 | — | `models_pipeline/20260830_021548/flat_f24_k4/_JOB_flat_f24_k4_03_09_2026__04_30_best_mae_model_f24_k4.h5` |
| 19 | _JOB_flat_f32_k4_03_09_2026__13_33_best_val_loss_model_f32_k4 | 0.121 | -0.339 | 0.151 | 0.137 | 0.134 | 0.063 | 0.324 | 0.214 | — | `models_pipeline/20260830_021548/flat_f32_k4/_JOB_flat_f32_k4_03_09_2026__13_33_best_val_loss_model_f32_k4.h5` |
| 20 | _JOB_flat_f32_k4_03_09_2026__19_56_son_model | 0.121 | -0.339 | 0.151 | 0.137 | 0.134 | 0.063 | 0.324 | 0.214 | — | `models_pipeline/20260830_021548/flat_f32_k4/_JOB_flat_f32_k4_03_09_2026__19_56_son_model.h5` |
| 21 | _JOB_flat_f32_k4_03_09_2026__13_33_best_mae_model_f32_k4 | 0.100 | -0.361 | 0.136 | 0.118 | 0.097 | 0.049 | 0.283 | 0.184 | — | `models_pipeline/20260830_021548/flat_f32_k4/_JOB_flat_f32_k4_03_09_2026__13_33_best_mae_model_f32_k4.h5` |
| 22 | _JOB_flat_f32_k4_03_09_2026__13_33_best_ssim_model_f32_k4 | 0.084 | -0.377 | 0.109 | 0.101 | 0.080 | 0.045 | 0.248 | 0.156 | — | `models_pipeline/20260830_021548/flat_f32_k4/_JOB_flat_f32_k4_03_09_2026__13_33_best_ssim_model_f32_k4.h5` |
| 23 | _JOB_classic_f64_k3_baseline_30_08_2026__12_29_be…s_model_f64_k3 | 0.011 | -0.450 | 0.014 | 0.013 | 0.014 | 0.002 | 0.042 | 0.022 | — | `models_pipeline/20260830_021548/classic_f64_k3_baseline/_JOB_classic_f64_k3_baseline_30_08_2026__12_29_best_val_loss_model_f64_k3.h5` |
| 24 | _JOB_classic_f32_k3_baseline_30_08_2026__05_29_son_model | 0.010 | -0.451 | 0.011 | 0.014 | 0.011 | 0.004 | 0.043 | 0.020 | — | `models_pipeline/20260830_021548/classic_f32_k3_baseline/_JOB_classic_f32_k3_baseline_30_08_2026__05_29_son_model.h5` |
| 25 | _JOB_classic_f16_k3_31_08_2026__00_25_best_mae_model_f16_k3 | 0.007 | -0.454 | 0.011 | 0.009 | 0.004 | 0.003 | 0.033 | 0.017 | — | `models_pipeline/20260830_021548/classic_f16_k3/_JOB_classic_f16_k3_31_08_2026__00_25_best_mae_model_f16_k3.h5` |

… ve 11 sonuç daha (PANO.xlsx → Resmi).

## urgup_30_cm_yeni 300 sorgu · protokol b932f7

RAW global: **0.748** · 546 sonuç

| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2022_deneme6_f32k5 | 0.894 | +0.146 | 0.900 | 0.910 | 0.907 | 0.860 | — | — | 0.480 | `deneyler/secili_modeller/2022_deneme6_f32k5.h5` |
| 2 | _15_10_2022__16_32_GPU_model_f32_k5_epoch_00006_sigmoid_(1_ 1)_ | 0.875 | +0.127 | 0.913 | 0.890 | 0.880 | 0.817 | — | — | 0.477 | `models/02_modeller___15_10_2022__16_32_gpu_model_f32_k5_epoch__epoch__sigmoi__1__82b000f07266.h5` |
| 3 | tpu_model_f48_k4_epoch_01900_sigmoid_(1, 1)_ | 0.852 | +0.103 | 0.853 | 0.833 | 0.907 | 0.813 | — | — | 0.436 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__47491621799b.h5` |
| 4 | 2022_k32_3_e1900 | 0.850 | +0.102 | 0.857 | 0.860 | 0.910 | 0.773 | — | — | 0.442 | `deneyler/secili_modeller/2022_k32_3_e1900.h5` |
| 5 | 2022_tpu_f48k4_e1100 | 0.843 | +0.095 | 0.837 | 0.823 | 0.887 | 0.827 | — | — | 0.460 | `deneyler/secili_modeller/2022_tpu_f48k4_e1100.h5` |
| 6 | tpu_model_f48_k4_epoch_02500_sigmoid_(1, 1)_ | 0.829 | +0.081 | 0.850 | 0.830 | 0.870 | 0.767 | — | — | — | `(dosya bulunamadı) C:\d_surucusu\visual_navigation\benchmark\models\diger\model_tpu_f48_k4_S1_ELU\tpu_model_f48_k4_epoch_02500_sigmoid_(1, 1)_.h5` |
| 7 | _26_09_2023__19_34__model_f32_k4_epoch_00001_relu_(2, 2)_ | 0.827 | +0.078 | 0.870 | 0.847 | 0.803 | 0.787 | — | — | — | `(dosya bulunamadı) C:\d_surucusu\visual_navigation\benchmark\models\diger\modeller\_26_09_2023__19_34__model_f32_k4_epoch_00001_relu_(2, 2)_.h5` |
| 8 | 2022_06_f40k3 | 0.823 | +0.075 | 0.843 | 0.833 | 0.823 | 0.793 | — | — | 0.536 | `deneyler/secili_modeller/2022_06_f40k3.h5` |
|  | **RAW (model yok)** | 0.748 | — | 0.830 | 0.823 | 0.680 | 0.660 | — | — | — | — |

… ve 538 sonuç daha (PANO.xlsx → Resmi).

## urgup_30_cm_yeni 2000 sorgu

RAW global: **0.760** · 232 sonuç

| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | epoch_00006 | 0.886 | +0.126 | 0.912 | 0.899 | 0.884 | 0.848 | — | — | 0.477 | `models/02_modeller___15_10_2022__16_32_gpu_model_f32_k5_epoch__epoch__sigmoi__1__82b000f07266.h5` |
| 2 | 2022_deneme6_f32k5 | 0.881 | +0.121 | 0.908 | 0.899 | 0.900 | 0.817 | — | — | 0.480 | `deneyler/secili_modeller/2022_deneme6_f32k5.h5` |
| 3 | epoch_01550 | 0.860 | +0.100 | 0.871 | 0.857 | 0.891 | 0.823 | — | — | 0.422 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5` |
| 4 | epoch_01950 | 0.860 | +0.100 | 0.868 | 0.851 | 0.891 | 0.829 | — | — | 0.426 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__0ba75572577e.h5` |
| 5 | epoch_01700 | 0.858 | +0.099 | 0.874 | 0.851 | 0.894 | 0.815 | — | — | 0.430 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__f5ec282aa346.h5` |
| 6 | epoch_02400 | 0.858 | +0.098 | 0.874 | 0.846 | 0.887 | 0.824 | — | — | 0.432 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__be1ed72dac32.h5` |
| 7 | epoch_01900 | 0.856 | +0.096 | 0.868 | 0.849 | 0.895 | 0.812 | — | — | 0.436 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__47491621799b.h5` |
| 8 | epoch_02050 | 0.854 | +0.094 | 0.870 | 0.849 | 0.889 | 0.806 | — | — | 0.444 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__d75d34b45400.h5` |
|  | **RAW (model yok)** | 0.760 | — | 0.842 | 0.830 | 0.694 | 0.675 | — | — | — | — |

… ve 224 sonuç daha (PANO.xlsx → Resmi).

## urgup_30_cm_yeni 300 sorgu · protokol a00519

RAW global: **0.748** · 28 sonuç

| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | GPU_model_f32_k5_epoch_00002_sigmoid__1__1__07_10_2022 | 0.894 | +0.146 | 0.900 | 0.910 | 0.907 | 0.860 | 0.956 | 0.932 | — | `(dosya bulunamadı)` |
| 2 | model_k_32_3_epoch_01900_sigmoid__1__1 | 0.850 | +0.102 | 0.857 | 0.860 | 0.910 | 0.773 | 0.939 | 0.899 | — | `(dosya bulunamadı)` |
| 3 | tpu_model_f48_k4_epoch_01100_sigmoid__1__1 | 0.843 | +0.095 | 0.837 | 0.823 | 0.887 | 0.827 | 0.927 | 0.878 | — | `(dosya bulunamadı)` |
| 4 | 18_10_2022__22_32_GPU_model_f40_k3_epoch_00001_sigmoid__1__1 | 0.807 | +0.058 | 0.853 | 0.853 | 0.793 | 0.727 | 0.912 | 0.858 | — | `(dosya bulunamadı)` |
| 5 | 18_10_2022__22_07_GPU_model_f40_k3_epoch_00004_sigmoid__1__1 | 0.801 | +0.052 | 0.867 | 0.830 | 0.777 | 0.730 | 0.918 | 0.859 | — | `(dosya bulunamadı)` |
| 6 | GPU_model_f32_k3_epoch_00001_sigmoid__1__1__06_10_2022 | 0.797 | +0.048 | 0.840 | 0.780 | 0.847 | 0.720 | 0.908 | 0.849 | — | `(dosya bulunamadı)` |
| 7 | model_k_32_3_epoch_02000_sigmoid__1__1 | 0.788 | +0.039 | 0.813 | 0.820 | 0.853 | 0.663 | 0.915 | 0.853 | — | `(dosya bulunamadı)` |
| 8 | 17_10_2022__16_25_GPU_model_f40_k3_epoch_00001_sigmoid__1__1 | 0.786 | +0.037 | 0.803 | 0.787 | 0.797 | 0.757 | 0.898 | 0.828 | — | `(dosya bulunamadı)` |
|  | **RAW (model yok)** | 0.748 | — | 0.830 | 0.823 | 0.680 | 0.660 | 0.870 | 0.799 | — | — |

… ve 20 sonuç daha (PANO.xlsx → Resmi).

## adana 100 sorgu

RAW global: **0.972** · 9 sonuç

| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2022_tpu_f48k4_cb62 | 0.972 | +0.000 | 0.990 | 0.960 | 0.970 | 0.970 | 0.972 | — | 0.440 | `deneyler/secili_modeller/2022_tpu_f48k4_cb62.h5` |
| 2 | **RAW (model yok)** | 0.972 | — | 0.990 | 0.980 | 0.960 | 0.960 | 0.972 | — | — | — |
| 3 | topluluk_x1_2022 | 0.955 | -0.017 | 0.960 | 0.940 | 0.970 | 0.950 | 0.958 | — | 0.406 | `deneyler/secili_modeller/topluluk_x1_2022.h5` |
| 4 | 2022_deneme6_f32k5 | 0.932 | -0.040 | 0.960 | 0.920 | 0.920 | 0.930 | 0.950 | — | 0.480 | `deneyler/secili_modeller/2022_deneme6_f32k5.h5` |
| 5 | 2022_k32_3_e1900 | 0.925 | -0.047 | 0.950 | 0.870 | 0.930 | 0.950 | 0.935 | — | 0.442 | `deneyler/secili_modeller/2022_k32_3_e1900.h5` |
| 6 | yeni_tta8_tam_f32k3_e5 | 0.922 | -0.050 | 0.950 | 0.870 | 0.940 | 0.930 | 0.950 | — | 0.482 | `deneyler/secili_modeller/yeni_tta8_tam_f32k3_e5.h5` |
| 7 | yeni_pipeline_z1_f32k3_e9 | 0.917 | -0.055 | 0.950 | 0.890 | 0.910 | 0.920 | 0.950 | — | 0.534 | `deneyler/secili_modeller/yeni_pipeline_z1_f32k3_e9.h5` |
| 8 | yeni_tam_f32k3_e5 | 0.897 | -0.075 | 0.930 | 0.840 | 0.920 | 0.900 | 0.930 | — | 0.517 | `deneyler/secili_modeller/yeni_tam_f32k3_e5.h5` |

… ve 1 sonuç daha (PANO.xlsx → Resmi).

## Kapı: kapı 150 örnek, 4 grup, train (en iyi 25 / 210)

RAW barajı: **0.584** (küçük olan iyi)

| # | Model | Oran | G2B temiz | G2B hard | B2G temiz | B2G hard | Konum |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | x6_bdb7_k323_d6 | 0.405 | 0.393 | 0.401 | 0.388 | 0.439 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x6_bdb7_k323_d6.h5` |
| 2 | x4_cb62_bdb7_k323_d6 | 0.406 | 0.395 | 0.406 | 0.391 | 0.432 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x4_cb62_bdb7_k323_d6.h5` |
| 3 | topluluk_x1_2022 | 0.406 | 0.398 | 0.402 | 0.391 | 0.434 | `deneyler/secili_modeller/topluluk_x1_2022.h5` |
| 4 | x7_cb62_k323 | 0.410 | 0.400 | 0.412 | 0.391 | 0.437 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x7_cb62_k323.h5` |
| 5 | x10_x1_tta8tam | 0.411 | 0.405 | 0.408 | 0.396 | 0.435 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x10_x1_tta8tam.h5` |
| 6 | x3_tpu_k323_yeni3 | 0.419 | 0.413 | 0.417 | 0.402 | 0.445 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x3_tpu_k323_yeni3.h5` |
| 7 | x8_cb62_d6 | 0.421 | 0.414 | 0.421 | 0.409 | 0.439 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x8_cb62_d6.h5` |
| 8 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__bdb7daa618c0 | 0.422 | 0.405 | 0.431 | 0.407 | 0.446 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5` |
| 9 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__0ba75572577e | 0.426 | 0.414 | 0.440 | 0.405 | 0.445 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__0ba75572577e.h5` |
| 10 | x9_k323_d6 | 0.428 | 0.414 | 0.419 | 0.407 | 0.474 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x9_k323_d6.h5` |
| 11 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__f5ec282aa346 | 0.430 | 0.410 | 0.440 | 0.413 | 0.457 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__f5ec282aa346.h5` |
| 12 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__be1ed72dac32 | 0.432 | 0.418 | 0.443 | 0.416 | 0.452 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__be1ed72dac32.h5` |
| 13 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__3ca910618354 | 0.434 | 0.417 | 0.446 | 0.415 | 0.460 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__3ca910618354.h5` |
| 14 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__83d14ff0bb51 | 0.435 | 0.412 | 0.445 | 0.420 | 0.462 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__83d14ff0bb51.h5` |
| 15 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__47491621799b | 0.436 | 0.417 | 0.446 | 0.418 | 0.462 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__47491621799b.h5` |
| 16 | 2022_tpu_f48k4_cb62 | 0.440 | 0.431 | 0.452 | 0.424 | 0.453 | `deneyler/secili_modeller/2022_tpu_f48k4_cb62.h5` |
| 17 | 2022_k32_3_e1900 | 0.442 | 0.423 | 0.436 | 0.408 | 0.499 | `deneyler/secili_modeller/2022_k32_3_e1900.h5` |
| 18 | x2_tpu_yeni3 | 0.442 | 0.436 | 0.446 | 0.426 | 0.462 | `deneyler/2026-09-25_parametre_arama/modeller/topluluklar/x2_tpu_yeni3.h5` |
| 19 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__d75d34b45400 | 0.444 | 0.429 | 0.449 | 0.427 | 0.470 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__d75d34b45400.h5` |
| 20 | 03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoc…__540ee65bcc35 | 0.458 | 0.447 | 0.476 | 0.445 | 0.466 | `models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__540ee65bcc35.h5` |
| 21 | 2022_tpu_f48k4_e1100 | 0.460 | 0.451 | 0.479 | 0.442 | 0.466 | `deneyler/secili_modeller/2022_tpu_f48k4_e1100.h5` |
| 22 | tta4_x1_2022_tpu_k323_d6 | 0.469 | 0.457 | 0.467 | 0.456 | 0.497 | `deneyler/2026-09-25_parametre_arama/modeller/tta/tta4_x1_2022_tpu_k323_d6.h5` |
| 23 | 02_modeller___15_10_2022__16_32_gpu_model_f32_k5_…__82b000f07266 | 0.477 | 0.475 | 0.448 | 0.489 | 0.494 | `models/02_modeller___15_10_2022__16_32_gpu_model_f32_k5_epoch__epoch__sigmoi__1__82b000f07266.h5` |
| 24 | 2022_deneme6_f32k5 | 0.480 | 0.460 | 0.460 | 0.481 | 0.518 | `deneyler/secili_modeller/2022_deneme6_f32k5.h5` |
| 25 | yeni_tta8_tam_f32k3_e5 | 0.482 | 0.471 | 0.470 | 0.474 | 0.513 | `deneyler/secili_modeller/yeni_tta8_tam_f32k3_e5.h5` |

## Uçuş testi: 1_tezde_ucus_5 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (216 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RAW_kimlik | 0.968 | 0.954 | 0.861 | 12.0 | 5 |
| 2 | 2022_deneme6_f32k5 | 0.926 | 0.912 | 0.875 | 11.0 | 11 |
| 3 | topluluk_x1_2022 | 0.917 | 0.907 | 0.889 | 11.0 | 13 |
| 4 | 2022_k32_3_e1900 | 0.898 | 0.889 | 0.861 | 12.0 | 10 |
| 5 | 2022_tpu_f48k4_cb62 | 0.880 | 0.866 | 0.838 | 12.0 | 16 |

## Uçuş testi: 2_tezde_ucus_6 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (146 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | topluluk_x1_2022 | 0.849 | 0.849 | 0.774 | 17.0 | 19 |
| 2 | 2022_k32_3_e1900 | 0.829 | 0.822 | 0.740 | 16.0 | 19 |
| 3 | 2022_tpu_f48k4_cb62 | 0.808 | 0.767 | 0.637 | 19.0 | 19 |
| 4 | 2022_deneme6_f32k5 | 0.801 | 0.781 | 0.712 | 19.0 | 26 |
| 5 | RAW_kimlik | 0.747 | 0.733 | 0.651 | 20.0 | 29 |

## Uçuş testi: 3_tezde_ucus_7 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (135 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_k32_3_e1900 | 0.822 | 0.815 | 0.674 | 16.0 | 16 |
| 2 | topluluk_x1_2022 | 0.822 | 0.807 | 0.726 | 14.0 | 16 |
| 3 | RAW_kimlik | 0.807 | 0.785 | 0.681 | 15.0 | 17 |
| 4 | 2022_tpu_f48k4_cb62 | 0.770 | 0.741 | 0.667 | 16.0 | 19 |
| 5 | 2022_deneme6_f32k5 | 0.726 | 0.689 | 0.578 | 18.0 | 25 |

## Uçuş testi: 4_tezde_ucus_8 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (115 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_k32_3_e1900 | 0.652 | 0.600 | 0.417 | 40.0 | 35 |
| 2 | topluluk_x1_2022 | 0.635 | 0.609 | 0.383 | 38.0 | 33 |
| 3 | RAW_kimlik | 0.591 | 0.504 | 0.287 | 49.0 | 39 |
| 4 | 2022_tpu_f48k4_cb62 | 0.583 | 0.574 | 0.374 | 42.0 | 41 |
| 5 | 2022_deneme6_f32k5 | 0.565 | 0.539 | 0.261 | 47.0 | 42 |

## Uçuş testi: 5_tezde_ucus_9 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (110 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | topluluk_x1_2022 | 0.945 | 0.909 | 0.864 | 10.0 | 4 |
| 2 | 2022_k32_3_e1900 | 0.936 | 0.909 | 0.864 | 10.0 | 3 |
| 3 | 2022_deneme6_f32k5 | 0.891 | 0.818 | 0.691 | 11.0 | 5 |
| 4 | 2022_tpu_f48k4_cb62 | 0.882 | 0.836 | 0.755 | 10.0 | 12 |
| 5 | RAW_kimlik | 0.764 | 0.745 | 0.600 | 11.0 | 20 |

## Uçuş testi: 6_tezde_ucus_4 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (167 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | d2_s1000 | 0.970 | 0.970 | 0.916 | 11.0 | 4 |
| 2 | topluluk_x1_2022 | 0.922 | 0.910 | 0.850 | 12.0 | 12 |
| 3 | 2022_deneme6_f32k5 | 0.916 | 0.910 | 0.814 | 13.0 | 14 |
| 4 | 2022_tpu_f48k4_cb62 | 0.868 | 0.832 | 0.695 | 14.0 | 21 |
| 5 | RAW_kimlik | 0.856 | 0.832 | 0.737 | 13.0 | 22 |
| 6 | 2022_k32_3_e1900 | 0.796 | 0.772 | 0.737 | 13.0 | 23 |

## Uçuş testi: guzergah_1 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (81 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_k32_3_e1900 | 0.963 | 0.951 | 0.802 | 22.0 | 1 |
| 2 | topluluk_x1_2022 | 0.963 | 0.926 | 0.741 | 22.0 | 2 |
| 3 | 2022_deneme6_f32k5 | 0.901 | 0.716 | 0.617 | 23.0 | 8 |
| 4 | 2022_tpu_f48k4_cb62 | 0.864 | 0.753 | 0.568 | 23.0 | 8 |
| 5 | RAW_kimlik | 0.506 | 0.395 | 0.247 | 65.0 | 21 |

## Uçuş testi: guzergah_1_donus · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (54 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | topluluk_x1_2022 | 0.815 | 0.815 | 0.019 | 36.0 | 11 |
| 2 | 2022_k32_3_e1900 | 0.796 | 0.796 | 0.019 | 36.0 | 10 |
| 3 | 2022_deneme6_f32k5 | 0.704 | 0.685 | 0.000 | 35.0 | 14 |
| 4 | 2022_tpu_f48k4_cb62 | 0.704 | 0.685 | 0.000 | 37.0 | 12 |
| 5 | RAW_kimlik | 0.537 | 0.481 | 0.019 | 65.0 | 27 |

## Uçuş testi: guzergah_2_tezde_ucus_1 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (145 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_k32_3_e1900 | 0.897 | 0.848 | 0.207 | 34.0 | 11 |
| 2 | topluluk_x1_2022 | 0.869 | 0.821 | 0.207 | 34.0 | 12 |
| 3 | 2022_deneme6_f32k5 | 0.807 | 0.786 | 0.179 | 35.0 | 14 |
| 4 | 2022_tpu_f48k4_cb62 | 0.793 | 0.745 | 0.159 | 35.0 | 23 |
| 5 | RAW_kimlik | 0.448 | 0.400 | 0.103 | 165.0 | 54 |

## Uçuş testi: guzergah_3_tezde_ucus_2 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (200 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_k32_3_e1900 | 0.975 | 0.970 | 0.895 | 21.0 | 2 |
| 2 | topluluk_x1_2022 | 0.975 | 0.965 | 0.890 | 22.0 | 2 |
| 3 | 2022_deneme6_f32k5 | 0.970 | 0.950 | 0.870 | 21.0 | 4 |
| 4 | 2022_tpu_f48k4_cb62 | 0.950 | 0.910 | 0.800 | 22.0 | 4 |
| 5 | RAW_kimlik | 0.930 | 0.920 | 0.825 | 22.0 | 7 |

## Uçuş testi: guzergah_4_tezde_ucus_3 · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (210 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | d2_s1000 | 0.914 | 0.838 | 0.605 | 22.0 | 17 |
| 2 | topluluk_x1_2022 | 0.838 | 0.776 | 0.552 | 23.0 | 31 |
| 3 | 2022_k32_3_e1900 | 0.819 | 0.724 | 0.533 | 24.0 | 32 |
| 4 | 2022_deneme6_f32k5 | 0.790 | 0.700 | 0.538 | 23.0 | 36 |
| 5 | RAW_kimlik | 0.781 | 0.690 | 0.424 | 28.0 | 38 |
| 6 | 2022_tpu_f48k4_cb62 | 0.681 | 0.557 | 0.400 | 40.0 | 44 |

## Uçuş testi: guzergah_5_to_karain · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (191 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | d2_s1000 | 0.848 | 0.848 | 0.728 | 10.0 | 25 |
| 2 | 2022_deneme6_f32k5 | 0.843 | 0.838 | 0.738 | 12.0 | 22 |
| 3 | topluluk_x1_2022 | 0.812 | 0.806 | 0.670 | 15.0 | 27 |
| 4 | RAW_kimlik | 0.801 | 0.785 | 0.660 | 11.0 | 30 |
| 5 | yeni_topluluk3 | 0.785 | 0.723 | 0.550 | 16.0 | 34 |
| 6 | 2022_k32_3_e1900 | 0.775 | 0.770 | 0.613 | 16.0 | 26 |
| 7 | yeni_tam_f32k3_e5 | 0.775 | 0.717 | 0.555 | 16.0 | — |
| 8 | 2022_tpu_f48k4_e1100 | 0.613 | 0.581 | 0.393 | 43.0 | 55 |
| 9 | 2022_tpu_f48k4_cb62 | 0.534 | 0.492 | 0.325 | 50.0 | 62 |
| 10 | yeni_pipeline_z1_f32k3_e9 | 0.508 | 0.440 | 0.267 | 63.0 | 82 |

## Uçuş testi: guzergah_5_to_karain · F2: zero_1 + none, kesisimsiz kare = tekil sablon (191 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_k32_3_e1900 | 0.545 | 0.508 | 0.325 | 49.0 | 77 |
| 2 | yeni_topluluk3 | 0.513 | 0.476 | 0.325 | 52.0 | 67 |
| 3 | yeni_tam_f32k3_e5 | 0.414 | 0.361 | 0.215 | 184.0 | 75 |
| 4 | 2022_tpu_f48k4_cb62 | 0.325 | 0.298 | 0.162 | 362.0 | 96 |

## Uçuş testi: guzergah_6_to_tagar · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (130 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | d2_s1000 | 0.785 | 0.777 | 0.654 | 15.0 | 13 |
| 2 | topluluk_x1_2022 | 0.754 | 0.754 | 0.662 | 15.0 | 15 |
| 3 | 2022_k32_3_e1900 | 0.715 | 0.715 | 0.615 | 15.0 | 19 |
| 4 | 2022_deneme6_f32k5 | 0.608 | 0.600 | 0.446 | 33.0 | 24 |
| 5 | 2022_tpu_f48k4_cb62 | 0.562 | 0.546 | 0.392 | 37.0 | 28 |
| 6 | RAW_kimlik | 0.538 | 0.477 | 0.392 | 51.0 | 25 |

## Uçuş testi: guzergah_7_to_gurlevek · F1: minus1_1 + none, kesisimsiz kare = tekil sablon (144 kare, GPS merkezli)

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2022_deneme6_f32k5 | 0.799 | 0.653 | 0.424 | 36.0 | 27 |
| 2 | d2_s1000 | 0.764 | 0.722 | 0.507 | 24.0 | 21 |
| 3 | topluluk_x1_2022 | 0.757 | 0.729 | 0.507 | 23.0 | 30 |
| 4 | RAW_kimlik | 0.625 | 0.438 | 0.292 | 54.0 | 36 |
| 5 | 2022_k32_3_e1900 | 0.514 | 0.493 | 0.333 | 53.0 | 54 |
| 6 | 2022_tpu_f48k4_cb62 | 0.444 | 0.424 | 0.257 | 238.0 | 63 |

## Uçuş testi: karlik_30_cm_bingmap_utm.tif · guzergah_5_to_karain (116 kare, GPS merkezli)

> 2026-09-12 koşusu: eski template-matching kodu ve model başına ön işleme sözleşmeleri; bugünkü F1/F2 ile kıyaslanmaz (bkz. deneyler/2026-09-25_genis_dogrulama/BENIOKU.md).

| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GPU_model_f48_k5_epoch_00500_sigmoid_(2, 2)_ | 0.784 | 0.552 | 0.138 | 48.0 | — |
| 2 | GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_ | 0.776 | 0.509 | 0.069 | 50.0 | — |
| 3 | 13_05_2023__14_07_model_step_300 | 0.767 | 0.569 | 0.078 | 46.0 | — |
| 4 | GPU_model_f48_k5_epoch_00300_sigmoid_(2, 2)_ | 0.750 | 0.517 | 0.121 | 49.0 | — |
| 5 | _03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__ | 0.741 | 0.491 | 0.181 | 51.0 | — |
| 6 | _03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__ | 0.733 | 0.491 | 0.164 | 51.0 | — |
| 7 | GPU_model_f24_k4_epoch_00006_sigmoid_(1, 1)_ | 0.724 | 0.517 | 0.121 | 49.0 | — |
| 8 | GPU_model_f48_k5_epoch_00450_sigmoid_(2, 2)_ | 0.724 | 0.517 | 0.155 | 49.5 | — |
| 9 | tpu_model_f48_k7_epoch_00100_sigmoid_(1, 1)_ | 0.716 | 0.440 | 0.129 | 52.5 | — |
| 10 | 2022_deneme6_f32k5 | 0.707 | 0.448 | 0.103 | 52.0 | — |
| 11 | _03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__ | 0.707 | 0.466 | 0.112 | 52.0 | — |
| 12 | model_k_32_7_epoch_00100_sigmoid_(1, 1)_ | 0.647 | 0.431 | 0.086 | 53.0 | — |
| 13 | GPU_model_f32_k2_epoch_00001_sigmoid_(1, 1)_ | 0.638 | 0.466 | 0.138 | 52.0 | — |
| 14 | _17_10_2022__16_25_GPU_model_f40_k3_epoch_00001_sigmoid_(1_  | 0.638 | 0.552 | 0.233 | 47.5 | — |
| 15 | 2022_k32_3_e1900 | 0.629 | 0.379 | 0.138 | 52.0 | — |
| 16 | 2022_tpu_f48k4_e1100 | 0.629 | 0.431 | 0.164 | 52.5 | — |
| 17 | GPU_model_f32_k5_epoch_00008_sigmoid_(1_ 1)_11_10_2022_ | 0.603 | 0.457 | 0.129 | 53.0 | — |
| 18 | _18_10_2022__22_32_GPU_model_f40_k3_epoch_00001_sigmoid_(1_  | 0.603 | 0.534 | 0.181 | 48.0 | — |
| 19 | 2022_06_f40k3 | 0.603 | 0.474 | 0.224 | 53.0 | — |
| 20 | GPU_model_f32_k3_epoch_00009_sigmoid_(1, 1)_ | 0.595 | 0.483 | 0.121 | 51.0 | — |
| 21 | GPU_model_f32_k6_epoch_00004_sigmoid_(1, 1)_ | 0.595 | 0.431 | 0.207 | 55.0 | — |
| 22 | _03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__ | 0.595 | 0.362 | 0.147 | 55.0 | — |
| 23 | model_k_32_3_epoch_02000_sigmoid_(1, 1)_ | 0.578 | 0.422 | 0.190 | 52.0 | — |
| 24 | _20_10_2022__00_27_GPU_model_f40_k5_epoch_00008_sigmoid_(1_  | 0.569 | 0.474 | 0.207 | 53.5 | — |
| 25 | 2022_tpu_f48k4_cb62 | 0.560 | 0.388 | 0.138 | 55.0 | — |

## Nerede ne var

- `deneyler/<deney>/modeller/` eğitilen modeller, `sonuclar/` benchmark ve kapı çıktıları, `kayitlar/` günlükler; her deneyin `BENIOKU.md` dosyası amacını ve bulgularını anlatır.
- `deneyler/pipeline_<run_id>/` `end_to_end_pipeline.py` koşuları (aynı düzen).
- `deneyler/secili_modeller/` öne çıkan modellerin kopyaları ve `*.kart.json` kartları.
- Eski düzen (yerinde okunur): `models_pipeline/<run_id>/`, `outputs/end_to_end_pipeline_<run_id>/`, `outputs/urgup_cevresi_models_4000/`, `models/`.

Taranan resmi koşu klasörü: 35
