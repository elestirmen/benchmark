import argparse
import csv
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ==========================================
# AYARLAR
# ==========================================
PYTHON_EXEC = r"C:\Users\ertugrul\anaconda3\envs\visual_navigation_cuda\python.exe"

DATA_DIR_PATH = (
    r"E:\temp\tez__ayna\python_calismalar"
    r"\larger-google-sat2maps-dataset\islenmis\sonuclar_19"
)

TRAINING_SCRIPT_DIR = r"C:\d_surucusu\visual_navigation\autoencoder_sat_to_map\egitim"
TRAINING_SCRIPT_NAME = "autoencoder_unified.py"

BENCHMARK_SCRIPT = r"C:\d_surucusu\visual_navigation\benchmark\run_urgup_cevresi_models.py"
MODEL_VALIDATOR_SCRIPT = r"C:\d_surucusu\visual_navigation\benchmark\validate_trained_models.py"
MARGIN_GATE_SCRIPT = r"C:\d_surucusu\visual_navigation\benchmark\margin_gate.py"
# Kapi ornek sayisi. 20 ornek siniflari (cokmus / RAW / kazanan) ayirmaya
# yeter; ama batch 1 checkpointleri birbirine yakin ve kapi artik sampiyonu
# SECIYOR. 2026-09-24'te 60 ornek bunun icin gurultulu cikti: ayni checkpoint
# 60 ornekte 0.37, 150 ornekte 0.51 (karma f32_k53 e1). 150 ornekle siralama
# resmi benchmark'la uyusur: k_32_3 0.42 / TPU 0.43 / deneme_6 0.46 /
# 06_modeller 0.52 / RAW 0.53 (resmi global 0.620 / 0.659 / 0.601 / 0.504 /
# 0.461). Maliyet checkpoint basina ~1 dk CPU.
MARGIN_GATE_SAMPLES = 150
# Kapi gruplari. Resmi global skor dort grubun ortalamasidir (G2B/B2G x
# clean/hard_v1); eski kapi yalniz G2B clean olcuyordu ve yeni modellerin
# en zayif oldugu B2G hard_v1'i gormuyordu. 2026-09-25 kalibrasyonu (resmi
# sonucu bilinen 39 model, deneyler/2026-09-25_parametre_arama):
# Spearman(kapi, resmi RAW farki) Urgup 1000 sorguda 4 grup -0.909 / eski
# -0.718, Urgup 4000 sorguda -0.918 / -0.847. 4 grupta RAW baraji 0.584;
# TPU cb62 0.440, k_32_3 0.442, deneme_6 0.480, yeni topluluk3 0.495,
# 06_modeller 0.536. Maliyet ~4x (dort kirpma kumesi).
MARGIN_GATE_VARIANTS = ("clean", "hard_v1")
MARGIN_GATE_BIDIRECTIONAL = True
# Kapi sampiyonunun yansima TTA'li surumu de kapidan gecer; daha iyiyse benchmark'a
# o girer (consider_tta_champion). Bos tuple = kapali.
BENCHMARK_TTA_TRANSFORMS = (
    "id", "fliplr", "flipud", "rot180", "rot90", "rot270", "transpose", "antitranspose",
)
TTA_BUILDER_SCRIPT = r"C:\d_surucusu\visual_navigation\benchmark\deney_araclari\yansima_toplulugu.py"
# Epoch checkpointleri buraya tasinir. model_files_in() alt klasorlere
# bakmadigi icin benchmark yalniz ust klasordeki adaylari calistirir.
EPOCH_CHECKPOINT_DIR = "_epochs"
# Marj kapisi checkpoint basina ~1 dk CPU harcar. 100 epochun tamamini
# olcmek is basina ~100 dk ekler; esit arali ornekleme egriyi yakalamak
# icin yeterlidir ve maliyeti ~20 dk'da tutar. Kaydedilen best_* ve
# son_model dosyalari bu sinirin disindadir, her zaman olculur.
MARGIN_GATE_MAX_EPOCH_CHECKPOINTS = 20
# Benchmark'a verilecek --model-dir: YALNIZ kapi sampiyonunu icerir.
# geospatial_model_benchmark.py katalogu model_dir.rglob("*") ile kurar
# (geospatial_model_benchmark.py:563), yani OZYINELEMELI tarar. Bu yuzden
# dosyalari alt klasore tasimak onlari benchmark'tan CIKARMAZ -- 2026-09-10'da
# job 1'in katalogu "dosya=25 | calistirilacak=20" dedi, cunku _epochs/ ve
# _benchmark_disi/ altindaki her sey de sayildi (20 x ~50 dk = ~16 saat).
# Dogru cozum benchmark'i yalniz sampiyonu iceren bir klasore yoneltmek.
BENCHMARK_CANDIDATE_DIR = "_benchmark_adayi"
# Marj kapisi bolgesi. margin_gate.py'nin kendi yardim metni egitim
# degerlendirmesinde 'train' kullanilmasini soyler: kapi artik benchmark'a
# girecek checkpointi SECTIGI icin, secimi guney %25'lik test seridine
# bakarak yapmak sonuclari iyimser gosterir.
MARGIN_GATE_AOI = "train"
BENCHMARK_CWD = r"C:\d_surucusu\visual_navigation\benchmark"
# Tek kok: her pipeline kosusu deneyler/pipeline_<run_id>/ altinda modeller/,
# sonuclar/ (benchmark + TUM_SONUCLAR.xlsx) ve kayitlar/ ile durur. Eski
# kosular (models_pipeline/<run_id> + outputs/end_to_end_pipeline_<run_id>)
# yerlerinde kalir ve resume icin o yollar kullanilir.
DENEYLER_DIR = Path(BENCHMARK_CWD) / "deneyler"
PIPELINE_RUN_PREFIX = "pipeline_"
MODELS_PIPELINE_DIR = Path(BENCHMARK_CWD) / "models_pipeline"
BENCHMARK_MAX_QUERIES = 1000
# Cikarim batch geri cekilme merdiveni. run_urgup_cevresi_models.py varsayilani
# 16'dir ve classic_f96_k3 tam olarak bu deger yuzunden build_model_map icinde
# GPU OOM alip kayboldu. --batch-size bilimsel degil operasyonel bir ayardir
# (benchmark/README.md), yani resume sirasinda degistirilmesi sonuclari bozmaz.
BENCHMARK_BATCH_ATTEMPTS = (16, 8, 4, 2)
# Egitim batch'i. 1, 2022 kazananlarinin tarifidir (bkz. NEDEN BU SWEEP BOYLE).
DEFAULT_BATCH_SIZE = 1
LEGACY_BATCH_SIZE = 8
# Benchmark'in HER modele verdigi giris normalizasyonu
# (run_urgup_cevresi_models.core_arguments: --normalization minus1_1).
# Egitim normalizasyonundan BAGIMSIZDIR: zero_1 ile egitilen bir model de
# benchmark'ta minus1_1 ile beslenir. Marj kapisi benchmark'in olctugunu
# olcmek zorunda oldugu icin bu degeri kullanir, egitiminkini degil.
BENCHMARK_INPUT_NORMALIZATION = "minus1_1"
RAW_MODEL_ID = "RAW_BASELINE"
BENCHMARK_VARIANTS = ("clean", "hard_v1")
BENCHMARK_SEARCH_MODES = (
    "roi_500m",
    "roi_1000m",
    "roi_2000m",
    "roi_4000m",
    "roi_8000m",
    "global",
)

# main() calisirken en yeni yarim run otomatik secilir.
PIPELINE_RUN_ID = None
PIPELINE_RUN_DIR = None
PIPELINE_MODELS_ROOT = None
PIPELINE_BENCHMARK_ROOT = None


# =====================================================================
# NEDEN BU SWEEP BOYLE (2026-09-24 guncellemesi)
# =====================================================================
# 2026-09-10 sweepinin butun isleri (flat_bottleneck f24-f96, batch 2-32,
# minus1_1) RAW'in altinda kaldi: en iyisi flat_f24_k4 global success_25m
# 0.378, RAW 0.461 (1000 sorgu). Mimari kazananlarla birebir aynidir; fark
# egitim tarifindedir ve uc olcumle bulundu:
#
#   1. BATCH_SIZE=1 BELIRLEYICI EKSEN. Ekim 2022 checkpoint adlari
#      (_14_10_2022__16_10_GPU_model_f32_k3_epoch_00001_...(1_ 1)_.h5)
#      arsiv/autoencoder_gpu_froom_kaggle.py sablonudur: npz, /255 giris ve
#      etiket, BCE, Adam 1e-3, model.fit(batch_size=1, shuffle=True). Ayni
#      f32_k3 mimarisi, ayni veri ve tarifle yalniz batch degistirilerek
#      (PyTorch esdegeri, hizli vekil benchmark; bkz. asagidaki tablo):
#        batch 8: epoch 1/3/5 = 0.301 / 0.311 / 0.265
#        batch 1: epoch 1/2/3/4/5 = 0.326 / 0.608 / 0.753 / 0.748 / 0.625
#      Eski sweep batch'i tam ters yone (8/16/32) tariyordu.
#   2. GIRIS [0, 1] (zero_1). TPU ve k_32_3 kazananlarinin Bern verisindeki
#      BCE'si zero_1 girisle 0.39-0.41, minus1_1 ile 1.19-1.25: [0, 1] ile
#      egitildiler.
#      Benchmark ise HER modeli minus1_1 ile besler; kazananlar bu yuzden
#      2x kontrastli girisle calisip Urgup'te doygun, ikili bir koyu-yapi
#      haritasi uretir (cikti std 0.26-0.33; eski pipeline modelleri
#      0.03-0.09). Eski pipeline egitimi bilerek minus1_1'e esitlemisti.
#      batch 1'de minus1_1 epoch 1-5: 0.515 / 0.388 / 0.505 / 0.515 / 0.440;
#      zero_1'in tepesi daha yuksek (0.753).
#   3. CHECKPOINT KALITESI EPOCHTAN EPOCHA VE SEED'E GORE DALGALANIR (batch 1
#      SGD'nin dogasi). Tam veri, seed 42: epoch 2/5/6 = 0.641 / 0.809 / 0.670
#      (vekil). 2022 serileri de boyle (TPU f48_k4: e50 0.28, e650 0.78,
#      e1300 0.49, e1900 0.85; urgup_bes_nokta_global_100). Her epoch diske
#      yazilir ve marj kapisi HEPSINI olcer; TEKRAR isleri bu yuzden var.
#
# RESMI BENCHMARK (1000 sorgu, global success_25m ortalamasi, 2026-09-24):
#   RAW 0.461 | eski pipeline en iyi (flat_f24_k4) 0.378
#   bu tarif, 3100 ornek e3 (kapi 0.53) 0.479
#   bu tarif, tam veri e5   (kapi 0.50) 0.564 -- roi_500m 0.788, roi_1000m 0.754
#   bu pipeline (TF), z1_f32_k3_b1 run 20260924_065751 (kapi 0.51) 0.569 --
#     roi_500m 0.803; yukaridaki PyTorch esdegeriyle ayni
#   uc bu-tarif modelinin ciktisi ortalanarak (kapi 0.47; 38.355 parametre,
#   TPU'dan kucuk) 0.615 -- roi_500m 0.793
#   2022 (4000 sorgu): 06_modeller 0.504 | deneme_6 0.601 | TPU e1100 0.614 |
#   k_32_3 0.620 | TPU 0.659. Tek modelde kalan fark global aramadadir:
#   roi_500m'de tam veri e5 (0.788) deneme_6 (0.789) ile ayni. Kapi degerleri
#   150 orneklik olcumdur. Araclar: deney_araclari/, veriler: deneyler/2026-09-24_zero1_tarif/.
#
# Hizli vekil (outputs disinda, 2 x 6x6 km egitim bolgesi, ayni
# coarse_to_fine_search + hard_v1) eski modellerin sirasini dogru verdi ama
# yeni modelleri IYIMSER olcer: 36 km2 aramada yanlis tepe az, 486 km2'de cok.
# Bu yuzden secici marj kapisidir (150 ornek), vekil degil.
#
# Mimari eksenleri (batch 1, zero_1): f48_k4 6 epochta kapi 0.57 (60 ornek),
# karma k5/k3 (deneme_6 f32_k5 mimarisi) tam veride en iyi 0.52 (f32_k3 0.40);
# output19_ps (dolu yol etiketleri) kapi 0.65 -- Bern sonuclar_19 daha iyi.
# Taban bu yuzden deneme_6 serisinin f32_k3 mimarisidir.
# ---------------------------------------------------------------------
# ONCEKI GEREKCE (2026-09-10). Mimari sinifi / sigmoid / BCE bulgulari
# gecerlidir; batch ekseni yorumu (buyuk batch) yukaridaki olcumle curudu.
# ---------------------------------------------------------------------
# Onceki sweep (16 is, MODEL_TYPE="classic", OUTPUT_ACTIVATION="tanh",
# EPOCHS=20) filtre sayisi / kernel / optimizer / lr / loss eksenlerini
# tariyordu ve tamamlanan 4 isin hepsi RAW baseline'in ~50 kati kotu cikti
# (success_25m 0.004-0.014, RAW 0.461-0.682).
#
# Sebep olculdu: taranan eksenlerin hicbiri belirleyici degildi. Belirleyici
# olan iki degisken sabit ve YANLIS degerde tutulmustu:
#
#   1. MIMARI SINIFI. "classic" 50 katmanli, iki kez 2x uzamsal indirme yapan,
#      skip connection + BatchNorm'lu, 158k-5.65M parametreli bir U-Net.
#      Bu kapasite sat->harita semantik cevirisini gercekten yapabiliyor ve
#      yaptigi anda piksel duzeyi karsilikliligi yok ediyor.
#      Benchmark'ta 0.68-0.70 alan top_modeller/ modelleri ise 7 katmanli,
#      HIC indirme yapmayan, 13k-50k parametreli daralan conv yiginlari.
#   2. CIKIS AKTIVASYONU. Kazanan modellerin hepsi sigmoid; eski sweep tum
#      islerde tanh'i zorunlu tutuyordu, yani bilinen-iyi konfigurasyonu
#      uretmesi imkansizdi.
#
# Olculen ayirt edicilik (margin_gate.py, yanlis tepe / gercek tepe orani;
# kucuk olan daha iyi, benchmark siralamasiyla uyusur):
#
#   top_modeller tpu_f48_k4 sigmoid  oran 0.32   benchmark 0.703
#   RAW (ham goruntu)                oran 0.62   benchmark 0.367-0.616
#   classic_f16_k3 tanh (eski sweep) oran 2.07   benchmark 0.011
#   classic_f32_k3 tanh (eski sweep) oran 2.91   benchmark 0.011
#
# Bu yuzden yeni sweep bilinen kazanan noktadan baslar ve onun cevresini
# tarar. Gecilecek baraj artik RAW degil, top_modeller'in kendisidir.
# =====================================================================

# Taban: diger_modeller/deneme_6 f32_k3 serisi (06.10.2022) ile katman katman
# ayni mimari -- 12.785 parametre, Conv 32/16/8 + ConvT 8/16/32 + Conv 1, k3,
# ELU, sigmoid -- ve 2022 Kaggle tarifi (zero_1, BCE, Adam 1e-3, batch 1).
# O serinin sifirdan egitilmis epoch-1 checkpointi vekilde 0.710 alir; ayni
# tarifin 2026 tekrari (tam veri, epoch 5) 0.809.
BASELINE_JOB = {
    "MODEL_TYPE": "flat_bottleneck",
    "FILTER_COUNT": 32,
    "KERNEL_SIZE": (3, 3),
    "STRIDES": (1, 1),
    "FLAT_DEPTH": 3,
    "HIDDEN_ACTIVATION": "elu",
    "OUTPUT_ACTIVATION": "sigmoid",
    # [0, 1] giris kazananlarin egitim normalizasyonudur. Benchmark yine
    # minus1_1 besler (BENCHMARK_INPUT_NORMALIZATION); bu fark bilerek korunur,
    # cunku kazananlarin Urgup'teki davranisi tam olarak bu farka dayanir.
    "INPUT_NORMALIZATION": "zero_1",
    "BATCH_SIZE": DEFAULT_BATCH_SIZE,
    # Her epoch yeniden karistir (2022: model.fit(numpy, shuffle=True)). Eski
    # egitim scripti yollari bir kez karistirip her epoch ayni sirayla
    # dolasiyordu; batch 1'de sira SGD yorungesinin kendisidir.
    "SHUFFLE_TRAIN": True,
    # 12 epoch: batch 1'de iyi checkpointler epoch 3-5'te cikti ve kalite
    # epochtan epocha dalgalanir; 12 epoch kapiya 12 aday verir. TF 2.10 bu
    # GPU'da batch 1'de ~122 ms/adim (paylasimli GPU, 2026-09-24 olcumu),
    # 5.218 adim/epoch -> ~10.6 dk/epoch, 12 epoch ~2.1 saat.
    #
    # Eski not (2026-09-09, batch 2, minus1_1): job 1'in en ayirt edici
    # checkpointi epoch 2'ydi ve epoch 8'den sonrasi degersizdi. Bu, eski
    # tarifin (batch 2) davranisidir; batch 1'de tepe daha gec ve cok daha
    # yuksek cikar.
    "EPOCHS": 12,
    "LEARNING_RATE": 0.001,
    "OPTIMIZER": "adam",
    # 2026-09-08 loss probe'u (models_pipeline/_loss_probe/probe_summary.json)
    # dort kaybi ayni mimaride 6 epoch olctu: binary_crossentropy 10/10
    # checkpointte AYAKTA (en iyi oran 0.347, kazananlarin 0.32-0.44 bandi
    # icinde), weighted_hybrid / hybrid / mae ise 0/10 -- epoch 1'den itibaren
    # gercek NCC 0.0, yani sabit siyah cikti. 2022 kazanani da
    # (arsiv/autoencoder_gpu_froom_kaggle.py:178) BinaryCrossentropy ile
    # derlenmisti. Sigmoid doygunlugunda gradyani kaybolmayan tek kayip bu
    # oldugu icin taban deger BCE'dir; hybrid/ssim isleri eksen kontrolu
    # olarak kaliyor.
    "LOSS_FUNCTION": "binary_crossentropy",
    "PRETRAINED_MODEL": None,
    # Erken durdurma KAPALI, bilerek. Egitim scriptinin varsayilani
    # patience=8 / monitor=val_loss / restore_best_weights=True idi ve bu
    # anahtar daha once CONFIG_KEYS'te olmadigi icin pipeline ona hic
    # dokunmuyordu. Iki sonucu vardi:
    #   1. "100 epoch = 522k ornek" butcesi hic gerceklesmeyebilirdi;
    #      val_loss 8 epoch duzelmezse egitim erken kesilirdi.
    #   2. restore_best_weights, son_model.h5'i val_loss-optimal agirliga
    #      geri dondururdu.
    # Ikisi de val_loss'u hedef metrik sayar; oysa bu sweepte hedef
    # ayirt ediciliktir (false_peak_ratio) ve mimari sinifi karsilastirmasi
    # tam olarak iyi rekonstruksiyonun eslesmeyi bozdugunu soyler. Epoch
    # butcesi olculerek secildigi icin val_loss'un onu kismasi istenmez;
    # her epoch diske yazildigindan (SAVE_CHECKPOINTS) en iyi nokta
    # tahmin edilmek yerine sonradan olculur.
    "EARLY_STOPPING_PATIENCE": 0,
    # LR programi bu sweepte taranan bir eksen degil. Sabit tutmak, isler
    # arasindaki farkin taranan eksenden geldigini garanti eder; val_loss'a
    # bagli otomatik LR dususu bunu gorunmez bicimde bozardi.
    "REDUCE_LR_PATIENCE": 0,
}


def job(name, **overrides):
    """Temel konfigurasyondan kontrollu bir deney tanimi olustur."""
    result = {"name": name, **BASELINE_JOB}
    result.update(overrides)
    return result


# Her is, taban noktadan TEK bir eksende ayrilir (tekrarlar hic ayrilmaz);
# boylece sonucun hangi degiskenden kaynaklandigi yorumlanabilir. Isler
# bilgi degerine gore siralidir: --max-jobs N bastan N is calistirir.
#
# Adlar 2026-09-10 sweepinden bilerek farklidir (z1_/m11_ oneki): eski bir
# run klasoru resume edilse bile ayni adli, farkli tarifle tamamlanmis bir
# benchmark yeni isin sonucu gibi benimsenmez.
TRAINING_JOBS = [
    # Tarifin kendisi.
    job("z1_f32_k3_b1"),

    # TEKRARLAR: ayni konfigurasyon, farkli rastgele agirlik baslangici
    # (egitim scripti TF seed'ini sabitlemez). batch 1'de checkpoint kalitesi
    # baslangica gore de dalgalanir: 3100 orneklik iki seed'in en iyi kapi
    # orani 0.42 ve 0.51 cikti. Her tekrar kapiya 12 yeni aday verir.
    job("z1_f32_k3_b1_tekrar2"),
    job("z1_f32_k3_b1_tekrar3"),

    # BATCH EKSENI KONTROLU. 2 eski pipeline varsayilaniydi; 8'de vekil
    # epoch 1-5 boyunca 0.27-0.31'de kaldi.
    job("z1_f32_k3_b2", BATCH_SIZE=2),
    job("z1_f32_k3_b8", BATCH_SIZE=8),

    # NORMALIZASYON KONTROLU: batch 1'de eski pipeline girisi.
    job("m11_f32_k3_b1", INPUT_NORMALIZATION="minus1_1"),

    # 2022 serilerinin diger noktalari: Ekim 2022 f40_k3 ve f32_k5 serileri
    # de ilk epochlarda iyiydi (urgup_bes_nokta_global_100); k_32_3 serisi
    # ReLU'luydu. f48_k4 (TPU mimarisi) batch 1'de 6 epochta vekil 0.343'te
    # kaldigi icin listede yok.
    job("z1_f40_k3_b1", FILTER_COUNT=40),
    job("z1_f32_k5_b1", KERNEL_SIZE=(5, 5)),
    job("z1_f32_k3_relu_b1", HIDDEN_ACTIVATION="relu"),

    # KARISTIRMA KONTROLU: eski egitim scriptinin her epoch ayni sirasi.
    job("z1_f32_k3_b1_sabit_sira", SHUFFLE_TRAIN=False),
]


CONFIG_KEYS = (
    "MODEL_TYPE",
    "FILTER_COUNT",
    "KERNEL_SIZE",
    "STRIDES",
    "FLAT_DEPTH",
    "HIDDEN_ACTIVATION",
    "OUTPUT_ACTIVATION",
    "INPUT_NORMALIZATION",
    "BATCH_SIZE",
    "SHUFFLE_TRAIN",
    "EPOCHS",
    "LEARNING_RATE",
    "OPTIMIZER",
    "LOSS_FUNCTION",
    "PRETRAINED_MODEL",
    "EARLY_STOPPING_PATIENCE",
    "REDUCE_LR_PATIENCE",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Egitimleri ve 1000 sorguluk benchmarklari guvenli resume ile calistir."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--new-run",
        action="store_true",
        help="Mevcut yarim calismayi kullanmadan yeni run-id olustur.",
    )
    group.add_argument(
        "--run-id",
        help="Belirli bir YYYYMMDD_HHMMSS run-id ile devam et.",
    )
    # Butce kontrolu. Bir is ~8.9 saat egitim + ~2.5 saat benchmark surer,
    # yani 15 isin tamami tek GPU'da yaklasik bir hafta demektir. Isler
    # bilgi degerine gore sirali oldugu icin bastan N is calistirmak
    # anlamli bir kismi sonuc verir.
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Yalniz ilk N gorevi calistir (isler bilgi degerine gore sirali).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Yalniz adi verilen gorevi calistir; birden fazla kez verilebilir.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Gorev listesini ve tahmini maliyeti yazdirip cik.",
    )
    return parser.parse_args(argv)


# Olculen maliyetler (bu GPU).
MEASURED_MINUTES_PER_EPOCH = 5.3            # flat_bottleneck f48/k4/d3, batch 2
MEASURED_CLASSIC_MINUTES_PER_EPOCH = 9.2    # classic f32/k3, ayni GPU
# batch 1: TF 2.10 f32/k3/d3 adim basina ~122 ms (paylasimli GPU,
# 2026-09-24), 5.218 adim/epoch. Bu batch'te sure agirlikla adim yukunden
# gelir; mimari maliyeti yalniz kismen etkiler (bkz. estimate_minutes_per_epoch).
MEASURED_B1_MINUTES_PER_EPOCH = 10.6        # flat_bottleneck f32/k3/d3, batch 1
# 1000 sorgu, iki yon: benchmark.log "MODEL TARAMA SURESI" yon basina ~48 dk
# (flat_f24_k4, 2026-09-10). Onceki 50 dk degeri tek yonu sayiyordu.
BENCHMARK_MINUTES_PER_CHECKPOINT = 96.0
# Benchmark'a giren aday: YALNIZ marj kapisinin sectigi sampiyon.
# 2026-09-09'da olculdu: job 1'in best_val_loss / best_mae / best_ssim
# dosyalarinin oranlari 1.155 / 1.50 / 1.71, yani olculen 16 checkpointin en
# kotuleri arasinda; hepsi PIKSEL rekonstruksiyonuna gore secildigi icin
# ayirt ediciligi olcmuyorlar. Dordunu benchmark'a sokmak is basina
# 4 x 50 dk = ~3,3 saat GPU'yu bilinen kotu adaylara harcamak demekti.
BENCHMARK_CHECKPOINTS_PER_JOB = 1
MARGIN_GATE_MINUTES_PER_CHECKPOINT = 2.0  # 4 grup (2026-09-25): ~4x kirpma, CPU


def estimate_minutes_per_epoch(training_job):
    """Olculen f48/k4 maliyetini mimarinin gercek maliyetine gore olcekle.

    Onceki surum tek bir olculen degeri (5.3 dk) HER ise uyguluyordu. Oysa
    flat_bottleneck'te conv maliyeti kabaca kanal^2 * kernel alani ile buyur,
    yani flat_f96_k4 ~4x ve flat_f48_k7 ~3x pahalidir; ikisi de ciddi bicimde
    dusuk tahmin ediliyor ve 15 isin toplami oldugundan ucuz gorunuyordu.

    Bu bir OLCUM DEGIL, olculen tek noktadan turetilen bir tahmindir; derinlik
    (FLAT_DEPTH) etkisi kanal daralmasi nedeniyle kucuk oldugu icin ihmal
    edilmistir.
    """
    if str(training_job.get("MODEL_TYPE")) != "flat_bottleneck":
        return MEASURED_CLASSIC_MINUTES_PER_EPOCH
    if int(training_job["BATCH_SIZE"]) == 1:
        # batch 1'de olculen f32/k3 suresinin yarisi adim yuku sayilir;
        # yalniz kalan yarisi mimarinin conv maliyetiyle olceklenir.
        compute_scale = (
            (float(training_job["FILTER_COUNT"]) / 32.0) ** 2
            * (float(training_job["KERNEL_SIZE"][0]) / 3.0) ** 2
        )
        return MEASURED_B1_MINUTES_PER_EPOCH * (0.5 + 0.5 * compute_scale)
    filter_scale = (float(training_job["FILTER_COUNT"]) / 48.0) ** 2
    kernel_scale = (float(training_job["KERNEL_SIZE"][0]) / 4.0) ** 2
    return MEASURED_MINUTES_PER_EPOCH * filter_scale * kernel_scale


def estimate_job_hours(training_job):
    train = estimate_minutes_per_epoch(training_job) * int(training_job["EPOCHS"])
    gate = MARGIN_GATE_MINUTES_PER_CHECKPOINT * (MARGIN_GATE_MAX_EPOCH_CHECKPOINTS + 4)
    benchmark = BENCHMARK_MINUTES_PER_CHECKPOINT * BENCHMARK_CHECKPOINTS_PER_JOB
    return (train + gate + benchmark) / 60.0


def select_jobs(args):
    """--only / --max-jobs filtrelerini uygula ve secilen gorevleri dondur."""
    jobs = list(TRAINING_JOBS)
    if args.only:
        by_name = {training_job["name"]: training_job for training_job in jobs}
        unknown = [name for name in args.only if name not in by_name]
        if unknown:
            raise ValueError(
                f"Bilinmeyen gorev adi: {unknown}. "
                f"Gecerli adlar: {sorted(by_name)}"
            )
        jobs = [by_name[name] for name in args.only]
    if args.max_jobs is not None:
        if args.max_jobs < 1:
            raise ValueError("--max-jobs en az 1 olmali.")
        jobs = jobs[: args.max_jobs]
    return jobs


def print_job_plan(jobs):
    total = sum(estimate_job_hours(training_job) for training_job in jobs)
    print(f"Gorev sayisi: {len(jobs)} | tahmini toplam: {total:.1f} saat")
    print(
        f"{'#':>3} {'gorev':30} {'mimari':16} {'f':>4} {'k':>2} {'d':>2} "
        f"{'cikis':8} {'kayip':20} {'norm':9} {'b':>3} {'ep':>4} {'saat':>6}"
    )
    for position, training_job in enumerate(jobs, start=1):
        print(
            f"{position:>3} {training_job['name']:30} "
            f"{training_job['MODEL_TYPE']:16} "
            f"{training_job['FILTER_COUNT']:>4} "
            f"{training_job['KERNEL_SIZE'][0]:>2} "
            f"{training_job['FLAT_DEPTH']:>2} "
            f"{training_job['OUTPUT_ACTIVATION']:8} "
            f"{training_job['LOSS_FUNCTION']:20} "
            f"{training_job['INPUT_NORMALIZATION']:9} "
            f"{training_job['BATCH_SIZE']:>3} "
            f"{training_job['EPOCHS']:>4} "
            f"{estimate_job_hours(training_job):>6.1f}"
        )


def configure_run_paths(run_id):
    global PIPELINE_RUN_ID, PIPELINE_RUN_DIR, PIPELINE_MODELS_ROOT, PIPELINE_BENCHMARK_ROOT
    if not re.fullmatch(r"\d{8}_\d{6}", run_id):
        raise ValueError(f"Gecersiz pipeline run-id: {run_id}")
    PIPELINE_RUN_ID = run_id
    run_dir = DENEYLER_DIR / f"{PIPELINE_RUN_PREFIX}{run_id}"
    legacy_models = MODELS_PIPELINE_DIR / run_id
    legacy_outputs = Path(BENCHMARK_CWD) / "outputs" / f"end_to_end_pipeline_{run_id}"
    if not run_dir.exists() and (legacy_models.exists() or legacy_outputs.exists()):
        PIPELINE_RUN_DIR = None
        PIPELINE_MODELS_ROOT = legacy_models
        PIPELINE_BENCHMARK_ROOT = legacy_outputs
    else:
        PIPELINE_RUN_DIR = run_dir
        PIPELINE_MODELS_ROOT = run_dir / "modeller"
        PIPELINE_BENCHMARK_ROOT = run_dir / "sonuclar"


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def benchmark_completion_status(output_dir):
    """Yalniz tam ve dogrulanmis benchmark ciktilarini tamamlanmis say."""
    required = (
        output_dir / "results.jsonl",
        output_dir / "summary.json",
        output_dir / "summary_metadata.json",
        output_dir / "summary_incomplete.json",
        output_dir / "benchmark_results.xlsx",
        output_dir / "excel_validation.json",
    )
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return False, f"eksik_ciktilar={missing}"

    metadata = read_json(output_dir / "summary_metadata.json", {})
    incomplete = read_json(output_dir / "summary_incomplete.json", None)
    excel_validation = read_json(output_dir / "excel_validation.json", {})
    if metadata.get("expected_queries_per_group") != BENCHMARK_MAX_QUERIES:
        return False, "sorgu_sayisi_uyusmuyor"
    if metadata.get("incomplete_group_count") != 0 or incomplete not in ([], {}):
        return False, "eksik_benchmark_gruplari_var"
    if metadata.get("group_count", 0) <= 0:
        return False, "ozet_grubu_yok"
    if excel_validation.get("zip_integrity") != "ok":
        return False, "excel_zip_dogrulamasi_yok"
    if excel_validation.get("formula_error_literals"):
        return False, "excel_formul_hatasi_var"
    return True, "tamamlandi"


def raw_baseline_completion_status(output_dir):
    """Iki yonun tum varyant/modlarinda RAW sonucunun tam oldugunu dogrula."""
    summary_path = output_dir / "summary.json"
    rows = read_json(summary_path, None)
    if not isinstance(rows, list):
        return False, "raw_ozeti_yok"
    raw_rows = [row for row in rows if row.get("model_id") == RAW_MODEL_ID]
    directions = {str(row.get("direction") or "") for row in raw_rows}
    expected_groups = {
        (direction, variant, mode)
        for direction in directions
        for variant in BENCHMARK_VARIANTS
        for mode in BENCHMARK_SEARCH_MODES
    }
    actual_groups = {
        (
            str(row.get("direction") or ""),
            str(row.get("query_variant") or ""),
            str(row.get("search_mode") or ""),
        )
        for row in raw_rows
        if int(row.get("total_queries") or 0) == BENCHMARK_MAX_QUERIES
    }
    if len(directions) != 2:
        return False, f"raw_yon_sayisi={len(directions)}"
    if actual_groups != expected_groups:
        return False, (
            f"raw_grup_sayisi={len(actual_groups)}/"
            f"{len(expected_groups)}"
        )
    return True, "tamamlandi"


def shared_raw_marker_path():
    return PIPELINE_BENCHMARK_ROOT / "_shared_raw_baseline.json"


def register_shared_raw_baseline(output_dir):
    complete, reason = raw_baseline_completion_status(output_dir)
    if not complete:
        raise RuntimeError(f"Ortak RAW baseline gecersiz: {reason} | {output_dir}")
    write_json_atomic(
        shared_raw_marker_path(),
        {
            "status": "complete",
            "run_id": PIPELINE_RUN_ID,
            "source_output_dir": str(output_dir.resolve()),
            "max_queries": BENCHMARK_MAX_QUERIES,
            "directions": 2,
            "query_variants": list(BENCHMARK_VARIANTS),
            "search_modes": list(BENCHMARK_SEARCH_MODES),
            "registered_at": datetime.datetime.now().isoformat(),
        },
    )
    return output_dir


def find_shared_raw_baseline():
    marker = read_json(shared_raw_marker_path(), {})
    source = marker.get("source_output_dir")
    if source:
        source_dir = Path(source)
        complete, _ = raw_baseline_completion_status(source_dir)
        if complete:
            return source_dir

    # Diske bakilir, is listesine degil: RAW baseline is listesinden once
    # baska adla tamamlanmis bir gorevde durabilir ve tekrar hesaplanmasi
    # gereksiz olur.
    if PIPELINE_BENCHMARK_ROOT.is_dir():
        for candidate in sorted(
            path
            for path in PIPELINE_BENCHMARK_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        ):
            complete, _ = raw_baseline_completion_status(candidate)
            if complete:
                return register_shared_raw_baseline(candidate)
    return None


def benchmark_include_raw_setting(output_dir, shared_raw_dir):
    """Mevcut kismi run imzasini koru; yeni runlarda RAW'i ortaklastir."""
    run_config = read_json(output_dir / "run_config.json", None)
    if isinstance(run_config, dict) and "include_raw" in run_config:
        include_raw = bool(run_config["include_raw"])
        if include_raw and shared_raw_dir is not None:
            print(
                "LEGACY BENCHMARK RESUME | Bu gorev RAW acik olarak baslamis; "
                "imzayi bozmamak icin yalniz bu kismi gorev ayni ayarla tamamlanacak."
            )
        return include_raw
    return shared_raw_dir is None


def build_job_status_rows(selected_names=None):
    """Gorev listesini durumlariyla birlikte birlesik rapora ver."""
    rows = []
    for position, training_job in enumerate(TRAINING_JOBS, start=1):
        output_dir = PIPELINE_BENCHMARK_ROOT / training_job["name"]
        model_dir = PIPELINE_MODELS_ROOT / training_job["name"]
        complete, _ = benchmark_completion_status(output_dir)
        if complete:
            state = "tamamlandi"
        elif (output_dir / "_pipeline_job_collapsed.json").is_file():
            state = "COKTU (sabit cikti)"
        elif reusable_training_job(training_job, model_dir) is not None:
            state = "egitim tamam, benchmark bekliyor"
        elif model_files_in(model_dir):
            state = "egitim yarim"
        else:
            state = "baslamadi"
        if selected_names is not None and training_job["name"] not in selected_names:
            state = f"{state} | bu kosuda secilmedi"
        rows.append(
            {
                "position": position,
                "name": training_job["name"],
                "state": state,
                "MODEL_TYPE": training_job["MODEL_TYPE"],
                "FILTER_COUNT": training_job["FILTER_COUNT"],
                "KERNEL_SIZE": int(training_job["KERNEL_SIZE"][0]),
                "FLAT_DEPTH": training_job["FLAT_DEPTH"],
                "HIDDEN_ACTIVATION": training_job["HIDDEN_ACTIVATION"],
                "OUTPUT_ACTIVATION": training_job["OUTPUT_ACTIVATION"],
                "LOSS_FUNCTION": training_job["LOSS_FUNCTION"],
                "INPUT_NORMALIZATION": training_job["INPUT_NORMALIZATION"],
                "BATCH_SIZE": training_job["BATCH_SIZE"],
                "EPOCHS": training_job["EPOCHS"],
                "estimated_hours": round(estimate_job_hours(training_job), 1),
            }
        )
    # Bu run klasorunde is listesinden once, baska adlarla tamamlanmis
    # gorevler olabilir; rapor onlari da gostermeli.
    known = {training_job["name"] for training_job in TRAINING_JOBS}
    if PIPELINE_BENCHMARK_ROOT.is_dir():
        for output_dir in sorted(
            path
            for path in PIPELINE_BENCHMARK_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith("_") and path.name not in known
        ):
            complete, _ = benchmark_completion_status(output_dir)
            rows.append(
                {
                    "position": None,
                    "name": output_dir.name,
                    "state": (
                        "tamamlandi | guncel is listesinde yok"
                        if complete
                        else "yarim | guncel is listesinde yok"
                    ),
                }
        )
    return rows


def write_combined_excel(combined_rows, selected_names=None):
    """Tum gorevleri ve RAW baseline'i tek Excel dosyasinda topla.

    Bilimsel dogruluk kaynagi JSON/CSV'dir; bu rapor okuma kolayligi icindir,
    bu yuzden uretimi basarisiz olursa kosu durdurulmaz -- 11 saatlik bir
    egitimin ardindan rapor yazimi yuzunden cikmak kabul edilemez.
    """
    try:
        import combined_report

        actual_path, mode = combined_report.write_combined_workbook(
            PIPELINE_BENCHMARK_ROOT,
            combined_rows,
            run_id=PIPELINE_RUN_ID,
            status_rows=build_job_status_rows(selected_names),
        )
    except Exception as exc:  # noqa: BLE001 - rapor opsiyoneldir, kosu surer
        print(f"[BIRLESIK RAPOR] UYARI: Excel uretilemedi ({exc}); kosu devam ediyor.")
        return None
    if mode == "locked_copy":
        print(
            f"[BIRLESIK RAPOR] Ana dosya Excel'de acik; guncel rapor kopyaya "
            f"yazildi: {actual_path}"
        )
    else:
        print(f"[BIRLESIK RAPOR] Tum sonuclar tek dosyada: {actual_path}")
    return actual_path


def refresh_combined_summary(shared_raw_dir, selected_names=None):
    """Ortak RAW satirlarini ve tamamlanan tum modelleri tek yerde topla.

    RAW baseline henuz hesaplanmamis olsa bile model satirlari yazilir; boylece
    ilk gorev biter bitmez birlesik rapor acilip o ana kadarki tablo gorulebilir.
    """
    if not PIPELINE_BENCHMARK_ROOT.is_dir():
        return None
    combined_rows = []
    if shared_raw_dir is not None:
        for row in read_json(shared_raw_dir / "summary.json", []):
            if row.get("model_id") == RAW_MODEL_ID:
                combined_rows.append(
                    {**row, "pipeline_job": "shared_raw_baseline", "source_output_dir": str(shared_raw_dir)}
                )

    # TRAINING_JOBS'a degil DISKE bakilir: bir run klasorunde, is listesi o
    # gunden beri degismis olsa bile tamamlanmis her gorev bulunur. Aksi halde
    # eski adlarla (or. classic_f32_k3_baseline) tamamlanmis benchmarklar
    # birlesik raporda hic gorunmezdi.
    for output_dir in sorted(
        path
        for path in PIPELINE_BENCHMARK_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ):
        complete, _ = benchmark_completion_status(output_dir)
        if not complete:
            continue
        for row in read_json(output_dir / "summary.json", []):
            if row.get("model_id") != RAW_MODEL_ID:
                combined_rows.append(
                    {**row, "pipeline_job": output_dir.name, "source_output_dir": str(output_dir)}
                )

    if not combined_rows:
        return None

    json_path = PIPELINE_BENCHMARK_ROOT / "combined_summary.json"
    csv_path = PIPELINE_BENCHMARK_ROOT / "combined_summary.csv"
    write_json_atomic(
        json_path,
        {
            "schema_version": 1,
            "run_id": PIPELINE_RUN_ID,
            "shared_raw_source": str(shared_raw_dir) if shared_raw_dir else None,
            "raw_baseline_row_count": sum(
                row.get("model_id") == RAW_MODEL_ID for row in combined_rows
            ),
            "model_row_count": sum(
                row.get("model_id") != RAW_MODEL_ID for row in combined_rows
            ),
            "rows": combined_rows,
        },
    )
    fieldnames = []
    for row in combined_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temp_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temp_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    os.replace(temp_csv, csv_path)
    write_combined_excel(combined_rows, selected_names)
    return json_path


def latest_pipeline_run_id():
    candidates = set()
    if MODELS_PIPELINE_DIR.is_dir():
        candidates.update(
            path.name
            for path in MODELS_PIPELINE_DIR.iterdir()
            if path.is_dir() and re.fullmatch(r"\d{8}_\d{6}", path.name)
        )
    if DENEYLER_DIR.is_dir():
        pattern = rf"{re.escape(PIPELINE_RUN_PREFIX)}(\d{{8}}_\d{{6}})"
        for path in DENEYLER_DIR.iterdir():
            match = re.fullmatch(pattern, path.name)
            if path.is_dir() and match:
                candidates.add(match.group(1))
    return max(candidates) if candidates else None


def choose_run_id(args):
    if args.run_id:
        return args.run_id, "acik_run_id"
    if args.new_run:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "yeni"
    latest = latest_pipeline_run_id()
    if latest:
        return latest, "otomatik_resume"
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "yeni"


def replace_assignment(code, key, value):
    """Config atamasini tam bir kez degistir; kaynak kod kaymasinda fail-fast yap."""
    pattern = rf"(?m)^([ \t]*){re.escape(key)}\s*=.*$"
    replacement = lambda match: f"{match.group(1)}{key} = {value!r}"
    updated, count = re.subn(pattern, replacement, code)
    if count != 1:
        raise RuntimeError(
            f"Egitim scriptinde {key} icin tam bir atama bekleniyordu; bulunan={count}"
        )
    return updated


def build_training_code(source_code, training_job, model_output_dir):
    """Bir is icin kaynak egitim kodunu guvenli ve denetlenebilir bicimde uret."""
    code = replace_assignment(source_code, "DATA_DIR", DATA_DIR_PATH)
    for key in CONFIG_KEYS:
        code = replace_assignment(code, key, training_job[key])
    code = replace_assignment(code, "SAVE_DIR", str(model_output_dir))
    # Her epoch diske yazilir. Onceden False idi ve diske yalniz
    # best_val_loss / best_mae / best_ssim / son_model dusuyordu -- dordu de
    # PIKSEL rekonstruksiyonuyla secilmis noktalar. Bu sweepin kendi tezi
    # (bkz. NEDEN BU SWEEP BOYLE) iyi rekonstruksiyonun eslesmeyi bozdugunu
    # soyledigi icin val_loss ile false_peak_ratio buyuk olasilikla ters
    # korelasyonludur; en iyi ayirt ediciligin oldugu epoch hic kaydedilmeden
    # kaybolabilirdi ve 9 saatlik egitim geri alinamazdi.
    # Epoch dosyalari egitimden sonra EPOCH_CHECKPOINT_DIR altina tasinir,
    # boylece benchmark yuzlerce modele girmez.
    code = replace_assignment(code, "SAVE_CHECKPOINTS", True)

    timestamp_source = (
        "return datetime.datetime.strftime(datetime.datetime.now(), '%d_%m_%Y__%H_%M')"
    )
    timestamp_target = (
        f"return 'JOB_{training_job['name']}_' + "
        "datetime.datetime.strftime(datetime.datetime.now(), '%d_%m_%Y__%H_%M')"
    )
    if code.count(timestamp_source) != 1:
        raise RuntimeError("get_timestamp govdesi beklenen bicimde bulunamadi.")
    return code.replace(timestamp_source, timestamp_target)


def model_files_in(model_dir):
    suffixes = {".h5", ".hdf5", ".keras"}
    if not model_dir.is_dir():
        return []
    return sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


EPOCH_FILENAME_PATTERN = re.compile(r"_epoch_(\d+)_")


def epoch_checkpoint_dir(model_dir):
    return model_dir / EPOCH_CHECKPOINT_DIR


def partition_epoch_checkpoints(model_dir):
    """Epoch dosyalarini alt klasore tasi; ust klasorde secilmis adaylar kalsin.

    Egitim her epoch icin bir dosya yazar. Ust klasorde best_val_loss /
    best_mae / best_ssim / son_model kalir; epoch serisi yalnizca marj
    kapisinin olcumu ve sonradan inceleme icin saklanir.

    DIKKAT: bu ayirma, epoch dosyalarini benchmark'tan CIKARMAZ. Benchmark
    katalogu --model-dir'i ozyinelemeli tarar, yani alt klasor de sayilir --
    bu fonksiyonun onceki aciklamasi ("benchmark'a girmez") yanlisti ve
    2026-09-10'da job 1'in katalogu 25 dosya bulup 20'sini calistirmaya
    kalkarak bunu gosterdi. Benchmark'i tek adaya indiren sey
    stage_benchmark_champion + benchmark_model_dir ikilisidir.
    """
    moved = []
    target_dir = epoch_checkpoint_dir(model_dir)
    for path in model_files_in(model_dir):
        if not EPOCH_FILENAME_PATTERN.search(path.name):
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / path.name
        shutil.move(str(path), str(destination))
        moved.append(destination)
    if moved:
        print(
            f"Epoch checkpointleri ayrildi: {len(moved)} dosya -> "
            f"{target_dir} (benchmark'a girmez)"
        )
    return moved


def checkpoint_epoch_number(path):
    match = EPOCH_FILENAME_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def sample_epoch_checkpoints(model_dir, limit=MARGIN_GATE_MAX_EPOCH_CHECKPOINTS):
    """Epoch serisini esit araliklarla ornekle; son epoch her zaman dahil."""
    candidates = sorted(
        (
            path
            for path in model_files_in(epoch_checkpoint_dir(model_dir))
            if checkpoint_epoch_number(path) is not None
        ),
        key=checkpoint_epoch_number,
    )
    if limit < 1 or len(candidates) <= limit:
        return candidates
    step = (len(candidates) - 1) / (limit - 1) if limit > 1 else len(candidates)
    picked = {int(round(index * step)) for index in range(limit)}
    picked.add(len(candidates) - 1)
    return [candidates[index] for index in sorted(picked)]


def margin_gate_candidates(model_dir):
    """Kapinin olcecegi dosyalar: secilmis adaylarin tamami + epoch ornegi."""
    return list(model_files_in(model_dir)) + sample_epoch_checkpoints(model_dir)


def promote_margin_gate_champion(model_dir, gate_report):
    """En ayirt edici epoch checkpointini benchmark'a girecek klasore al.

    Kapinin olctugu sey benchmark'in success_25m sonucunu belirleyen niceliktir
    (margin_gate.py). Sampiyon zaten ust klasordeyse (best_* dosyalarindan biri)
    kopyalanacak bir sey yoktur.
    """
    results = (gate_report or {}).get("results") or {}
    ranked = sorted(
        (
            (float(entry["false_peak_ratio"]), name)
            for name, entry in results.items()
            if isinstance(entry, dict)
            and name != "RAW"
            and isinstance(entry.get("false_peak_ratio"), (int, float))
            and math.isfinite(float(entry["false_peak_ratio"]))
        ),
        key=lambda item: item[0],
    )
    if not ranked:
        return None
    best_ratio, best_name = ranked[0]
    if any(path.name == best_name for path in model_files_in(model_dir)):
        print(
            f"[MARJ KAPISI] En iyi aday zaten benchmark listesinde: "
            f"{best_name} (oran={best_ratio:.2f})"
        )
        return model_dir / best_name
    source = epoch_checkpoint_dir(model_dir) / best_name
    if not source.is_file():
        return None
    destination = model_dir / best_name
    shutil.copy2(str(source), str(destination))
    print(
        f"[MARJ KAPISI] En ayirt edici epoch benchmark'a eklendi: "
        f"{best_name} (oran={best_ratio:.2f})"
    )
    return destination


def stage_benchmark_champion(model_dir, champion_path):
    """Sampiyonu, benchmark'a verilecek tek-modelli klasore KOPYALA.

    Neden tasima degil kopyalama ve neden ayri klasor: benchmark katalogu
    --model-dir'i ozyinelemeli tarar, bu yuzden "digerlerini alt klasore tasi"
    yaklasimi ise yaramaz (bkz. BENCHMARK_CANDIDATE_DIR). Kaynak dosyalar
    yerinde kalir, boylece marj kapisi bir sonraki kosuda ayni checkpoint
    kumesini olcer ve olcumler kosular arasinda karsilastirilabilir kalir.

    Sampiyon secilemediyse (kapi olcum yapamadi) None doner ve benchmark eski
    davranisina, yani tum is klasorune duser -- kapi bir gorevi durdurmaz.
    """
    if champion_path is None:
        return None
    source = Path(champion_path)
    if not source.is_file():
        return None
    target_dir = model_dir / BENCHMARK_CANDIDATE_DIR
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target_dir / source.name))
    print(
        f"[MARJ KAPISI] Benchmark tek adayla calisacak: {source.name} -> "
        f"{target_dir.name}/"
    )
    return target_dir


def benchmark_model_dir(model_dir):
    """Benchmark'in --model-dir'i: sampiyon klasoru varsa o, yoksa is klasoru.

    Geri dusme bilincli: kapi olcum yapamadiysa sampiyon klasoru olusmaz ve
    benchmark eskiden oldugu gibi is klasorunun tamamini calistirir. Bu, kapi
    hatasinin gorevi sessizce bos benchmark'a cevirmesini onler.
    """
    candidate_dir = model_dir / BENCHMARK_CANDIDATE_DIR
    if candidate_dir.is_dir() and any(model_files_in(candidate_dir)):
        return candidate_dir
    return model_dir


def archive_partial_models(training_job, model_dir, attempt_batch_size):
    files = model_files_in(model_dir)
    epoch_dir = epoch_checkpoint_dir(model_dir)
    if not files and not epoch_dir.is_dir():
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = (
        PIPELINE_MODELS_ROOT
        / "_failed_attempts"
        / training_job["name"]
        / f"{timestamp}_batch_{attempt_batch_size}"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.move(str(path), str(archive_dir / path.name))
    # Yarim kalan epoch serisi de gitmeli; aksi halde bir sonraki denemenin
    # checkpointleriyle karisir ve marj kapisi iki kosuyu ayirt edemez.
    if epoch_dir.is_dir():
        shutil.move(str(epoch_dir), str(archive_dir / EPOCH_CHECKPOINT_DIR))
    print(f"[{training_job['name']}] Kismi modeller arsivlendi: {archive_dir}")


def run_training_process(training_job, temp_script_name, log_path):
    """Ciktisini canli goster; OOM metnini guvenli yeniden deneme icin yakala."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    oom_detected = False
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n=== {datetime.datetime.now().isoformat()} | "
            f"batch={training_job['BATCH_SIZE']} ===\n"
        )
        process = subprocess.Popen(
            [PYTHON_EXEC, "-u", temp_script_name],
            cwd=TRAINING_SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            if "ResourceExhaustedError" in line or "OOM when allocating" in line:
                oom_detected = True
        return_code = process.wait()
    return return_code, oom_detected


class TrainingOutOfMemory(RuntimeError):
    """Egitim GPU bellegine sigmadi; is atlanir, BATCH DEGISTIRILMEZ.

    Neden geri cekilme yok: BATCH_SIZE bu sweepin eksen degiskeni. Onceki iki
    surum de sessizce baska bir batch'e geciyordu -- ilki dogrudan 1'e, sonra
    yariya. 2026-09-10'da bce_b32 batch 32'de OOM aldi ve batch 16'ya dustu,
    yani bce_b16 ile AYNI deneyi "bce_b32" adiyla ~4,5 saat boyunca yeniden
    kosacakti: yanlis etiketli, kopya bir veri noktasi. "Batch 32 bu GPU'ya
    sigmiyor" bilgisinin kendisi bir olcumdur ve oyle kaydedilmelidir.
    """


def modify_and_run_training(training_job):
    print(f"\n[{training_job['name']}] Egitim baslatiliyor...")
    original_script_path = Path(TRAINING_SCRIPT_DIR) / TRAINING_SCRIPT_NAME
    model_output_dir = PIPELINE_MODELS_ROOT / training_job["name"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    temp_script_name = f"temp_autoencoder_unified_{training_job['name']}.py"
    temp_script_path = Path(TRAINING_SCRIPT_DIR) / temp_script_name
    source_code = original_script_path.read_text(encoding="utf-8")
    # Tek deneme: OOM'da batch DEGISTIRILMEZ (bkz. TrainingOutOfMemory).
    batch_attempts = [int(training_job["BATCH_SIZE"])]

    for attempt_index, batch_size in enumerate(batch_attempts, start=1):
        attempt_job = dict(training_job, BATCH_SIZE=batch_size)
        code = build_training_code(source_code, attempt_job, model_output_dir)
        temp_script_path.write_text(code, encoding="utf-8")
        print(
            f"[{training_job['name']}] Konfigurasyon | "
            f"model={attempt_job['MODEL_TYPE']} | filtre={attempt_job['FILTER_COUNT']} | "
            f"kernel={attempt_job['KERNEL_SIZE']} | batch={attempt_job['BATCH_SIZE']} | "
            f"gizli_aktivasyon={attempt_job['HIDDEN_ACTIVATION']} | "
            f"cikis_aktivasyonu={attempt_job['OUTPUT_ACTIVATION']} | "
            f"optimizer={attempt_job['OPTIMIZER']} | lr={attempt_job['LEARNING_RATE']} | "
            f"loss={attempt_job['LOSS_FUNCTION']} | epoch={attempt_job['EPOCHS']}"
        )
        log_path = (
            PIPELINE_MODELS_ROOT
            / "_logs"
            / f"{training_job['name']}_training.log"
        )
        try:
            return_code, oom_detected = run_training_process(
                attempt_job,
                temp_script_name,
                log_path,
            )
        finally:
            if temp_script_path.exists():
                temp_script_path.unlink()

        if return_code == 0:
            partition_epoch_checkpoints(model_output_dir)
            return model_output_dir, attempt_job
        if oom_detected:
            archive_partial_models(training_job, model_output_dir, batch_size)
            raise TrainingOutOfMemory(
                f"[{training_job['name']}] GPU OOM | BATCH_SIZE={batch_size} bu "
                f"GPU'ya sigmiyor. Is atlandi; batch degistirilmedi cunku eksen "
                f"degiskeni odur. Log: {log_path}"
            )
        raise RuntimeError(
            f"[{training_job['name']}] egitim hatasi; cikis_kodu={return_code}. "
            f"Log: {log_path}"
        )

    raise RuntimeError(f"[{training_job['name']}] egitim denemeleri tamamlanamadi.")


def validate_trained_models(training_job, model_dir):
    """Kaydedilen modeller benchmark'a girmeden cikis sozlesmesini dogrula."""
    print(f"\n[ON KONTROL] Kaydedilen modeller dogrulaniyor: {training_job['name']}")
    try:
        subprocess.run(
            [
                PYTHON_EXEC,
                MODEL_VALIDATOR_SCRIPT,
                "--model-dir",
                str(model_dir),
                "--expected-output-activation",
                training_job["OUTPUT_ACTIVATION"],
            ],
            cwd=BENCHMARK_CWD,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"[ON KONTROL] HATA: Model cikis sozlesmesi gecersiz. "
            f"Gorev: {training_job['name']}"
        )


# Gercek konum NCC'si bunun altindaysa model cikti uretmiyor demektir.
# margin_gate.ncc(), taraflardan biri sabit oldugunda (std ~ 0) 0.0 dondurur,
# yani cokmus bir modelin tum eslesme skorlari sifira duser ve oran inf olur.
COLLAPSED_TRUTH_NCC = 0.01


def collapsed_model_output(gate_report):
    """Model sabit cikti mi uretiyor -- yani goruntu bilgisi tamamen yok mu.

    2026-09-04 kosusunda flat_f48/f64/f96 isleri tam olarak boyle coktu:
    her checkpoint icin gercek NCC 0.0, oran inf. Bu durumda benchmark
    calistirmak checkpoint basina ~50 dk GPU'yu bilinen bir sifir icin harcar.
    Olcum yapilamadiysa (bos rapor) cokme VARSAYILMAZ; kosu normal surer.
    """
    results = (gate_report or {}).get("results") or {}
    measured = [
        entry
        for name, entry in results.items()
        if isinstance(entry, dict) and name != "RAW" and "truth_ncc_mean" in entry
    ]
    if not measured:
        return False
    return all(
        abs(float(entry["truth_ncc_mean"])) < COLLAPSED_TRUTH_NCC for entry in measured
    )


def run_margin_gate(training_job, model_dir):
    """Benchmark'tan once CPU'da ayirt ediciligi olc ve en iyi epochu sec.

    Benchmark bir checkpoint icin ~50 dk GPU harcar; marj kapisi ayni bilginin
    vekilini CPU'da bir dakikada verir. Olculen olcut, yanlis tepe / gercek
    tepe oranidir ve benchmark siralamasiyla uyusur (bkz. margin_gate.py).

    Kapi artik yalniz raporlamaz, SECER: olculen adaylar arasinda en ayirt
    edici olan epoch checkpointi benchmark listesine eklenir. Bunun sebebi,
    diske dusen best_val_loss / best_mae / best_ssim dosyalarinin hepsinin
    PIKSEL rekonstruksiyonuyla secilmis olmasi ve bu sweepin kendi olcumune
    gore iyi rekonstruksiyonun eslesmeyi bozmasidir; en iyi val_loss epochu
    genellikle en iyi ayirt edicilik epochu DEGILDIR.

    Kapi bir gorevi durdurmaz: olcum basarisiz olursa benchmark yine calisir,
    yalniz secim yapilmadan devam edilir.
    """
    gate_path = PIPELINE_BENCHMARK_ROOT / "_margin_gate" / f"{training_job['name']}.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = margin_gate_candidates(model_dir)
    if not candidates:
        print(f"[MARJ KAPISI] {training_job['name']} | olculecek checkpoint yok; atlandi.")
        return None
    print(
        f"\n[MARJ KAPISI] {training_job['name']} | {len(candidates)} checkpoint | "
        f"CPU, ~{len(candidates)} dk | bolge={MARGIN_GATE_AOI}"
    )
    command = [
        PYTHON_EXEC,
        MARGIN_GATE_SCRIPT,
        "--samples",
        str(MARGIN_GATE_SAMPLES),
        "--aoi",
        MARGIN_GATE_AOI,
        # Egitim normalizasyonu DEGIL: kapi benchmark'in olctugunu olcmeli.
        # zero_1 ile egitilen bir aday benchmark'ta minus1_1 ile beslenir;
        # kapiyi zero_1 ile calistirmak baska bir modeli olcmek olurdu (2022
        # kazananlari zero_1 kapisinda 0.80-2.66'ya, minus1_1'de 0.32-0.33'e
        # cikar/iner: outputs/margin_gate/top_zero1.json vs validate.json).
        "--normalization",
        BENCHMARK_INPUT_NORMALIZATION,
        "--variants",
        ",".join(MARGIN_GATE_VARIANTS),
        "--json-out",
        str(gate_path),
    ]
    if MARGIN_GATE_BIDIRECTIONAL:
        command.append("--bidirectional")
    for path in candidates:
        command.extend(["--model", str(path)])
    try:
        subprocess.run(command, cwd=BENCHMARK_CWD, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[MARJ KAPISI] UYARI: kapi calistirilamadi ({exc}); benchmark devam ediyor.")
        return None

    report = read_json(gate_path, {})
    results = report.get("results", {})
    raw_ratio = report.get("raw_false_peak_ratio")
    # RAW'i DISLA. Onceki hali RAW'i da listeye aliyordu, dolayisiyla model
    # oranlarinin hepsi RAW'dan kotu oldugunda "en iyi oran" olarak RAW'in
    # kendi baraj degerini yaziyordu: 2026-09-10'da job 1 icin log satiri
    # "en iyi oran=0.45 | RAW baraji=0.45" dedi, oysa en iyi MODEL 0.571'di.
    # Excel tarafi zaten dogruydu (combined_report.best_gate_ratio_by_job RAW
    # satirini atliyor), yanlis olan yalniz bu satir ve dondurulen ozetti.
    ratios = [
        float(entry["false_peak_ratio"])
        for name, entry in results.items()
        if isinstance(entry, dict)
        and name != "RAW"
        and "false_peak_ratio" in entry
    ]
    if not ratios or raw_ratio is None:
        return None
    best = min(ratios)
    passed = bool(report.get("passed"))
    print(
        f"[MARJ KAPISI] {training_job['name']} | en iyi oran={best:.2f} | "
        f"RAW baraji={float(raw_ratio):.2f} | gecen={len(report.get('passed', []))}"
        f"/{len(ratios)}"
    )
    if collapsed_model_output(report):
        print(
            f"[MARJ KAPISI] COKME: {training_job['name']} checkpointlerinin hicbiri "
            "goruntu bilgisi tasimiyor (sabit cikti, gercek NCC ~ 0). Benchmark "
            "atlanacak; bu durumda 1000 sorgu calistirmak yalniz GPU harcar."
        )
        return {
            "best_false_peak_ratio": best,
            "raw_false_peak_ratio": float(raw_ratio),
            "passed": passed,
            "collapsed": True,
        }
    if not passed:
        print(
            f"[MARJ KAPISI] DIKKAT: {training_job['name']} hicbir checkpoint ile "
            "RAW barajini gecmedi. Benchmark sonucunun da zayif olmasi beklenir."
        )
    champion_path = promote_margin_gate_champion(model_dir, report)
    champion_path, tta_info = consider_tta_champion(training_job, model_dir, champion_path, best)
    stage_benchmark_champion(model_dir, champion_path)
    return {
        "best_false_peak_ratio": best,
        "raw_false_peak_ratio": float(raw_ratio),
        "passed": passed,
        "collapsed": False,
        "tta": tta_info,
    }


def consider_tta_champion(training_job, model_dir, champion_path, champion_ratio):
    """Sampiyonun yansima TTA'li surumunu kapidan gecir; daha ayirt ediciyse onu sec.

    2026-09-25/26 olcumu (deneyler/2026-09-25_parametre_arama): bu sweepin tarifiyle
    egitilen modellerde 8 donusumlu TTA kapiyi 0.517 -> 0.482, resmi 1000 sorguluk
    global'i 0.564 -> 0.673 yapti. Ama 2022 kazananlarinin doygun ciktisinda TTA
    bozar (TPU cb62 0.440 -> 0.518). Bu yuzden karari her seferinde kapi verir.
    TTA modeli yalniz standart katmanlarla kurulur (deney_araclari/yansima_toplulugu.py),
    benchmark degismeden yukler. Hicbir hata gorevi durdurmaz: sampiyon aynen doner.
    """
    if not BENCHMARK_TTA_TRANSFORMS or champion_path is None:
        return champion_path, None
    champion = Path(champion_path)
    tta_dir = model_dir / "_tta"
    tta_path = tta_dir / f"{champion.stem}_tta{len(BENCHMARK_TTA_TRANSFORMS)}.h5"
    gate_path = PIPELINE_BENCHMARK_ROOT / "_margin_gate" / f"{training_job['name']}_tta.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    try:
        tta_dir.mkdir(parents=True, exist_ok=True)
        if not tta_path.is_file():
            subprocess.run(
                [PYTHON_EXEC, TTA_BUILDER_SCRIPT, str(champion),
                 "--transforms", ",".join(BENCHMARK_TTA_TRANSFORMS), "--out", str(tta_path)],
                cwd=BENCHMARK_CWD, check=True, env=env,
            )
        command = [
            PYTHON_EXEC, MARGIN_GATE_SCRIPT, "--samples", str(MARGIN_GATE_SAMPLES),
            "--aoi", MARGIN_GATE_AOI, "--normalization", BENCHMARK_INPUT_NORMALIZATION,
            "--variants", ",".join(MARGIN_GATE_VARIANTS), "--json-out", str(gate_path),
            "--model", str(tta_path),
        ]
        if MARGIN_GATE_BIDIRECTIONAL:
            command.append("--bidirectional")
        subprocess.run(command, cwd=BENCHMARK_CWD, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[TTA] UYARI: TTA adayi kurulamadi/olculemedi ({exc}); sampiyon aynen kullanilacak.")
        return champion_path, {"error": str(exc)}
    entry = (read_json(gate_path, {}) or {}).get("results", {}).get(tta_path.name, {})
    ratio = entry.get("false_peak_ratio") if isinstance(entry, dict) else None
    if not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)):
        print("[TTA] UYARI: TTA orani okunamadi; sampiyon aynen kullanilacak.")
        return champion_path, {"error": "oran_yok"}
    chosen = float(ratio) < float(champion_ratio)
    print(
        f"[TTA] {training_job['name']} | sampiyon={float(champion_ratio):.3f} | "
        f"TTA{len(BENCHMARK_TTA_TRANSFORMS)}={float(ratio):.3f} | "
        f"benchmark adayi={'TTA' if chosen else 'sampiyon'}"
    )
    info = {"ratio": float(ratio), "champion_ratio": float(champion_ratio), "chosen": chosen, "path": str(tta_path)}
    return (tta_path if chosen else champion_path), info


def benchmark_oom_detected(output_dir):
    """model_errors.jsonl icinde GPU bellek hatasi var mi.

    Benchmark bir model yuklenemez veya cikarim patlarsa hatayi
    model_errors.jsonl'e yazip varsayilan olarak sonraki modele gecer, yani
    cikis kodu 0 olabilir. classic_f96_k3 tam olarak boyle kayboldu:
    build_model_map icinde ResourceExhaustedError alip sessizce bos sonuc
    uretti. Bu yuzden OOM'u dosyadan tespit etmek gerekir.
    """
    errors_path = output_dir / "model_errors.jsonl"
    if not errors_path.is_file():
        return False
    try:
        content = errors_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "ResourceExhaustedError" in content or "OOM when allocating" in content


def run_benchmark(training_job, model_dir, *, include_raw):
    """Benchmark'i calistir; GPU OOM'da cikarim batch'ini kademeli dusur.

    Egitim tarafinda batch 2->1 geri cekilmesi vardi ama cikarim tarafinda
    yoktu; oysa classic_f96_k3'u olduren sey egitim degil, benchmark'in
    harita cikarimiydi (build_model_map, --batch-size 16 sabit).
    """
    job_name = training_job["name"]
    output_dir = PIPELINE_BENCHMARK_ROOT / job_name

    for attempt_index, batch_size in enumerate(BENCHMARK_BATCH_ATTEMPTS, start=1):
        print(
            f"\n[BENCHMARK] Modeller test ediliyor | gorev={job_name} | "
            f"max_sorgu={BENCHMARK_MAX_QUERIES} | "
            f"raw_baseline={'evet' if include_raw else 'hayir'} | "
            f"cikarim_batch={batch_size} (deneme {attempt_index}/{len(BENCHMARK_BATCH_ATTEMPTS)})"
        )
        command = [
            PYTHON_EXEC,
            BENCHMARK_SCRIPT,
            "--model-dir",
            str(benchmark_model_dir(model_dir)),
            "--output-dir",
            str(output_dir),
            "--max-queries",
            str(BENCHMARK_MAX_QUERIES),
            "--batch-size",
            str(batch_size),
        ]
        if not include_raw:
            command.append("--no-include-raw")

        failed = False
        try:
            subprocess.run(command, cwd=BENCHMARK_CWD, check=True)
        except subprocess.CalledProcessError:
            failed = True

        oom = benchmark_oom_detected(output_dir)
        complete, reason = benchmark_completion_status(output_dir)
        if complete:
            return
        if oom and attempt_index < len(BENCHMARK_BATCH_ATTEMPTS):
            print(
                f"[BENCHMARK] GPU OOM algilandi (gorev={job_name}); "
                f"cikarim batch={BENCHMARK_BATCH_ATTEMPTS[attempt_index]} ile "
                "yeniden deneniyor. Tamamlanmis sorgular checkpointten korunur."
            )
            continue
        if failed:
            raise RuntimeError(
                f"[BENCHMARK] HATA: Benchmark sirasinda hata olustu. Gorev: {job_name}"
            )
        raise RuntimeError(
            f"[BENCHMARK] HATA: Benchmark tamamlanmadi. Gorev: {job_name} | "
            f"neden={reason} | gpu_oom={oom}"
        )

    raise RuntimeError(
        f"[BENCHMARK] HATA: Tum cikarim batch denemeleri tukendi. Gorev: {job_name}"
    )


def training_marker_path(model_dir):
    return model_dir / "_training_complete.json"


def write_training_marker(training_job, model_dir):
    write_json_atomic(
        training_marker_path(model_dir),
        {
            "status": "complete",
            "completed_at": datetime.datetime.now().isoformat(),
            "job": {key: training_job[key] for key in ("name",) + CONFIG_KEYS},
            "model_files": [path.name for path in model_files_in(model_dir)],
        },
    )


def reusable_training_job(training_job, model_dir):
    marker = read_json(training_marker_path(model_dir), {})
    if marker.get("status") != "complete" or not model_files_in(model_dir):
        return None
    saved_job = marker.get("job", {})
    for key in CONFIG_KEYS:
        if key == "BATCH_SIZE":
            continue
        expected = training_job[key]
        actual = saved_job.get(key)
        if isinstance(expected, tuple):
            expected = list(expected)
        if actual != expected:
            return None
    saved_batch = int(saved_job.get("BATCH_SIZE", 0))
    if saved_batch not in {int(training_job["BATCH_SIZE"]), 1}:
        return None
    return dict(training_job, BATCH_SIZE=saved_batch)


def write_job_completion_marker(training_job, output_dir, adopted_existing=False):
    write_json_atomic(
        output_dir / "_pipeline_job_complete.json",
        {
            "status": "complete",
            "completed_at": datetime.datetime.now().isoformat(),
            "run_id": PIPELINE_RUN_ID,
            "job": {key: training_job[key] for key in ("name",) + CONFIG_KEYS},
            "benchmark_max_queries": BENCHMARK_MAX_QUERIES,
            "adopted_existing_completion": bool(adopted_existing),
        },
    )


SUPPORTED_OUTPUT_ACTIVATIONS = ("tanh", "sigmoid")


def validate_pipeline_configuration():
    names = [training_job["name"] for training_job in TRAINING_JOBS]
    if len(names) != len(set(names)):
        raise ValueError("TRAINING_JOBS icinde yinelenen gorev adi var.")
    if BENCHMARK_MAX_QUERIES != 1000:
        raise ValueError("Bu pipeline icin benchmark sorgu sayisi 1000 olmali.")
    if any(training_job["PRETRAINED_MODEL"] is not None for training_job in TRAINING_JOBS):
        raise ValueError("Tum pipeline egitimleri sifirdan baslamali.")
    # BATCH_SIZE artik sweepin BIRINCIL EKSENI (bkz. TRAINING_JOBS batch blogu),
    # bu yuzden tum islerde ayni olmasi SART DEGIL. Onceki kisit, batch'in
    # kazananin tek yeniden uretilmemis degiskeni oldugu olculunce anlamini
    # yitirdi. Korunan sey kontrol noktasi: taban batch'i kullanan bir is
    # bulunmali, yoksa eksenin karsilastirma zemini kalmaz.
    if not any(
        int(training_job["BATCH_SIZE"]) == DEFAULT_BATCH_SIZE
        for training_job in TRAINING_JOBS
    ):
        raise ValueError(
            f"Batch ekseninin kontrolu icin BATCH_SIZE={DEFAULT_BATCH_SIZE} "
            "kullanan en az bir gorev olmali."
        )
    if any(int(training_job["BATCH_SIZE"]) < 1 for training_job in TRAINING_JOBS):
        raise ValueError("BATCH_SIZE en az 1 olmali.")
    # Egitim uzunlugunu belirleyen her sey pipeline'da acikca durmali. Bu iki
    # anahtar daha once CONFIG_KEYS disindaydi, yani egitim scriptinin
    # varsayilani (patience=8, monitor=val_loss, restore_best_weights=True)
    # sessizce yururlukteydi ve secilen epoch butcesini gecersiz kilabiliyordu.
    for key in ("EARLY_STOPPING_PATIENCE", "REDUCE_LR_PATIENCE"):
        values = {int(training_job[key]) for training_job in TRAINING_JOBS}
        if len(values) != 1:
            raise ValueError(
                f"{key} tum islerde ayni olmali; eksen karsilastirmasi bozulur: {values}"
            )
    if int(TRAINING_JOBS[0]["EARLY_STOPPING_PATIENCE"]) != 0:
        raise ValueError(
            "EARLY_STOPPING_PATIENCE=0 bekleniyor: val_loss bu sweepin hedef "
            "metrigi degil ve epoch butcesi olculerek secildi."
        )
    # Cikis aktivasyonu ile etiket araligi egitim scriptinde birbirine bagli
    # (OUTPUT_RANGES): tanh -> [-1, 1], sigmoid -> [0, 1]. Ikisi de gecerli.
    # sigmoid ozellikle desteklenir cunku benchmark'ta en yuksek success_25m
    # alan modellerin hepsi sigmoid ciktiliydi.
    invalid_outputs = [
        training_job["name"]
        for training_job in TRAINING_JOBS
        if str(training_job["OUTPUT_ACTIVATION"]).lower()
        not in SUPPORTED_OUTPUT_ACTIVATIONS
    ]
    if invalid_outputs:
        raise ValueError(
            f"OUTPUT_ACTIVATION {SUPPORTED_OUTPUT_ACTIVATIONS} icinden biri "
            f"olmali: {invalid_outputs}"
        )
    # binary_crossentropy YALNIZ [0, 1] etiket araliginda tanimlidir; tanh
    # ciktisinda etiketler [-1, 1] olur (OUTPUT_RANGES) ve negatif degerlerin
    # logaritmasi NaN verir. Kayip sessizce NaN'a gider, egitim epoch butcesini
    # sonuna kadar yakar ve marj kapisi ancak is bittikten sonra cokmus modeli
    # gorur -- flat_f48_tanh + classic kontrol icin 33 saat GPU demekti. Taban
    # kayip BCE yapildiginda bu kombinasyon kazayla olusabildigi icin
    # konfigurasyon duzeyinde reddedilir.
    bce_without_sigmoid = [
        training_job["name"]
        for training_job in TRAINING_JOBS
        if str(training_job["LOSS_FUNCTION"]).lower() == "binary_crossentropy"
        and str(training_job["OUTPUT_ACTIVATION"]).lower() != "sigmoid"
    ]
    if bce_without_sigmoid:
        raise ValueError(
            "binary_crossentropy yalniz OUTPUT_ACTIVATION='sigmoid' ile "
            f"gecerlidir (etiket araligi [0, 1]): {bce_without_sigmoid}"
        )
    # flat_bottleneck tam cozunurluk gerektirir; stride 2 sessizce uzamsal
    # indirme ekler ve mimarinin butun anlamini ortadan kaldirir.
    bad_strides = [
        training_job["name"]
        for training_job in TRAINING_JOBS
        if training_job.get("MODEL_TYPE") == "flat_bottleneck"
        and tuple(training_job.get("STRIDES", (1, 1))) != (1, 1)
    ]
    if bad_strides:
        raise ValueError(
            f"flat_bottleneck islerinde STRIDES=(1, 1) olmali: {bad_strides}"
        )


def main(argv=None):
    args = parse_args(argv)
    validate_pipeline_configuration()
    selected_jobs = select_jobs(args)
    # --list-jobs hicbir sey calistirmaz; plani ve olculen maliyeti yazip ciker.
    if args.list_jobs:
        print_job_plan(selected_jobs)
        return
    run_id, run_mode = choose_run_id(args)
    configure_run_paths(run_id)
    print("=== UCTAN UCA EGITIM VE BENCHMARK PIPELINE BASLIYOR ===")
    print(f"Pipeline run-id: {PIPELINE_RUN_ID}")
    print(f"Calisma modu: {run_mode}")
    filtered = len(selected_jobs) != len(TRAINING_JOBS)
    print(
        f"Egitim konfigurasyonu sayisi: {len(selected_jobs)}"
        + (f" / {len(TRAINING_JOBS)} (filtrelenmis)" if filtered else "")
    )
    print(f"Benchmark sorgu sayisi: {BENCHMARK_MAX_QUERIES}")
    print(
        f"Egitim batch: gorev basina (taban={DEFAULT_BATCH_SIZE}); "
        "OOM durumunda is atlanir, batch degistirilmez"
    )
    print(
        f"Cikarim batch merdiveni: {list(BENCHMARK_BATCH_ATTEMPTS)} "
        "(OOM durumunda dusurulur)"
    )
    print(
        "Tahmini toplam sure: "
        f"{sum(estimate_job_hours(j) for j in selected_jobs):.1f} saat"
    )

    PIPELINE_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    selected_names = {training_job["name"] for training_job in selected_jobs}
    print(
        "Birlesik rapor (her gorev sonunda yenilenir): "
        f"{PIPELINE_BENCHMARK_ROOT / 'TUM_SONUCLAR.xlsx'}"
    )
    shared_raw_dir = find_shared_raw_baseline()
    if shared_raw_dir is not None:
        print(f"Ortak RAW baseline bulundu; tekrar hesaplanmayacak: {shared_raw_dir}")
    else:
        print("Ortak RAW baseline yok; ilk tamamlanan benchmarkta bir kez hesaplanacak.")
    refresh_combined_summary(shared_raw_dir, selected_names)

    for position, training_job in enumerate(selected_jobs, start=1):
        print(f"\n=== GOREV {position}/{len(selected_jobs)} ({training_job['name']}) ===")
        model_dir = PIPELINE_MODELS_ROOT / training_job["name"]
        output_dir = PIPELINE_BENCHMARK_ROOT / training_job["name"]
        benchmark_complete, completion_reason = benchmark_completion_status(output_dir)

        if benchmark_complete:
            if not model_files_in(model_dir):
                raise RuntimeError(
                    f"[{training_job['name']}] Benchmark tamam ama model dosyalari eksik: {model_dir}"
                )
            saved_job = reusable_training_job(training_job, model_dir)
            if saved_job is None:
                saved_job = dict(training_job, BATCH_SIZE=LEGACY_BATCH_SIZE)
            validate_trained_models(saved_job, model_dir)
            write_job_completion_marker(saved_job, output_dir, adopted_existing=True)
            if shared_raw_dir is None:
                raw_complete, _ = raw_baseline_completion_status(output_dir)
                if raw_complete:
                    shared_raw_dir = register_shared_raw_baseline(output_dir)
            refresh_combined_summary(shared_raw_dir, selected_names)
            print(
                f"[{training_job['name']}] ATLANDI | tamamlanmis egitim ve "
                f"{BENCHMARK_MAX_QUERIES} sorguluk benchmark dogrulandi."
            )
            continue

        reusable_job = reusable_training_job(training_job, model_dir)
        if reusable_job is not None:
            print(
                f"[{training_job['name']}] Egitim daha once tamamlanmis; "
                f"benchmark kaldigi yerden devam edecek | neden={completion_reason}"
            )
            validate_trained_models(reusable_job, model_dir)
            actual_job = reusable_job
        else:
            if model_files_in(model_dir) or epoch_checkpoint_dir(model_dir).is_dir():
                archive_partial_models(
                    training_job,
                    model_dir,
                    training_job["BATCH_SIZE"],
                )
            try:
                model_dir, actual_job = modify_and_run_training(training_job)
            except TrainingOutOfMemory as exc:
                # Sweep durmaz: bu is icin "sigmadi" bilgisini yazip devam et.
                write_json_atomic(
                    output_dir / "_pipeline_job_oom.json",
                    {
                        "status": "oom",
                        "detected_at": datetime.datetime.now().isoformat(),
                        "run_id": PIPELINE_RUN_ID,
                        "job": {
                            key: training_job[key] for key in ("name",) + CONFIG_KEYS
                        },
                        "reason": str(exc),
                    },
                )
                print(f"[{training_job['name']}] GOREV ATLANDI | {exc}")
                refresh_combined_summary(shared_raw_dir, selected_names)
                continue
            validate_trained_models(actual_job, model_dir)
            write_training_marker(actual_job, model_dir)

        gate_result = run_margin_gate(actual_job, model_dir)
        if gate_result is not None and gate_result.get("collapsed"):
            write_json_atomic(
                output_dir / "_pipeline_job_collapsed.json",
                {
                    "status": "collapsed",
                    "detected_at": datetime.datetime.now().isoformat(),
                    "run_id": PIPELINE_RUN_ID,
                    "job": {key: actual_job[key] for key in ("name",) + CONFIG_KEYS},
                    "margin_gate": gate_result,
                    "reason": (
                        "Egitilen checkpointler sabit cikti uretiyor (gercek NCC ~ 0); "
                        "benchmark calistirilmadi."
                    ),
                },
            )
            print(
                f"[{training_job['name']}] BENCHMARK ATLANDI | model cokmus. "
                "Tani icin _pipeline_job_collapsed.json ve marj kapisi raporuna bakin."
            )
            refresh_combined_summary(shared_raw_dir, selected_names)
            continue
        # Kapi bir epoch checkpointini benchmark listesine eklemis olabilir;
        # o dosya da cikis sozlesmesini saglamadan benchmark'a girmemeli.
        if gate_result is not None:
            validate_trained_models(actual_job, model_dir)

        include_raw = benchmark_include_raw_setting(output_dir, shared_raw_dir)
        needed_new_shared_raw = shared_raw_dir is None
        run_benchmark(actual_job, model_dir, include_raw=include_raw)
        benchmark_complete, completion_reason = benchmark_completion_status(output_dir)
        if not benchmark_complete:
            raise RuntimeError(
                f"[{training_job['name']}] Benchmark sureci cikis=0 verdi fakat "
                f"tamamlanma kaniti gecersiz: {completion_reason}"
            )
        if needed_new_shared_raw and include_raw:
            shared_raw_dir = register_shared_raw_baseline(output_dir)
        elif not include_raw:
            raw_complete, _ = raw_baseline_completion_status(output_dir)
            if raw_complete:
                raise RuntimeError(
                    f"[{training_job['name']}] Ortak baseline varken RAW sonucu "
                    "yeniden uretilmis; tekrar hesaplama sozlesmesi ihlal edildi."
                )
        write_job_completion_marker(actual_job, output_dir)
        refresh_combined_summary(shared_raw_dir, selected_names)
        refresh_results_dashboard()
        print(f"\n=== [{training_job['name']}] GOREV TAMAMLANDI ===\n")

    print("=== TUM GOREVLER BASARIYLA TAMAMLANDI ===")
    print(f"Birlesik rapor: {PIPELINE_BENCHMARK_ROOT / 'TUM_SONUCLAR.xlsx'}")
    print(f"Tum kosularin panosu: {DENEYLER_DIR / 'PANO.md'}")


def refresh_results_dashboard():
    """deneyler/PANO.md + PANO.xlsx'i yeniler (tum kosular tek tabloda).

    En iyi caba: pano kosuyu asla durdurmaz. Kosu kokleri benchmark klasoru
    disindaysa (testlerin gecici klasorleri) gercek pano yazilmaz.
    """
    try:
        Path(PIPELINE_BENCHMARK_ROOT).resolve().relative_to(Path(BENCHMARK_CWD).resolve())
    except (TypeError, ValueError, OSError):
        return
    try:
        if BENCHMARK_CWD not in sys.path:
            sys.path.insert(0, BENCHMARK_CWD)
        import sonuc_panosu

        sonuc_panosu.main([])
    except Exception as exc:  # noqa: BLE001 - pano kosuyu durdurmamali
        print(f"[PANO] UYARI: sonuc panosu guncellenemedi ({exc}); kosu devam ediyor.")


if __name__ == "__main__":
    main()
