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
# Kapi ornek sayisi. 20 ornek siniflari ayirmaya yeter (dogrulandi), 60 ornek
# oran tahminini stabilize eder ve yine bir dakika altinda kalir.
MARGIN_GATE_SAMPLES = 60
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
MODELS_PIPELINE_DIR = Path(BENCHMARK_CWD) / "models_pipeline"
BENCHMARK_MAX_QUERIES = 1000
# Cikarim batch geri cekilme merdiveni. run_urgup_cevresi_models.py varsayilani
# 16'dir ve classic_f96_k3 tam olarak bu deger yuzunden build_model_map icinde
# GPU OOM alip kayboldu. --batch-size bilimsel degil operasyonel bir ayardir
# (benchmark/README.md), yani resume sirasinda degistirilmesi sonuclari bozmaz.
BENCHMARK_BATCH_ATTEMPTS = (16, 8, 4, 2)
DEFAULT_BATCH_SIZE = 2
LEGACY_BATCH_SIZE = 8
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
PIPELINE_MODELS_ROOT = None
PIPELINE_BENCHMARK_ROOT = None


# =====================================================================
# NEDEN BU SWEEP BOYLE
# =====================================================================
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

# 2022'de benchmark'ta 0.703 success_25m alan konfigurasyon
# (top_modeller/_03_model_tpu_f48_k4_s1_elu...sigmoid). Mimari birebir
# yeniden uretildi: 50.089 parametre, katman katman ayni.
BASELINE_JOB = {
    "MODEL_TYPE": "flat_bottleneck",
    "FILTER_COUNT": 48,
    "KERNEL_SIZE": (4, 4),
    "STRIDES": (1, 1),
    "FLAT_DEPTH": 3,
    "HIDDEN_ACTIVATION": "elu",
    "OUTPUT_ACTIVATION": "sigmoid",
    "INPUT_NORMALIZATION": "minus1_1",
    "BATCH_SIZE": DEFAULT_BATCH_SIZE,
    # Epoch butcesi OLCULDU (f48/k4/d3, batch 2, 2608 step/epoch, bu GPU):
    #   122 ms/step -> 5.3 dk/epoch
    #    50 epoch ->  4.4 saat/is      100 epoch ->  8.9 saat/is
    #   200 epoch -> 17.7 saat/is      400 epoch -> 35.5 saat/is
    # Karsilastirma: classic_f32 9.2 dk/epoch, yani flat_bottleneck tam
    # cozunurluge ragmen 1.7x daha hizli (kanal sayisi cok daha az).
    #
    # Eski basarisiz kosu 20 epoch = ~104k ornek gormustu. Kazanan 2022
    # modelleri "epoch 1100-2000" checkpointleriydi ama kucuk, RAM'e sigan bir
    # veri kumesi uzerinde; dolayisiyla dogru karsilastirma epoch degil GORULEN
    # ORNEK sayisidir. 100 epoch = ~522k ornek, basarisiz kosunun 5 kati.
    #
    # 100 epoch, is basina bir gecelik birim olacak sekilde secildi. Egitim
    # scripti her epoch best_val_loss / best_mae / best_ssim checkpointlerini
    # kaydeder, bu yuzden marj kapisi ara checkpointleri de degerlendirebilir;
    # en iyi epochu tahmin etmek yerine olcmek mumkundur.
    # 2026-09-09'da OLCULDU: BCE ile 75 epoch kosan job 1'in en ayirt edici
    # checkpointi EPOCH 2 (oran 0.363), epoch 8'den sonrasi degersiz ve epoch
    # 20'de truth NCC 0.003 -- yani 100 epoch bilgi tasiyan pencerenin ~50 kati.
    # 20 epoch iki sebeple secildi: (a) tepe bolgesini (1-8) fazlasiyla kapsar,
    # (b) batch ekseninin ust ucuna kazananin adim sayisini verir --
    # BATCH_SIZE=32'de 20 epoch ~3.260 gradyan adimi, tpu kazananinin ~3.300
    # adimiyla ayni mertebe. Epoch sayisi tum batch'lerde ayni tutulur, boylece
    # GORULEN ORNEK sabit kalir ve batch tek degisken olur.
    "EPOCHS": 20,
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


# Her is, bilinen kazanan noktadan TEK bir eksende ayrilir; boylece sonucun
# hangi degiskenden kaynaklandigi yorumlanabilir.
TRAINING_JOBS = [
    # Kazanan konfigurasyonun yeniden uretimi. Bu is basarisiz olursa
    # sorun mimaride degil egitim verisinde/tarifindedir; sweep'in geri
    # kalanini calistirmanin anlami yoktur.
    job("flat_f48_k4_sigmoid_repro"),

    # BATCH EKSENI -- kazananin yeniden uretilmemis TEK degiskeni.
    # tpu_model_f48_k4_epoch_01100 TPU'da egitildi ve arsivin kendi satiri
    # BATCH_SIZE = 128 * num_replicas (~1024) diyor: ~3.100 goruntu x 1100
    # epoch = ~3,4M ornek, ama yalnizca ~3.300 adimda. Bizim epoch-2 tepemiz
    # ~5.200 adim, yani adim sayisi ayni mertebede; fark bizim adimlarimizin
    # batch-2 gurultulu, onlarinkinin batch-1024 temiz gradyan olmasi.
    # lr_batch_probe yalniz batch 2 ve 8'i denedi. Epoch sayisi sabit oldugu
    # icin bu isler ayni ornegi gorur, yalniz gradyan gurultusu degisir.
    job("bce_b8", BATCH_SIZE=8),
    job("bce_b16", BATCH_SIZE=16),
    job("bce_b32", BATCH_SIZE=32),

    # Kapasite: kazanan nokta 50k parametrede. Olculen egilim, kapasite
    # arttikca ayirt ediciligin DUSTUGU yonunde; bu eksen o egilimin
    # kazanan bolgede nerede tepe yaptigini bulur.
    job("flat_f24_k4", FILTER_COUNT=24),
    job("flat_f32_k4", FILTER_COUNT=32),
    job("flat_f64_k4", FILTER_COUNT=64),
    job("flat_f96_k4", FILTER_COUNT=96),

    # Alici alan: kernel k ve derinlik d icin ~1 + 2*d*(k-1) piksel.
    # Kaynaklar-arasi dokunun hangi olcekte kararli oldugunu belirler.
    job("flat_f48_k3", KERNEL_SIZE=(3, 3)),
    job("flat_f48_k5", KERNEL_SIZE=(5, 5)),
    job("flat_f48_k7", KERNEL_SIZE=(7, 7)),

    # Darboga derinligi: f/2^d zinciri. d=2 daha genis darboga,
    # d=4 daha sikistirilmis temsil.
    job("flat_f48_d2", FLAT_DEPTH=2),
    job("flat_f48_d4", FLAT_DEPTH=4),

    # Cikis aktivasyonu kontrolu. Kazananlar sigmoid; bu is ayni mimariyi
    # tanh ile egiterek farkin gercekten aktivasyondan mi geldigini olcer.
    # BCE etiketleri [0, 1] ister, tanh ise [-1, 1] uretir (OUTPUT_RANGES), yani
    # bu is taban kaybi devralamaz; kaybi acikca weighted_hybrid'e sabitlenmistir
    # (bu isin BCE oncesindeki degeri). Sonuc olarak job 1 ile arasinda IKI
    # degisken farkeder, aktivasyon ve kayip; BCE matematiksel olarak sigmoid'e
    # bagli oldugu icin bu konfound kacinilmazdir.
    job("flat_f48_tanh", OUTPUT_ACTIVATION="tanh",
        LOSS_FUNCTION="weighted_hybrid"),

    # Kayip fonksiyonu. weighted_hybrid koyu piksellere agirlik verir;
    # bu iki is agirliklandirmanin ayirt ediciligi nasil etkiledigini olcer.
    job("flat_f48_hybrid_loss", LOSS_FUNCTION="hybrid"),
    job("flat_f48_ssim_loss", LOSS_FUNCTION="ssim"),

    # Gizli aktivasyon.
    job("flat_f48_relu", HIDDEN_ACTIVATION="relu"),

    # Kontrol grubu: eski sweep'in mimari sinifi, ayni epoch butcesiyle.
    # Kapasite/indirme hipotezini test eder ve eski sonucun 20 epochtan mi
    # yoksa mimariden mi kaynaklandigini ayirir.
    job("classic_f32_k3_control", MODEL_TYPE="classic", FILTER_COUNT=32,
        KERNEL_SIZE=(3, 3), OUTPUT_ACTIVATION="tanh",
        LOSS_FUNCTION="weighted_hybrid"),
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
BENCHMARK_MINUTES_PER_CHECKPOINT = 50.0
# Benchmark'a giren aday: YALNIZ marj kapisinin sectigi sampiyon.
# 2026-09-09'da olculdu: job 1'in best_val_loss / best_mae / best_ssim
# dosyalarinin oranlari 1.155 / 1.50 / 1.71, yani olculen 16 checkpointin en
# kotuleri arasinda; hepsi PIKSEL rekonstruksiyonuna gore secildigi icin
# ayirt ediciligi olcmuyorlar. Dordunu benchmark'a sokmak is basina
# 4 x 50 dk = ~3,3 saat GPU'yu bilinen kotu adaylara harcamak demekti.
BENCHMARK_CHECKPOINTS_PER_JOB = 1
MARGIN_GATE_MINUTES_PER_CHECKPOINT = 1.0


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
        f"{'cikis':8} {'kayip':16} {'ep':>4} {'saat':>6}"
    )
    for position, training_job in enumerate(jobs, start=1):
        print(
            f"{position:>3} {training_job['name']:30} "
            f"{training_job['MODEL_TYPE']:16} "
            f"{training_job['FILTER_COUNT']:>4} "
            f"{training_job['KERNEL_SIZE'][0]:>2} "
            f"{training_job['FLAT_DEPTH']:>2} "
            f"{training_job['OUTPUT_ACTIVATION']:8} "
            f"{training_job['LOSS_FUNCTION']:16} "
            f"{training_job['EPOCHS']:>4} "
            f"{estimate_job_hours(training_job):>6.1f}"
        )


def configure_run_paths(run_id):
    global PIPELINE_RUN_ID, PIPELINE_MODELS_ROOT, PIPELINE_BENCHMARK_ROOT
    if not re.fullmatch(r"\d{8}_\d{6}", run_id):
        raise ValueError(f"Gecersiz pipeline run-id: {run_id}")
    PIPELINE_RUN_ID = run_id
    PIPELINE_MODELS_ROOT = MODELS_PIPELINE_DIR / run_id
    PIPELINE_BENCHMARK_ROOT = (
        Path(BENCHMARK_CWD) / "outputs" / f"end_to_end_pipeline_{run_id}"
    )


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
    if not MODELS_PIPELINE_DIR.is_dir():
        return None
    candidates = sorted(
        path.name
        for path in MODELS_PIPELINE_DIR.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}_\d{6}", path.name)
    )
    return candidates[-1] if candidates else None


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
        "--normalization",
        str(training_job.get("INPUT_NORMALIZATION", "minus1_1")),
        "--json-out",
        str(gate_path),
    ]
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
    stage_benchmark_champion(model_dir, champion_path)
    return {
        "best_false_peak_ratio": best,
        "raw_false_peak_ratio": float(raw_ratio),
        "passed": passed,
        "collapsed": False,
    }


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
        f"Egitim batch: gorev basina (kontrol={DEFAULT_BATCH_SIZE}); "
        "OOM durumunda kademeli yariya iner"
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
        print(f"\n=== [{training_job['name']}] GOREV TAMAMLANDI ===\n")

    print("=== TUM GOREVLER BASARIYLA TAMAMLANDI ===")
    print(f"Birlesik rapor: {PIPELINE_BENCHMARK_ROOT / 'TUM_SONUCLAR.xlsx'}")


if __name__ == "__main__":
    main()
