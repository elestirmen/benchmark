#!/usr/bin/env python
"""Kayip fonksiyonu teshisi: 2022 kazananinin kaybi hic denenmedi.

NEDEN BU SCRIPT VAR
===================
lr_batch_probe.py lr/batch ekseninin sucsuz oldugunu gosterdi: b2 ayarlarinin
hepsi ve lr1e-4_b8 cokmeden calisti, ama en iyi oran 0.50 idi ve RAW'in ayni
bolgedeki 0.337'sinden kotuydu. Yani cokmenin sebebi optimizasyon rejimi degil.

Sonra 2026-09-08'de olculen sey:

    arsiv/autoencoder_gpu_froom_kaggle.py:178
        model.compile(optimizer='adam', loss='BinaryCrossentropy', ...)

0.703 alan 2022 modeli BinaryCrossentropy ile egitilmis. TRAINING_JOBS'taki 15
isin HICBIRI bunu kullanmiyor (13 weighted_hybrid, 1 hybrid, 1 ssim). Mimari
birebir yeniden uretildi, kayip hic yeniden uretilmedi.

Ayni gun olculen etiket dagilimi (sonuclar_19, 60 karo ornegi):

    ortalama parlaklik 0.847 | beyaz (>0.925) 0.847 | koyu (<=0.2) 0.153

Oldurulen flat_f96_k4'un son log satiri `mae: 0.8274 - mse: 0.8277`. mae ile
mse'nin dort hanede esit olmasi, neredeyse ikili bir etikete karsi cikti sabit
0 iken mumkundur (o zaman ikisi de beyaz piksel oranina, 0.847'ye esitlenir).
Ag saf siyah basiyordu.

MEKANIZMA (hipotez)
===================
Sabit siyah, agirlikli kayba gore EN KOTU sabittir (agirlikli L1: siyah 0.587,
beyaz 0.413, ortalama 0.335), yani ag oraya kaybi azaltarak yerlesmedi -- oraya
itildi ve oldu. weighted_hybrid koyu pikselleri 8x agirliklar
(WEIGHTED_DARK_WEIGHT = 8.0); bu, ciktiyi erken ve sert bicimde asagi bastirir,
sigmoid 0'da doyar ve gradyani sifirlanir. Kapasite bunun hizini belirler, ki
gozlenen sira birebir uyuyor: f96 epoch 1'de, f48 kismen canli, f24 epoch 77'ye
kadar dayandi. lr'nin yalnizca geciktirmesi de buradan gelir.

BCE, sigmoid ile birlestiginde gradyani (pred - true) olan tek kayiptir; yanlis
yonde doydugunda SIFIRLANMAZ. Kazananin cokmemesinin muhtemel sebebi bu.

NE YAPAR
========
Repro konfigurasyonunu (BASELINE_JOB: flat_bottleneck f48/k4/d3, sigmoid) sabit
tutar, YALNIZ LOSS_FUNCTION degistirir, her kombinasyonu birkac epoch egitir ve
HER epoch checkpointini marj kapisindan gecirir. Benchmark hic calistirilmaz.

Isler bilgi degerine gore sirali: bce ilk. Kosuyu yarida kesersen elinde yine
belirleyici cevap olur.

TRAINING_JOBS'a dokunmaz.

KULLANIM
========
    python loss_probe.py --list          # plani ve tahmini sureyi goster
    python loss_probe.py                 # tum kayiplari calistir
    python loss_probe.py --only bce      # yalniz kazananin kaybi
    python loss_probe.py --epochs 4      # daha ucuz tarama
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import end_to_end_pipeline as pipeline
import lr_batch_probe as harness


PROBE_ROOT = pipeline.MODELS_PIPELINE_DIR / "_loss_probe"
# Cokme f96'da epoch 1'de, f48'de ilk birkac epochta olculdu; alti epoch
# "cokuyor mu" ve "oran epochla iyilesiyor mu kotulesiyor mu" sorularinin
# ikisini de ayirmaya yeter. lr_batch_probe ile ayni butce, boylece iki
# probe'un satirlari dogrudan karsilastirilabilir.
DEFAULT_PROBE_EPOCHS = 6
PROBE_GATE_SAMPLES = 20

# Kazananin kaybi. Kontrol disindaki her satirin varlik sebebi bu satiri
# yorumlanabilir kilmak.
WINNER_LOSS = "binary_crossentropy"


def build_probe_jobs(epochs):
    """Repro konfigurasyonundan yalniz LOSS_FUNCTION ekseninde ayril."""
    jobs = [
        # Kazananin kaybi (arsiv/autoencoder_gpu_froom_kaggle.py:178). Sirada
        # ilk, cunku tek basina calisirsa tarif geri kazanilmis olur.
        harness.probe("bce", LOSS_FUNCTION=WINNER_LOSS),
        # Kontrol: mevcut sweep ayari. f96'da epoch 1'de coktugu olculdu;
        # ayni butce ve ayni olcumle tekrar edilmesi diger satirlari
        # yorumlanabilir kilar.
        harness.probe("weighted_hybrid_kontrol", LOSS_FUNCTION="weighted_hybrid"),
        # Ayni L1+SSIM karisimi, koyu piksel agirliklandirmasi OLMADAN. Bu
        # satir, sucun 8x agirlikta mi yoksa L1 ailesinin kendisinde mi
        # oldugunu ayirir.
        harness.probe("hybrid", LOSS_FUNCTION="hybrid"),
        # Duz L1: SSIM teriminin katkisini ayirir.
        harness.probe("mae", LOSS_FUNCTION="mae"),
    ]
    for job in jobs:
        job["EPOCHS"] = epochs
    return jobs


def estimate_probe_minutes(job):
    """Kayip fonksiyonu adim maliyetini kayda deger bicimde degistirmez."""
    train = pipeline.estimate_minutes_per_epoch(job) * int(job["EPOCHS"])
    gate = (int(job["EPOCHS"]) + 1) * 0.35  # 20 ornekle kapi ~20 sn/checkpoint
    return train + gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Kayip fonksiyonu cokme teshisi; benchmark calistirmaz."
    )
    parser.add_argument("--list", action="store_true", help="Plani yazip cik.")
    parser.add_argument("--only", action="append", default=[], help="Yalniz bu kayip.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_PROBE_EPOCHS)
    parser.add_argument("--gate-samples", type=int, default=PROBE_GATE_SAMPLES)
    return parser.parse_args(argv)


def print_plan(jobs):
    total = sum(estimate_probe_minutes(job) for job in jobs)
    print(f"Kayip sayisi: {len(jobs)} | tahmini toplam: {total / 60.0:.1f} saat")
    print(f"{'ad':24} {'kayip':20} {'epoch':>6} {'~dk':>6}")
    for job in jobs:
        print(
            f"{job['name']:24} {job['LOSS_FUNCTION']:20} "
            f"{job['EPOCHS']:>6} {estimate_probe_minutes(job):>6.0f}"
        )


def print_summary(summaries):
    print()
    print("=" * 82)
    print("TESHIS SONUCU (oran: kucuk = ayirt edici; gercek NCC ~ 0 = model cokmus)")
    print("=" * 82)
    raw = next(
        (s["raw_false_peak_ratio"] for s in summaries if s.get("raw_false_peak_ratio")),
        None,
    )
    if raw is not None:
        print(f"RAW baraji: oran {raw:.2f}\n")
    print(f"{'ad':24} {'kayip':20} {'saglam':>7} {'en iyi oran':>12} {'epoch':>6}")
    print("-" * 82)
    for summary in summaries:
        if summary.get("status") and summary["status"] != "ok":
            print(f"{summary['name']:24} {'':20} {summary['status']}")
            continue
        best = summary["best"]
        ratio = f"{best['false_peak_ratio']:.2f}" if best else "-"
        epoch = str(best["epoch"]) if best and best["epoch"] is not None else "-"
        healthy = f"{summary['healthy_checkpoints']}/{len(summary['checkpoints'])}"
        flag = "  <- COKTU" if summary["collapsed"] else ""
        print(
            f"{summary['name']:24} {summary['loss_function']:20} "
            f"{healthy:>7} {ratio:>12} {epoch:>6}{flag}"
        )
    print()

    winner = next(
        (s for s in summaries if s.get("loss_function") == WINNER_LOSS), None
    )
    if winner is None or winner.get("status") != "ok":
        print("Kazananin kaybi olculemedi; asagidaki yorum gecersiz.")
        return
    if winner["collapsed"]:
        print(
            "Kazananin kaybi da cokuyor. O halde suclu kayip DEGIL: siradaki\n"
            "adaylar egitim verisinin kendisi (DATA_DIR eslesmesi, karo hizalamasi)\n"
            "ve mimarinin kazanandan farkli kalan yanlaridir."
        )
        return
    ratio = winner["best"]["false_peak_ratio"]
    print(f"BCE cokmeden cikti | en iyi oran {ratio:.2f} (epoch {winner['best']['epoch']})")
    if raw is not None and ratio >= raw:
        print(
            f"Ama RAW barajini ({raw:.2f}) hala gecmiyor. Kayip cokmeyi cozdu,\n"
            "ayirt ediciligi cozmedi; sweepi bu tabanla baslatmadan once daha\n"
            "uzun bir BCE kosusu (epoch butcesi) olculmeli."
        )
    else:
        print(
            "RAW barajini geciyor: tarif geri kazanildi. Sweep BCE tabaniyla\n"
            "yeniden planlanabilir; kazananlarin bandi 0.32-0.44'tur."
        )


def main(argv=None):
    args = parse_args(argv)
    jobs = harness.select(build_probe_jobs(args.epochs), args.only)
    if args.list:
        print_plan(jobs)
        return 0

    print("=== KAYIP FONKSIYONU COKME TESHISI ===")
    print_plan(jobs)
    print(f"\nCikti koku: {PROBE_ROOT}")
    print("Benchmark CALISTIRILMAZ; olcum yalniz marj kapisidir.\n")

    summaries = []
    for position, job in enumerate(jobs, start=1):
        print(f"\n=== {position}/{len(jobs)} {job['name']} "
              f"(kayip={job['LOSS_FUNCTION']}) ===")
        model_dir = PROBE_ROOT / job["name"]
        if model_dir.exists():
            shutil.rmtree(model_dir)
        status = harness.train_one(job, model_dir, probe_root=PROBE_ROOT)
        if status != "ok":
            print(f"[{job['name']}] {status}")
            summaries.append(
                {
                    "name": job["name"],
                    "loss_function": job["LOSS_FUNCTION"],
                    "status": status,
                    "collapsed": None,
                }
            )
            continue
        report = harness.gate_one(
            job, model_dir, args.gate_samples, probe_root=PROBE_ROOT
        )
        summary = harness.summarize(job, report)
        summary["status"] = "ok"
        summaries.append(summary)
        state = "COKTU" if summary["collapsed"] else "saglam"
        print(f"[{job['name']}] {state} | saglam checkpoint={summary['healthy_checkpoints']}")

    print_summary(summaries)
    payload = {
        "created_at": datetime.datetime.now().isoformat(),
        "epochs": args.epochs,
        "gate_samples": args.gate_samples,
        "gate_aoi": pipeline.MARGIN_GATE_AOI,
        "winner_loss": WINNER_LOSS,
        "baseline_job": {key: pipeline.BASELINE_JOB[key] for key in pipeline.CONFIG_KEYS},
        "results": summaries,
    }
    out_path = PROBE_ROOT / "probe_summary.json"
    pipeline.write_json_atomic(out_path, payload)
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
