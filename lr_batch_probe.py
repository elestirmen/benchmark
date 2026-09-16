#!/usr/bin/env python
"""lr/batch teshisi: 249 saatlik sweepten once cokmenin sebebini bul.

NEDEN BU SCRIPT VAR
===================
Kosu 20260830_021548'de egitilen her model RAW baseline'dan (success_25m 0.581)
kotu cikti ve buyuk olanlar tamamen coktu:

    flat_f48_k4_sigmoid_repro   10/100 epoch   gercek NCC 0.0   success 0.000
    flat_f64_k4                 11/100 epoch   gercek NCC 0.0   success 0.000
    flat_f32_k4                 62/100 epoch   gercek NCC 0.084 success 0.151
    flat_f24_k4                 77/100 epoch   gercek NCC 0.092 success 0.198
    RAW                              -         gercek NCC 0.306 success 0.581

Gercek NCC 0.0, modelin SABIT goruntu urettigi anlamina gelir; margin_gate.ncc()
taraflardan biri sabit oldugunda 0 dondurur. Yani sorun "model zayif ogrendi"
degil, "model hic ogrenmedi ve dejenere bir cozume dustu".

Erken durdurma bunu agirlastirdi (cokme val_loss'u platoya sokar, patience=8
tetiklenir, restore_best_weights olu agirliga geri doner) ve o kusur duzeltildi.
Ama tek sebep o degil: flat_f24_k4 77 epoch kostu ve yine 0.198'de kaldi.

Geriye kalan en olasi sebep TRAINING_JOBS'ta taranmayan bir eksen: 544x544 tam
cozunurlukte lr=1e-3 ve BATCH_SIZE=2. 2022'de 0.703 alan modelin adi `tpu_...`,
yani cok daha buyuk bir batch ile egitilmisti; "mimari birebir yeniden uretildi"
dogru ama optimizasyon rejimi hic yeniden uretilmedi.

NE YAPAR
========
Repro konfigurasyonunu sabit tutar, YALNIZ lr ve batch degistirir, her
kombinasyonu birkac epoch egitir ve HER epoch checkpointini marj kapisindan
gecirir. Boylece cokmenin olup olmadigi ve kacinci epochta basladigi CPU'da
dakikalar icinde gorulur; benchmark (checkpoint basina ~50 dk GPU) hic
calistirilmaz.

TRAINING_JOBS'a dokunmaz. Sweepin kontrol degiskenlerini degistirmez; amaci
sweep icin saglam bir taban secmektir.

KULLANIM
========
    python lr_batch_probe.py --list          # plani ve tahmini sureyi goster
    python lr_batch_probe.py                 # tum kombinasyonlari calistir
    python lr_batch_probe.py --only lr1e-4_b2
    python lr_batch_probe.py --epochs 4      # daha ucuz tarama
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import end_to_end_pipeline as pipeline


PROBE_ROOT = pipeline.MODELS_PIPELINE_DIR / "_lr_batch_probe"
# Cokme ilk epochlarda basliyor (repro 10 epochta olmustu ve erken durdurma
# 8 epoch iyilesme gormedigi icin tetiklenmisti), bu yuzden kisa bir butce
# ayrimi yapmaya yeter.
DEFAULT_PROBE_EPOCHS = 6
# Kapi burada siralama degil, "cokme var mi" sorusunu cevaplar; margin_gate.py
# 20 orneğin siniflari ayirmaya yettigini dogrulamis durumda.
PROBE_GATE_SAMPLES = 20


def probe(name, **overrides):
    job = dict(pipeline.BASELINE_JOB)
    job["name"] = name
    job.update(overrides)
    return job


def build_probe_jobs(epochs):
    """Repro konfigurasyonundan yalniz lr ve batch ekseninde ayril."""
    jobs = [
        # Kontrol: mevcut sweep ayari. Coktugu biliniyor; ayni butce ve ayni
        # olcumle tekrar edilmesi diger satirlarin yorumlanmasini saglar.
        probe("lr1e-3_b2", LEARNING_RATE=0.001, BATCH_SIZE=2),
        # lr'yi dusurmek, batch 2'nin gradyan gurultusunu telafi eden en ucuz
        # mudahaledir.
        probe("lr1e-4_b2", LEARNING_RATE=0.0001, BATCH_SIZE=2),
        probe("lr1e-5_b2", LEARNING_RATE=0.00001, BATCH_SIZE=2),
        # Batch'i buyutmek 2022 rejimine yaklasir; lineer olcekleme kuralina
        # gore lr'nin de birlikte artmasi beklenir, bu yuzden iki varyant.
        probe("lr1e-3_b8", LEARNING_RATE=0.001, BATCH_SIZE=8),
        probe("lr1e-4_b8", LEARNING_RATE=0.0001, BATCH_SIZE=8),
    ]
    for job in jobs:
        job["EPOCHS"] = epochs
    return jobs


def estimate_probe_minutes(job):
    """Batch buyudukce epoch basina sure duser; olcum f48/k4/batch2'den."""
    per_epoch = pipeline.estimate_minutes_per_epoch(job)
    batch_speedup = math.sqrt(int(job["BATCH_SIZE"]) / 2.0)
    train = per_epoch * int(job["EPOCHS"]) / batch_speedup
    gate = (int(job["EPOCHS"]) + 1) * 0.35  # 20 ornekle kapi ~20 sn/checkpoint
    return train + gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="lr/batch cokme teshisi; benchmark calistirmaz."
    )
    parser.add_argument("--list", action="store_true", help="Plani yazip cik.")
    parser.add_argument(
        "--only", action="append", default=[], help="Yalniz bu kombinasyon."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_PROBE_EPOCHS)
    parser.add_argument(
        "--gate-samples", type=int, default=PROBE_GATE_SAMPLES
    )
    return parser.parse_args(argv)


def select(jobs, names):
    if not names:
        return jobs
    by_name = {job["name"]: job for job in jobs}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"Bilinmeyen kombinasyon: {unknown}. Gecerli: {sorted(by_name)}")
    return [by_name[name] for name in names]


def print_plan(jobs):
    total = sum(estimate_probe_minutes(job) for job in jobs)
    print(f"Kombinasyon: {len(jobs)} | tahmini toplam: {total / 60.0:.1f} saat")
    print(f"{'ad':14} {'lr':>9} {'batch':>6} {'epoch':>6} {'~dk':>6}")
    for job in jobs:
        print(
            f"{job['name']:14} {job['LEARNING_RATE']:>9} {job['BATCH_SIZE']:>6} "
            f"{job['EPOCHS']:>6} {estimate_probe_minutes(job):>6.0f}"
        )


def train_one(job, model_dir, probe_root=PROBE_ROOT):
    """Egitimi calistir; OOM'da batch DUSURULMEZ, kombinasyon elenir.

    Batch burada kontrol degiskenidir; OOM'da 1'e dusurmek olcumu bozar.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(pipeline.TRAINING_SCRIPT_DIR) / pipeline.TRAINING_SCRIPT_NAME
    source = script_path.read_text(encoding="utf-8")
    code = pipeline.build_training_code(source, job, model_dir)
    temp_name = f"temp_probe_{job['name']}.py"
    temp_path = Path(pipeline.TRAINING_SCRIPT_DIR) / temp_name
    temp_path.write_text(code, encoding="utf-8")
    log_path = probe_root / "_logs" / f"{job['name']}.log"
    try:
        return_code, oom = pipeline.run_training_process(job, temp_name, log_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    if oom:
        return "oom"
    if return_code != 0:
        return f"egitim_hatasi_{return_code}"
    pipeline.partition_epoch_checkpoints(model_dir)
    return "ok"


def gate_one(job, model_dir, gate_samples, probe_root=PROBE_ROOT):
    """Her epoch checkpointini ve son modeli olc."""
    gate_path = probe_root / "_gate" / f"{job['name']}.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = list(pipeline.model_files_in(model_dir)) + list(
        pipeline.model_files_in(pipeline.epoch_checkpoint_dir(model_dir))
    )
    if not candidates:
        return None
    command = [
        pipeline.PYTHON_EXEC,
        pipeline.MARGIN_GATE_SCRIPT,
        "--samples",
        str(gate_samples),
        "--aoi",
        pipeline.MARGIN_GATE_AOI,
        "--normalization",
        str(job.get("INPUT_NORMALIZATION", "minus1_1")),
        "--json-out",
        str(gate_path),
    ]
    for path in candidates:
        command.extend(["--model", str(path)])
    try:
        subprocess.run(command, cwd=pipeline.BENCHMARK_CWD, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[KAPI] {job['name']} olculemedi: {exc}")
        return None
    return pipeline.read_json(gate_path, {})


def summarize(job, report):
    """Kombinasyon icin: cokme var mi, en iyi oran, hangi epochta."""
    results = (report or {}).get("results") or {}
    raw_ratio = (report or {}).get("raw_false_peak_ratio")
    rows = []
    for name, entry in results.items():
        if not isinstance(entry, dict) or name == "RAW" or "error" in entry:
            continue
        truth = entry.get("truth_ncc_mean")
        ratio = entry.get("false_peak_ratio")
        rows.append(
            {
                "checkpoint": name,
                "epoch": pipeline.checkpoint_epoch_number(Path(name)),
                "truth_ncc_mean": None if truth is None else float(truth),
                "false_peak_ratio": (
                    None
                    if ratio is None or not math.isfinite(float(ratio))
                    else float(ratio)
                ),
            }
        )
    healthy = [row for row in rows if (row["truth_ncc_mean"] or 0.0) >= 0.01]
    best = min(
        (row for row in healthy if row["false_peak_ratio"] is not None),
        key=lambda row: row["false_peak_ratio"],
        default=None,
    )
    return {
        "name": job["name"],
        "learning_rate": job["LEARNING_RATE"],
        "batch_size": job["BATCH_SIZE"],
        "loss_function": job["LOSS_FUNCTION"],
        "epochs": job["EPOCHS"],
        "raw_false_peak_ratio": None if raw_ratio is None else float(raw_ratio),
        "checkpoints": sorted(rows, key=lambda row: (row["epoch"] is None, row["epoch"])),
        "collapsed": not healthy,
        "healthy_checkpoints": len(healthy),
        "best": best,
    }


def print_summary(summaries):
    print()
    print("=" * 82)
    print("TESHIS SONUCU (oran: kucuk = ayirt edici; gercek NCC ~ 0 = model cokmus)")
    print("=" * 82)
    raw = next(
        (s["raw_false_peak_ratio"] for s in summaries if s["raw_false_peak_ratio"]), None
    )
    if raw is not None:
        print(f"RAW baraji: oran {raw:.2f}\n")
    print(f"{'kombinasyon':14} {'lr':>9} {'b':>3} {'saglam':>7} {'en iyi oran':>12} {'epoch':>6}")
    print("-" * 82)
    for summary in summaries:
        if summary.get("status") and summary["status"] != "ok":
            print(f"{summary['name']:14} {'':>9} {'':>3} {summary['status']}")
            continue
        best = summary["best"]
        ratio = f"{best['false_peak_ratio']:.2f}" if best else "-"
        epoch = str(best["epoch"]) if best and best["epoch"] is not None else "-"
        healthy = f"{summary['healthy_checkpoints']}/{len(summary['checkpoints'])}"
        flag = "  <- COKTU" if summary["collapsed"] else ""
        print(
            f"{summary['name']:14} {summary['learning_rate']:>9} "
            f"{summary['batch_size']:>3} {healthy:>7} {ratio:>12} {epoch:>6}{flag}"
        )
    print()
    survivors = [s for s in summaries if not s.get("collapsed") and s.get("best")]
    if not survivors:
        print(
            "Hicbir kombinasyon cokmeden cikamadi. Sebep lr/batch degil; siradaki\n"
            "adaylar egitim verisi (DATA_DIR), etiket araligi ve kayip fonksiyonudur."
        )
        return
    best = min(survivors, key=lambda s: s["best"]["false_peak_ratio"])
    print(
        f"En saglam taban: {best['name']} "
        f"(lr={best['learning_rate']}, batch={best['batch_size']}, "
        f"oran={best['best']['false_peak_ratio']:.2f})"
    )
    if raw is not None and best["best"]["false_peak_ratio"] >= raw:
        print(
            "UYARI: en iyi kombinasyon bile RAW barajini gecmiyor. Sweepi bu\n"
            "tabanla baslatmadan once egitim verisi/tarifi incelenmeli."
        )


def main(argv=None):
    args = parse_args(argv)
    jobs = select(build_probe_jobs(args.epochs), args.only)
    if args.list:
        print_plan(jobs)
        return 0

    print("=== LR / BATCH COKME TESHISI ===")
    print_plan(jobs)
    print(f"\nCikti koku: {PROBE_ROOT}")
    print("Benchmark CALISTIRILMAZ; olcum yalniz marj kapisidir.\n")

    summaries = []
    for position, job in enumerate(jobs, start=1):
        print(f"\n=== {position}/{len(jobs)} {job['name']} "
              f"(lr={job['LEARNING_RATE']}, batch={job['BATCH_SIZE']}) ===")
        model_dir = PROBE_ROOT / job["name"]
        if model_dir.exists():
            shutil.rmtree(model_dir)
        status = train_one(job, model_dir)
        if status != "ok":
            print(f"[{job['name']}] {status}")
            summaries.append(
                {
                    "name": job["name"],
                    "learning_rate": job["LEARNING_RATE"],
                    "batch_size": job["BATCH_SIZE"],
                    "status": status,
                    "collapsed": None,
                }
            )
            continue
        report = gate_one(job, model_dir, args.gate_samples)
        summary = summarize(job, report)
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
        "baseline_job": {
            key: pipeline.BASELINE_JOB[key] for key in pipeline.CONFIG_KEYS
        },
        "results": summaries,
    }
    out_path = PROBE_ROOT / "probe_summary.json"
    pipeline.write_json_atomic(out_path, payload)
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
