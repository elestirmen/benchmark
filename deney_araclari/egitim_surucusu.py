#!/usr/bin/env python
"""Run one autoencoder_unified.py training job with the pipeline's own code generator.

Reuses end_to_end_pipeline.build_training_code so the training code path is
identical to the pipeline's; only the job dictionary differs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BENCH = Path(r"C:\d_surucusu\visual_navigation\benchmark")
sys.path.insert(0, str(BENCH))
import end_to_end_pipeline as pipe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=str(BENCH / "deneyler" / "2026-09-24_zero1_tarif" / "modeller" / "egitim"))
    ap.add_argument("--model-type", default="flat_bottleneck")
    ap.add_argument("--filters", type=int, default=48)
    ap.add_argument("--kernel", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hidden", default="elu")
    ap.add_argument("--output", default="sigmoid")
    ap.add_argument("--norm", default="zero_1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--loss", default="binary_crossentropy")
    ap.add_argument("--data-dir", default=pipe.DATA_DIR_PATH)
    args = ap.parse_args()

    job = dict(pipe.BASELINE_JOB)
    job.update(
        name=args.name,
        MODEL_TYPE=args.model_type,
        FILTER_COUNT=args.filters,
        KERNEL_SIZE=(args.kernel, args.kernel),
        STRIDES=(1, 1),
        FLAT_DEPTH=args.depth,
        HIDDEN_ACTIVATION=args.hidden,
        OUTPUT_ACTIVATION=args.output,
        INPUT_NORMALIZATION=args.norm,
        BATCH_SIZE=args.batch,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        OPTIMIZER="adam",
        LOSS_FUNCTION=args.loss,
        PRETRAINED_MODEL=None,
        EARLY_STOPPING_PATIENCE=0,
        REDUCE_LR_PATIENCE=0,
    )
    out_dir = Path(args.root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    source = (Path(pipe.TRAINING_SCRIPT_DIR) / pipe.TRAINING_SCRIPT_NAME).read_text(encoding="utf-8")
    pipe.DATA_DIR_PATH = args.data_dir
    code = pipe.build_training_code(source, job, out_dir)
    temp = Path(pipe.TRAINING_SCRIPT_DIR) / f"temp_autoencoder_unified_{args.name}.py"
    temp.write_text(code, encoding="utf-8")
    (out_dir / "_job.json").write_text(
        json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in job.items()}, indent=2),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("CUDA_CACHE_PATH", str(BENCH / "outputs" / "cuda_cache"))
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
    log = out_dir / "_training.log"
    try:
        with log.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [sys.executable, "-u", temp.name],
                cwd=pipe.TRAINING_SCRIPT_DIR,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            code_ = proc.wait()
    finally:
        if temp.exists():
            temp.unlink()
    (out_dir / "_done.json").write_text(json.dumps({"return_code": code_}), encoding="utf-8")
    print(f"[{args.name}] bitti, cikis={code_}, log={log}")
    return code_


if __name__ == "__main__":
    sys.exit(main())
