#!/usr/bin/env python
"""Sigmoid cikis katmaninin on-aktivasyonunu T ile olcekler (sicaklik; T>1 daha keskin).

    python sicaklik.py model.h5 --t 2 4 8 --out-dir D

Son Conv2D'nin (model_output) kernel ve bias'i T ile carpilir: sigmoid(T * z).
Mimari degismez, benchmark aynen yukler. Topluluk (ic ice) modellerde her uyenin
cikis katmanina uygulanir.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def output_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layers.extend(output_layers(layer))
        elif isinstance(layer, tf.keras.layers.Conv2D) and layer.get_config().get("activation") == "sigmoid":
            layers.append(layer)
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--t", type=float, nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in args.t:
        model = tf.keras.models.load_model(args.model, compile=False)
        heads = output_layers(model)
        if not heads:
            raise SystemExit("sigmoid cikis katmani bulunamadi")
        for head in heads:
            kernel, bias = head.get_weights()
            head.set_weights([kernel * t, bias * t])
        target = out_dir / f"{Path(args.model).stem}_T{t:g}.h5"
        model.save(target)
        print(f"{target.name}  cikis_katmani={len(heads)}")
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
