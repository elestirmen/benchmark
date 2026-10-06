#!/usr/bin/env python
"""Bake an input affine into the first conv layer of a Keras model.

The benchmark always feeds z = 2x - 1 (x in [0, 1]).  After baking, the model
computes f(a*x + c) for that input, i.e. it behaves as if it were fed a*x + c.
Exact except inside the zero-padded 'same' border (a few px, cropped anyway).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def bake(model, a, c):
    first = next(l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D))
    if isinstance(first, tf.keras.layers.Conv2DTranspose):
        raise ValueError("ilk katman Conv2DTranspose; desteklenmiyor")
    w, b = first.get_weights()
    sum_w = w.sum(axis=(0, 1, 2))
    first.set_weights([w * (a / 2.0), b + (a / 2.0 + c) * sum_w])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = tf.keras.models.load_model(args.model, compile=False)
    ref = tf.keras.models.load_model(args.model, compile=False)
    bake(model, args.a, args.c)
    x = np.random.default_rng(0).uniform(0, 1, size=(1, 544, 544, 1)).astype(np.float32)
    got = model.predict(2 * x - 1, verbose=0)
    want = ref.predict(args.a * x + args.c, verbose=0)
    inner = (slice(None), slice(12, -12), slice(12, -12))
    err = float(np.abs(got[inner] - want[inner]).max())
    print(f"ic bolge max fark={err:.2e}")
    if err > 1e-4:
        raise SystemExit("kalibrasyon dogrulamasi basarisiz")
    model.save(args.out)
    print("yazildi:", args.out)


if __name__ == "__main__":
    main()
