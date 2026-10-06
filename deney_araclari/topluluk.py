#!/usr/bin/env python
"""Average the sigmoid outputs of several flat models into one Keras .h5.

Built-in layers only (nested Models + Average), so the unchanged benchmark
loads it with load_model(compile=False); its output range stays [0, 1].
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    members = []
    for i, path in enumerate(args.models):
        m = tf.keras.models.load_model(path, compile=False)
        m._name = f"uye_{i}"
        # Ayni seriden iki uye (or. iki TPU checkpoint'i) ayni ic katman adlarini tasir;
        # h5 kaydinda "name already exists" ile carpisir. Her uyenin katmanlarina onek.
        for layer in m.layers:
            layer._name = f"u{i}_{layer.name}"
        members.append(m)
    shape = members[0].input_shape[1:]
    inp = tf.keras.Input(shape=shape)
    out = tf.keras.layers.Average(name="topluluk_ortalama")([m(inp) for m in members])
    ens = tf.keras.Model(inp, out, name="topluluk")
    x = np.random.default_rng(0).uniform(-1, 1, size=(1,) + tuple(shape)).astype(np.float32)
    want = np.mean([m.predict(x, verbose=0) for m in members], axis=0)
    got = ens.predict(x, verbose=0)
    print("max fark", float(np.abs(got - want).max()))
    ens.save(args.out)
    back = tf.keras.models.load_model(args.out, compile=False)
    print("yeniden yukleme fark", float(np.abs(back.predict(x, verbose=0) - got).max()), "params", back.count_params())


if __name__ == "__main__":
    main()
