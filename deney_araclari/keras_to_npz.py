#!/usr/bin/env python
"""Keras flat_bottleneck (.h5; tekil ya da topluluk.py toplulugu) -> torch_flat_egitim .npz.

torch_to_keras.py'nin tersi. Topluluk icin her uye ayri .npz olur (<out>_uye<i>.npz).
Her .npz, Keras'in ayni girdideki ciktisini (probe_x/probe_y, NCHW) tasir; PyTorch
tarafi yukleyince bu ciktiyla karsilastirir.

    python keras_to_npz.py topluluk_x1_2022.h5 --out D/x1     # D/x1_uye0.npz, ...
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.layers import Conv2D, Conv2DTranspose  # noqa: E402


def members(model):
    subs = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    return subs or [model]


def export_member(member, path, probe):
    convs = [l for l in member.layers if isinstance(l, Conv2D)]  # Conv2DTranspose da Conv2D alt sinifi
    kinds, arrays, ks = [], {}, []
    for i, layer in enumerate(convs):
        kernel, bias = layer.get_weights()
        if layer.strides != (1, 1) or layer.padding != "same":
            raise ValueError(f"{layer.name}: yalniz stride 1 'same' destekli")
        if isinstance(layer, Conv2DTranspose):
            kinds.append("deconv")
            w = kernel.transpose(2, 3, 0, 1)[:, :, ::-1, ::-1]  # [kh,kw,out,in] -> [out,in,kh,kw], uzamsal cevirme
        else:
            kinds.append("head" if i == len(convs) - 1 else "conv")
            w = kernel.transpose(3, 2, 0, 1)  # [kh,kw,in,out] -> [out,in,kh,kw]
        arrays[f"w{i}"] = np.ascontiguousarray(w, dtype=np.float32)
        arrays[f"b{i}"] = bias.astype(np.float32)
        ks.append(int(kernel.shape[0]))
    if kinds != ["conv"] * 3 + ["deconv"] * 3 + ["head"]:
        raise ValueError(f"beklenmeyen katman dizisi: {kinds}")
    acts = {convs[i].activation.__name__ for i in range(6)}
    if len(acts) != 1 or convs[-1].activation.__name__ != "sigmoid":
        raise ValueError(f"aktivasyonlar: {acts}, cikis {convs[-1].activation.__name__}")
    meta = {"filters": int(convs[0].filters), "kernel": ks[1], "depth": 3, "hidden": acts.pop(), "kernels": ks}
    sub = tf.keras.Model(member.inputs, member.outputs)
    probe_y = sub.predict(probe, verbose=0)
    np.savez(path, kinds=np.array(kinds), meta=json.dumps(meta), probe_x=probe.transpose(0, 3, 1, 2),
             probe_y=probe_y.transpose(0, 3, 1, 2), **arrays)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--out", required=True, help="cikti oneki (klasor/ad)")
    args = ap.parse_args()
    model = tf.keras.models.load_model(args.h5, compile=False)
    probe = np.random.default_rng(0).uniform(-1, 1, size=(2, 544, 544, 1)).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    subs = members(model)
    for i, m in enumerate(subs):
        path = f"{args.out}_uye{i}.npz" if len(subs) > 1 else f"{args.out}.npz"
        meta = export_member(m, path, probe)
        print(f"{path}: {meta}")
    if len(subs) > 1:
        np.savez(f"{args.out}_topluluk_probe.npz", probe_x=probe.transpose(0, 3, 1, 2),
                 probe_y=model.predict(probe, verbose=0).transpose(0, 3, 1, 2), uye_sayisi=len(subs))


if __name__ == "__main__":
    main()
