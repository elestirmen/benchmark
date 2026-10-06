#!/usr/bin/env python
"""Yansima/dondurme test-zamani ortalamasi (TTA) -> tek Keras .h5.

Tam evrisimli bir modelde  T^-1 f(T x) == f_T(x);  f_T, butun cekirdekleri
ayni uzamsal donusumle cevrilmis modeldir (tek cekirdek boyunda birebir; cift
cekirdekte TF 'same' dolgusu asimetrik oldugu icin 1 px kayar; kayma olculup ZeroPadding2D +
Cropping2D ile duzeltilir, bkz. build_transformed).  Bu
yuzden TTA yalniz standart katmanlarla kurulur (ic ice Model + Average) ve
benchmark degistirilmeden yukler.

    python yansima_toplulugu.py uye1.h5 [uye2.h5 ...] --transforms id,fliplr,flipud,rot180 --out tta.h5

Birden cok uye verilirse cikti, uyeler x donusumlerin ortalamasidir.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# Goruntu donusumu (H, W eksenleri) ve cekirdege uygulanacak karsiligi.
IMAGE_OPS = {
    "id": lambda a: a,
    "fliplr": lambda a: a[:, :, ::-1],
    "flipud": lambda a: a[:, ::-1, :],
    "rot180": lambda a: a[:, ::-1, ::-1],
    "rot90": lambda a: np.rot90(a, 1, axes=(1, 2)),
    "rot270": lambda a: np.rot90(a, 3, axes=(1, 2)),
    "transpose": lambda a: np.swapaxes(a, 1, 2),
    "antitranspose": lambda a: np.rot90(np.swapaxes(a, 1, 2), 2, axes=(1, 2)),
}
INVERSE = {"rot90": "rot270", "rot270": "rot90"}
KERNEL_CANDIDATES = {
    "id": [lambda k: k],
    "fliplr": [lambda k: k[:, ::-1]],
    "flipud": [lambda k: k[::-1, :]],
    "rot180": [lambda k: k[::-1, ::-1]],
    "rot90": [lambda k: np.rot90(k, 1, axes=(0, 1)), lambda k: np.rot90(k, 3, axes=(0, 1))],
    "rot270": [lambda k: np.rot90(k, 3, axes=(0, 1)), lambda k: np.rot90(k, 1, axes=(0, 1))],
    "transpose": [lambda k: np.swapaxes(k, 0, 1)],
    "antitranspose": [lambda k: np.rot90(np.swapaxes(k, 0, 1), 2, axes=(0, 1))],
}


def transformed_copy(model, kernel_op, name):
    clone = tf.keras.models.clone_model(model)
    clone._name = name

    for layer_src, layer_dst in zip(model.layers, clone.layers):
        w = layer_src.get_weights()
        if isinstance(layer_src, tf.keras.layers.Conv2D) and w:
            w = [np.ascontiguousarray(kernel_op(w[0]))] + w[1:]
        layer_dst.set_weights(w)

    return clone


def shift_array(a, dy, dx):
    """a[:, i, j] -> a[:, i - dy, j - dx] (disari tasan kenar sifir)."""
    out = np.zeros_like(a)
    h, w = a.shape[1], a.shape[2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, yd, xd] = a[:, ys, xs]
    return out


def shifted_model(clone, dy, dx, name):
    """Cikisi (dy, dx) piksel kaydiran sarmalayici; yalniz standart katmanlar."""
    if dy == 0 and dx == 0:
        return clone
    inp = tf.keras.Input(shape=clone.input_shape[1:])
    x = clone(inp)
    pad = ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0)))
    crop = ((max(-dy, 0), max(dy, 0)), (max(-dx, 0), max(dx, 0)))
    x = tf.keras.layers.ZeroPadding2D(pad)(x)
    x = tf.keras.layers.Cropping2D(crop)(x)
    return tf.keras.Model(inp, x, name=name)


M = 12  # dogrulama kenar payi (px)


def build_transformed(model, transform, name, probe):
    """Cekirdek donusumunu sayisal olarak dogrula; rot90 yonunu da boyle sec.

    Cift cekirdekte (k4) TF 'same' dolgusu asimetrik oldugu icin cevrilmis model
    gercek TTA'dan tam sayi piksel kayar; kayma (-2..2)^2 icinde aranir ve
    ZeroPadding2D + Cropping2D ile duzeltilir. Sifir dolgunun kenar etkisi 7 katmanda
    ~7 px'e yayildigi icin dogrulama M=12 px iceride yapilir; benchmark karo
    kenarindan zaten 16 px kirptigi icin kullanilan bolgede birebirdir.
    """
    want = IMAGE_OPS[INVERSE.get(transform, transform)](
        model.predict(IMAGE_OPS[transform](probe), verbose=0)
    )
    best = None
    for op in KERNEL_CANDIDATES[transform]:
        clone = transformed_copy(model, op, name + "_ic")
        got = clone.predict(probe, verbose=0)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                shifted = shift_array(got, dy, dx)
                inner = float(np.abs(shifted[:, M:-M, M:-M] - want[:, M:-M, M:-M]).max())
                if best is None or inner < best[2]:
                    best = (clone, (dy, dx), inner)
    clone, (dy, dx), inner = best
    wrapped = shifted_model(clone, dy, dx, name)
    err = float(np.abs(wrapped.predict(probe, verbose=0)[:, M:-M, M:-M] - want[:, M:-M, M:-M]).max())
    return wrapped, err, (dy, dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--transforms", default="id,fliplr,flipud,rot180")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    rng = np.random.default_rng(0)
    members = []
    for i, path in enumerate(args.models):
        base = tf.keras.models.load_model(path, compile=False)
        shape = base.input_shape[1:]
        probe = rng.uniform(-1, 1, size=(1,) + tuple(shape)).astype(np.float32)
        for t in transforms:
            clone, err, shift = build_transformed(base, t, f"uye{i}_{t}", probe)
            print(f"{os.path.basename(path)[:50]:50} {t:13} ic_fark={err:.2e} kayma={shift}")
            if err > 1e-3:
                raise SystemExit(f"TTA dogrulamasi basarisiz: {path} {t} fark={err:.2e}")
            members.append(clone)
    shape = members[0].input_shape[1:]
    inp = tf.keras.Input(shape=shape)
    outs = [m(inp) for m in members]
    out = outs[0] if len(outs) == 1 else tf.keras.layers.Average(name="tta_ortalama")(outs)
    tta = tf.keras.Model(inp, out, name="tta")
    tta.save(args.out)
    back = tf.keras.models.load_model(args.out, compile=False)
    x = rng.uniform(-1, 1, size=(1,) + tuple(shape)).astype(np.float32)
    print("yeniden yukleme fark", float(np.abs(back.predict(x, verbose=0) - tta.predict(x, verbose=0)).max()),
          "uye", len(members), "params", back.count_params())


if __name__ == "__main__":
    main()
