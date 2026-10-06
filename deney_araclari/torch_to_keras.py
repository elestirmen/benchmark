#!/usr/bin/env python
"""Convert torch_flat_egitim.py checkpoints (.npz) into Keras .h5 models.

The Keras graph is autoencoder_unified.create_flat_bottleneck_model's graph:
Conv2D x depth, Conv2DTranspose x depth (stride 1, 'same'), Conv2D head with
sigmoid named 'model_output'.  The transposed-conv kernel is the spatially
flipped, channel-swapped conv kernel.  Every export is checked against the
PyTorch output stored in the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input  # noqa: E402


def build(meta, image_size):
    f, k, depth, hidden = meta["filters"], meta["kernel"], meta["depth"], meta["hidden"]
    widths = [f // (2 ** i) for i in range(depth)]
    ks = list(meta.get("kernels") or [k] * (2 * depth + 1))
    inp = Input(shape=(image_size, image_size, 1))
    x = inp
    i = 0
    for w in widths:
        x = Conv2D(w, (ks[i], ks[i]), strides=(1, 1), activation=hidden, padding="same")(x); i += 1
    for w in reversed(widths):
        x = Conv2DTranspose(w, (ks[i], ks[i]), strides=(1, 1), activation=hidden, padding="same")(x); i += 1
    out = Conv2D(1, (ks[i], ks[i]), strides=(1, 1), activation="sigmoid", padding="same", name="model_output")(x)
    return tf.keras.Model(inp, out)


def convert(npz_path, out_path, image_size=544, norm="zero_1", tol=2e-4):
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    kinds = [str(v) for v in data["kinds"]]
    model = build(meta, image_size)
    convs = [l for l in model.layers if isinstance(l, Conv2D)]  # Conv2DTranspose is a Conv2D subclass
    if len(convs) != len(kinds):
        raise ValueError(f"katman sayisi uyusmuyor: keras={len(convs)} torch={len(kinds)}")
    for i, (layer, kind) in enumerate(zip(convs, kinds)):
        w = data[f"w{i}"]  # torch [out, in, kh, kw]
        b = data[f"b{i}"]
        if kind == "deconv":
            if not isinstance(layer, Conv2DTranspose):
                raise TypeError(f"katman {i} Conv2DTranspose olmali")
            kernel = w[:, :, ::-1, ::-1].transpose(2, 3, 0, 1)  # [kh, kw, filters(out), in]
        else:
            kernel = w.transpose(2, 3, 1, 0)  # [kh, kw, in, out]
        layer.set_weights([np.ascontiguousarray(kernel), b])
    # Verify against the stored PyTorch forward pass (NCHW -> NHWC).
    px = data["probe_x"].transpose(0, 2, 3, 1)
    want = data["probe_y"].transpose(0, 2, 3, 1)
    got = model.predict(px, verbose=0)
    err = float(np.abs(got - want).max())
    if err > tol:
        raise SystemExit(f"DOGRULAMA BASARISIZ {npz_path}: max fark={err:.2e}")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.save(out_path)
    sidecar = os.path.splitext(str(out_path))[0] + ".pre.json"
    Path(sidecar).write_text(json.dumps({
        "input_normalization": norm,
        "enhancement": "none",
        "color_mode": "grayscale",
        "image_size": image_size,
        "output_activation": "sigmoid",
        "output_color_mode": "grayscale",
        "trained_by": "torch_flat_egitim.py (PyTorch, autoencoder_unified flat_bottleneck esdegeri)",
    }, indent=2), encoding="utf-8")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--norm", default="zero_1")
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--skip-existing", action="store_true", help="hedef .h5 varsa donusturme")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.npz:
        name = keras_name(p, args.prefix)
        if args.skip_existing and (out_dir / name).is_file():
            continue
        err = convert(p, out_dir / name, norm=args.norm, tol=args.tol)
        print(f"{name}  max_fark={err:.2e}")


def keras_name(npz_path, prefix):
    """epoch_00005.npz -> <prefix>_model_..._epoch_00005_...; ema_/swa_ checkpoint'leri
    onekte ayrisir (<prefix>_ema_model_..., <prefix>_swa_model_...)."""
    stem = Path(npz_path).stem
    tag, _, epoch = stem.rpartition("_")
    meta = json.loads(str(np.load(npz_path)["meta"]))
    ktag = f"k{meta['kernel']}" if not meta.get("kernels") or len(set(meta["kernels"])) == 1 else f"k{meta['kernels'][0]}{meta['kernel']}"
    kind = "" if tag in ("", "epoch") else f"_{tag}"
    return (
        f"{prefix}{kind}_model_f{meta['filters']}_{ktag}_epoch_{epoch}"
        f"_h{meta['hidden']}_osigmoid_(1, 1)_.h5"
    )


if __name__ == "__main__":
    main()
