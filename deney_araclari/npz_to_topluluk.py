#!/usr/bin/env python
"""torch uye checkpoint'leri (.npz) -> tek Keras .h5 (uye sigmoid ciktilarinin ortalamasi).

Her uye torch_to_keras.convert ile donusturulur (kayitli PyTorch ciktisiyla dogrulanir),
sonra tek duz grafikte birlestirilir: ortak giris, her uyenin katman zinciri (onekli
adlarla kopya), Average. (Ic ice Model'li topluluk.py yapisi bu uyelerle kaydedilirken
Keras serilestirme hatasi veriyor.)

    python npz_to_topluluk.py epoch_00003_uye0.npz epoch_00003_uye1.npz epoch_00003_uye2.npz --out X.h5
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import h5py  # noqa: F401,E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from torch_to_keras import convert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        members = []
        for i, path in enumerate(args.npz):
            h5 = Path(tmp) / f"uye{i}.h5"
            err = convert(path, h5, norm="minus1_1")
            print(f"uye {i}: {Path(path).name} max_fark={err:.2e}")
            members.append(tf.keras.models.load_model(h5, compile=False))
        inp = tf.keras.Input(shape=members[0].input_shape[1:], name="giris")
        heads = []
        for i, m in enumerate(members):
            x = inp
            for j, layer in enumerate(m.layers[1:]):  # duz zincir: InputLayer haric
                cfg = layer.get_config()
                cfg["name"] = "model_output" if (len(members) == 1 and j == len(m.layers) - 2) else f"u{i}_{j}_{layer.name}"
                clone = layer.__class__.from_config(cfg)
                x = clone(x)
                clone.set_weights(layer.get_weights())
            heads.append(x)
        out = heads[0] if len(heads) == 1 else tf.keras.layers.Average(name="topluluk_ortalama")(heads)
        model = tf.keras.Model(inp, out, name="topluluk")
        x = np.random.default_rng(0).uniform(-1, 1, size=(1, 544, 544, 1)).astype(np.float32)
        want = np.mean([m.predict(x, verbose=0) for m in members], axis=0)
        print("topluluk max fark", float(np.abs(model.predict(x, verbose=0) - want).max()))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        back = tf.keras.models.load_model(args.out, compile=False)
        print("yeniden yukleme fark", float(np.abs(back.predict(x, verbose=0) - want).max()))
    print(f"KAYIT {args.out}")


if __name__ == "__main__":
    main()
