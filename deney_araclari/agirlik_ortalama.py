#!/usr/bin/env python
"""Ayni kosunun birkac checkpoint'inin agirlik ortalamasi (SWA benzeri).

Topluluktan farki: sonuc TEK modeldir (cikarim maliyeti tek model kadar).
Ayni baslangictan, sabit lr ile ilerleyen kosunun ardisik epochlari ayni
havzada oldugu icin agirliklari ortalanabilir; farkli seed'ler ortalanamaz.

    python agirlik_ortalama.py --out swa_00005.npz epoch_00003.npz epoch_00004.npz ...

Cikti .npz torch_flat_egitim.py bicimindedir (probe_y ortalanmis agirliklarla
CPU'da yeniden hesaplanir), torch_to_keras.py ile .h5'e cevrilir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torch_flat_egitim import FlatBottleneck, load_npz_weights, save_checkpoint


def model_from_meta(meta):
    kernels = meta.get("kernels") or []
    outer = kernels[0] if kernels and kernels[0] != meta["kernel"] else None
    return FlatBottleneck(meta["filters"], meta["kernel"], meta["depth"], meta["hidden"], outer_kernel=outer)


def average(paths, out_path):
    datas = [np.load(p, allow_pickle=False) for p in paths]
    metas = [json.loads(str(d["meta"])) for d in datas]
    if any(m != metas[0] for m in metas[1:]):
        raise SystemExit("mimariler farkli; yalniz ayni kosunun checkpointleri ortalanabilir")
    model = model_from_meta(metas[0])
    n_layers = len(model.layers)
    arrays = {}
    for i in range(n_layers):
        arrays[f"w{i}"] = np.mean([d[f"w{i}"] for d in datas], axis=0)
        arrays[f"b{i}"] = np.mean([d[f"b{i}"] for d in datas], axis=0)
    tmp = Path(out_path).with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    load_npz_weights(model, tmp)
    tmp.unlink()
    probe = torch.from_numpy(datas[0]["probe_x"])
    save_checkpoint(model, out_path, probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    average(args.npz, args.out)
    print(f"{args.out}  <- {len(args.npz)} checkpoint")


if __name__ == "__main__":
    main()
