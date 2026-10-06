#!/usr/bin/env python
"""Yakalanan karelerde modelden bagimsiz (GPS tabanli) vekil olcut: kare modelinin x1 harita
kesitindeki NCC tepesi EXIF GPS konumuna esik metreden yakin mi?

Etiket kullanmaz: yakalanan tum tam-pencereli kareler. Kesit GPS merkezli; GPS'e tam oturan
sablonun kesitteki sol ustu (pay, pay). Hata = |tepe - (pay, pay)| * 0.2985 m/px (merkezden
merkeze). Arama kesitle sinirli (+-pay px), gercek TM 5000 px pencere ve 3 olcekle arar; bu
yuzden mutlak deger TM'den yuksektir, karsilastirma icindir.

    python ucus_uyarlama_olc.py --veri D/veri --rotalar 6_tezde_ucus_4 ... \
        --model ogretmen=D/modeller/ogretmen/x1 --model a1e2=D/modeller/egitim/a1_lr3e5/epoch_00002
(--model ad=onek: onek_uye0.npz, onek_uye1.npz, ... okunur)
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch

from ucus_uyarlama_egitim import Student, gps_hits, load_frames

KAYIK = {"guzergah_1_donus", "guzergah_2_tezde_ucus_1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", required=True)
    ap.add_argument("--rotalar", nargs="+", required=True)
    ap.add_argument("--model", action="append", required=True, help="ad=onek")
    ap.add_argument("--esik", type=float, default=25.0)
    ap.add_argument("--kayik-esik", type=float, default=45.0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    frames = {r: load_frames(args.veri, r) for r in args.rotalar}
    results = {}
    for spec in args.model:
        name, prefix = spec.split("=", 1)
        paths = sorted(glob.glob(prefix + "_uye*.npz"))
        model = Student(paths).to(dev).eval()
        per = {r: gps_hits(model, items, args.kayik_esik if r in KAYIK else args.esik, dev)
               for r, items in frames.items()}
        results[name] = per
        allok = [o for v in per.values() for o in v]
        print(f"{name:28s} GPS-vekil basari={np.mean(allok):.3f} (n={len(allok)})  " +
              "  ".join(f"{r}={np.mean(v):.3f}" for r, v in per.items()), flush=True)
    names = list(results)
    if len(names) > 1:
        base = names[0]
        b = np.concatenate([results[base][r] for r in args.rotalar])
        for n in names[1:]:
            o = np.concatenate([results[n][r] for r in args.rotalar])
            print(f"  {n} vs {base}: yalniz {n} {int((o & ~b).sum())} / yalniz {base} {int((b & ~o).sum())}")


if __name__ == "__main__":
    main()
