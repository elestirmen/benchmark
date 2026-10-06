#!/usr/bin/env python
"""Gercek IHA kareleriyle kare modeli uyarlamasi (ogretmen haritasi sabit).

Harita tarafi ogretmenin (x1) uydu haritasidir ve degismez; egitilen yalniz IHA karesini
isleyen modeldir. Hedef: TM'nin olcutu TM_CCOEFF_NORMED ile ogrenci ciktisinin x1
haritasinda dogru konumda tepe vermesi.

Veri: ucus_testi.py test --yakala ciktisi (veri/<rota>/kare_XXXX.npz): merkez olcek
model girdisi (544 px, uint8), x1 harita kesiti (1280 px, GPS merkezli), x1'in eslesme
konumu ve hatasi. Etiket (dogru konum): x1 bu karede dogruysa (hata < esik) x1'in eslesmesi;
degilse ayni karede dogru olan baska bir modelin (ucus test kayitlarindaki max_loc2)
eslesmesi. Arama penceresi tum modellerde ayni (GPS merkezli 5000 px), konumlar ortaktir.

Kayip: NCC haritasi (kesit x 512 px sablon, FFT ile) uzerinde softmax(NCC / tau);
dogru konumun yaricap-r komsulugunun olasiligi buyutulur, artı lambda * (1 - NCC_dogru).
Ogrenci, ogretmen agirliklariyla baslar (0. adim = x1).

    python ucus_uyarlama_egitim.py --veri D/veri --out-dir D/modeller/egitim/a1 \
        --egitim 1_tezde_ucus_5 ... --dogrulama 6_tezde_ucus_4 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_flat_egitim import FlatBottleneck, load_npz_weights, photometric_augment, save_checkpoint

BORDER = 16
LOG_LOC = re.compile(r"^-?[\d.e+]+ (-?[\d.e+-]+) \((-?\d+), (-?\d+)\) \((-?\d+), (-?\d+)\)\s*$")
LOG_ERR = re.compile(r"^\[(\d+)/\d+\] ERR=(-?\d+)m")


def log_matches(path):
    """ucus_testi test kaydi -> {kare: (x, y, err_m)}; kareyi bitiren ERR satirindan onceki son max_loc2."""
    out, loc = {}, None
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            m = LOG_LOC.match(line)
            if m:
                loc = (int(m.group(4)), int(m.group(5)))
                continue
            m = LOG_ERR.match(line)
            if m and loc is not None:
                out[int(m.group(1))] = (loc[0], loc[1], float(m.group(2)))
                loc = None
    return out


def load_route(veri, route, threshold, extra_logs):
    """Bir rotanin etiketli ornekleri: (giris u8 544x544, harita u8 kesit, (lx, ly), kaynak)."""
    others = {}
    for pattern in extra_logs:
        for path in sorted(glob.glob(pattern.format(rota=route))):
            for frame, rec in log_matches(path).items():
                others.setdefault(frame, []).append((Path(path).stem, rec))

    def one(path):
        d = np.load(path)
        if not bool(d["tam_cerceve"]):
            return None
        frame = int(d["kare"])
        off = d["kesit_ofset"]  # (satir, sutun) = (y, x)
        size = d["harita"].shape[0]
        tsize = d["sablon"].shape[0]
        cands = []
        if float(d["err_m"]) < threshold:
            cands.append(("x1", tuple(int(v) for v in d["max_loc2_xy"])))
        for name, (x, y, err) in others.get(frame, []):
            if err < threshold:
                cands.append((name, (x, y)))
        for name, (x, y) in cands:
            lx, ly = x - int(off[1]), y - int(off[0])
            if 0 <= lx <= size - tsize and 0 <= ly <= size - tsize:
                return d["giris"][1].copy(), d["harita"].copy(), (lx, ly), name, frame
        return None

    paths = sorted(glob.glob(os.path.join(veri, route, "kare_*.npz")))
    with ThreadPoolExecutor(max_workers=8) as pool:
        items = [r for r in pool.map(one, paths) if r is not None]
    return items, len(paths)


M_PER_PX = 0.2985


def load_frames(veri, route):
    """Etiketsiz tum tam-pencereli kareler: (kare, giris u8, harita kesiti u8, pay)."""
    out = []
    for path in sorted(glob.glob(os.path.join(veri, route, "kare_*.npz"))):
        d = np.load(path)
        if bool(d["tam_cerceve"]):
            pay = (d["harita"].shape[0] - d["sablon"].shape[0]) // 2
            out.append((int(d["kare"]), d["giris"][1].copy(), d["harita"].copy(), pay))
    return out


def gps_hits(model, frames, thr_m, dev):
    """GPS vekili (modelden bagimsiz): kesitteki NCC tepesi GPS'e thr_m'den yakin mi? Kesit GPS
    merkezli, GPS'e oturan sablonun sol ustu (pay, pay). Arama kesitle sinirli (TM: 5000 px)."""
    oks = []
    with torch.no_grad():
        for _frame, x8, h8, pay in frames:
            x = torch.from_numpy(x8)[None, None].to(dev).float() / 127.5 - 1.0
            tpl = torch.round(model(x)[0, 0, BORDER:-BORDER, BORDER:-BORDER] * 255.0)
            ncc = ncc_map(tpl, torch.from_numpy(h8).to(dev).float())
            py, px = divmod(int(torch.argmax(ncc)), ncc.shape[1])
            oks.append(math.hypot(px - pay, py - pay) * M_PER_PX < thr_m)
    return oks


def ncc_map(template, region):
    """TM_CCOEFF_NORMED: template [t,t], region [R,R] (float, ayni olcek) -> [R-t+1]^2."""
    t = template.shape[-1]
    size = region.shape[-1]
    n = t * t
    tz = template - template.mean()
    fr = torch.fft.rfft2(region)
    ft = torch.fft.rfft2(tz, s=(size, size))
    num = torch.fft.irfft2(fr * torch.conj(ft), s=(size, size))[: size - t + 1, : size - t + 1]
    r64 = region.double()
    ii = F.pad(r64.cumsum(0).cumsum(1), (1, 0, 1, 0))
    ii2 = F.pad((r64 * r64).cumsum(0).cumsum(1), (1, 0, 1, 0))
    box = lambda a: a[t:, t:] - a[:-t, t:] - a[t:, :-t] + a[:-t, :-t]  # noqa: E731
    var_r = (box(ii2) - box(ii) ** 2 / n).clamp_min(0).float()
    denom = torch.sqrt((tz * tz).sum() * var_r).clamp_min(1e-6)
    return num / denom


def peak_stats(ncc, lx, ly, radius):
    """(basari: tepe <= radius px, dogru NCC, disaridaki en buyuk NCC)."""
    h, w = ncc.shape
    flat = int(torch.argmax(ncc))
    py, px = divmod(flat, w)
    ok = max(abs(px - lx), abs(py - ly)) <= radius
    true = float(ncc[ly, lx])
    mask = torch.ones_like(ncc, dtype=torch.bool)
    mask[max(0, ly - 8): ly + 9, max(0, lx - 8): lx + 9] = False
    other = float(ncc[mask].max())
    return ok, true, other


class Student(torch.nn.Module):
    def __init__(self, npz_paths):
        super().__init__()
        members = []
        for p in npz_paths:
            meta = json.loads(str(np.load(p)["meta"]))
            ks = meta["kernels"]
            m = FlatBottleneck(meta["filters"], meta["kernel"], meta["depth"], meta["hidden"],
                               outer_kernel=ks[0] if ks[0] != meta["kernel"] else None)
            load_npz_weights(m, p)
            members.append(m)
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([torch.sigmoid(m(x)) for m in self.members]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ogretmen", nargs="+", required=True, help="baslangic uye .npz dosyalari")
    ap.add_argument("--egitim", nargs="+", required=True)
    ap.add_argument("--dogrulama", nargs="+", required=True)
    ap.add_argument("--esik", type=float, default=25.0, help="etiket icin hata esigi (m)")
    ap.add_argument("--kayik-rotalar", nargs="*", default=["guzergah_1_donus", "guzergah_2_tezde_ucus_1"],
                    help="sabit GPS kaymali rotalar (esik --kayik-esik)")
    ap.add_argument("--kayik-esik", type=float, default=45.0)
    ap.add_argument("--diger-kayitlar", nargs="*", default=[], help="glob kaliplari, {rota} yer tutucusu")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--radius", type=int, default=2, help="dogru konum komsulugu (px)")
    ap.add_argument("--ignore", type=int, default=10,
                    help="radius < d <= ignore halkasi negatiflerden cikar (NCC tepesinin dogal genisligi; "
                         "0 = halka yok, tepe sivriltmeye zorlanir)")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--aug", type=float, default=0.0, help="fotometrik bozulma olasiligi")
    ap.add_argument("--optim", default="adam", choices=("adam", "sgd"))
    ap.add_argument("--l2sp", type=float, default=0.0, help="ogretmen agirliklarina uzaklik cezasi (toplam kare fark x katsayi)")
    ap.add_argument("--anchor", type=float, default=0.0, help="ogrenci ciktisi ~ ogretmen ciktisi (ayni karede, MSE) katsayisi")
    ap.add_argument("--eval-every", type=int, default=0, help="her N adimda dogrulama + checkpoint (0=yalniz epoch sonu)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1000)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    def load(routes):
        items, stats = [], {}
        for r in routes:
            thr = args.kayik_esik if r in args.kayik_rotalar else args.esik
            got, total = load_route(args.veri, r, thr, args.diger_kayitlar)
            src = {}
            for it in got:
                src[it[3]] = src.get(it[3], 0) + 1
            stats[r] = {"etiketli": len(got), "yakalanan": total, "kaynak": src}
            items += [(r,) + it for it in got]
        return items, stats

    t0 = time.time()
    train, tstats = load(args.egitim)
    val, vstats = load(args.dogrulama)
    print(f"veri: egitim={len(train)} dogrulama={len(val)} ({time.time() - t0:.0f}s)", flush=True)
    for name, st in {**tstats, **vstats}.items():
        print(f"  {name}: {st}", flush=True)
    cfg = dict(vars(args), egitim_istatistik=tstats, dogrulama_istatistik=vstats)
    (out / "_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    model = Student(args.ogretmen).to(dev)
    teacher = Student(args.ogretmen).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    anchors = [p.detach().clone() for p in model.parameters()]
    if args.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    log = (out / "_log.jsonl").open("a", encoding="utf-8")
    probe = torch.from_numpy(np.stack([train[0][1], val[0][1]])[:, None]).to(dev).float() / 127.5 - 1.0

    def to_input(u8):
        return torch.from_numpy(u8)[None, None].to(dev)

    train_probe = [train[i] for i in torch.randperm(len(train), generator=torch.Generator().manual_seed(7))[:300].tolist()]
    # Secim olcutu: GPS vekili (etiketler cogunlukla x1'in kendi eslesmesi oldugundan etiketli
    # basari ogretmen lehine yanlidir: x1 o karelerde tanim geregi %100).
    gps_frames = {r: load_frames(args.veri, r) for r in args.dogrulama}

    def evaluate():
        rec = measure(val, "val")
        rec.update({k: v for k, v in measure(train_probe, "egt").items() if k != "egt_rota"})
        model.eval()
        per = {r: gps_hits(model, fr, args.kayik_esik if r in args.kayik_rotalar else args.esik, dev)
               for r, fr in gps_frames.items()}
        model.train()
        rec["val_gps"] = float(np.mean([o for v in per.values() for o in v]))
        rec["val_gps_rota"] = {r: float(np.mean(v)) for r, v in per.items()}
        return rec

    def measure(items, prefix):
        model.eval()
        oks, ratios, trues, per = [], [], [], {}
        with torch.no_grad():
            for route, x8, h8, (lx, ly), _src, frame in items:
                x = to_input(x8).float() / 127.5 - 1.0
                tpl = model(x)[0, 0, BORDER:-BORDER, BORDER:-BORDER]
                tpl = torch.round(tpl * 255.0)  # TM uint8'e cevirir
                ncc = ncc_map(tpl, torch.from_numpy(h8).to(dev).float())
                ok, true, other = peak_stats(ncc, lx, ly, args.radius + 1)
                oks.append(ok)
                ratios.append(other / max(true, 1e-6))
                trues.append(true)
                per.setdefault(route, []).append(ok)
        model.train()
        return {
            f"{prefix}_basari": float(np.mean(oks)), f"{prefix}_oran": float(np.median(ratios)),
            f"{prefix}_ncc": float(np.mean(trues)),
            f"{prefix}_rota": {r: float(np.mean(v)) for r, v in per.items()},
        }

    def line(rec):
        return (f"VAL_GPS={rec['val_gps']:.3f} | val_basari={rec['val_basari']:.3f} oran={rec['val_oran']:.3f} ncc={rec['val_ncc']:.3f} | "
                f"egt_basari={rec['egt_basari']:.3f} oran={rec['egt_oran']:.3f}")

    rec = {"epoch": 0, **evaluate()}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"epoch 0 (ogretmen) {line(rec)}", flush=True)

    def checkpoint(tag):
        for k, member in enumerate(model.members):
            save_checkpoint(member, out / f"{tag}_uye{k}.npz", probe)

    step = 0
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        perm = torch.randperm(len(train), generator=gen).tolist()
        total, count, hits = 0.0, 0, 0
        for i in perm:
            _route, x8, h8, (lx, ly), _src, _frame = train[i]
            xu8 = to_input(x8)
            if args.aug > 0:
                xu8 = photometric_augment(xu8, args.aug, gen)
            x = xu8.float() / 127.5 - 1.0
            full = model(x)
            tpl = full[0, 0, BORDER:-BORDER, BORDER:-BORDER] * 255.0
            region = torch.from_numpy(h8).to(dev).float()
            ncc = ncc_map(tpl, region)
            logits = (ncc / args.tau).flatten()
            w = ncc.shape[1]
            r = args.radius
            ys = torch.arange(max(0, ly - r), min(ncc.shape[0], ly + r + 1), device=dev)
            xs = torch.arange(max(0, lx - r), min(w, lx + r + 1), device=dev)
            pos = (ys[:, None] * w + xs[None, :]).flatten()
            if args.ignore > r:
                keep = torch.ones_like(logits, dtype=torch.bool).view(ncc.shape)
                g = args.ignore
                keep[max(0, ly - g): ly + g + 1, max(0, lx - g): lx + g + 1] = False
                keep = keep.flatten()
                keep[pos] = True
                loss = torch.logsumexp(logits[keep], 0) - torch.logsumexp(logits[pos], 0)
            else:
                loss = torch.logsumexp(logits, 0) - torch.logsumexp(logits[pos], 0)
            if args.pos_weight > 0:
                loss = loss + args.pos_weight * (1.0 - ncc[ly, lx])
            if args.anchor > 0:
                with torch.no_grad():
                    ref = teacher(x)
                loss = loss + args.anchor * F.mse_loss(full, ref) * 100.0
            if args.l2sp > 0:
                loss = loss + args.l2sp * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchors))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            total += float(loss.detach())
            count += 1
            with torch.no_grad():
                py, px = divmod(int(torch.argmax(ncc)), w)
                hits += max(abs(px - lx), abs(py - ly)) <= r + 1
            if args.eval_every and step % args.eval_every == 0 and count < len(perm):
                rec = {"epoch": epoch, "step": step, **evaluate()}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  adim {step} {line(rec)}", flush=True)
                checkpoint(f"step_{step:06d}")
        rec = {"epoch": epoch, "step": step, "loss": total / count, "egitim_basari": hits / count, **evaluate(),
               "seconds": time.time() - te}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"epoch {epoch} loss={rec['loss']:.3f} {line(rec)} ({rec['seconds']:.0f}s)", flush=True)
        checkpoint(f"step_{step:06d}")
    print("bitti", flush=True)


if __name__ == "__main__":
    main()
