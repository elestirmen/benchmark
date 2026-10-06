#!/usr/bin/env python
"""PyTorch re-implementation of autoencoder_unified.py's flat_bottleneck training.

Same data (sonuclar_19 pairs), same preprocessing (gray, 512->544 nearest,
empty-label filter on train), same architecture (f, f/2, f/4 conv + mirrored
stride-1 transposed conv + sigmoid head, TF 'same' padding), same loss (BCE on
logits, as Keras does for a sigmoid head) and Adam(1e-3, eps=1e-7).  Runs on
Blackwell natively (TF 2.10 only has PTX-JIT kernels there).

Checkpoints are .npz weight dumps; torch_to_keras.py turns them into .h5 files
that the unchanged benchmark loads like any pipeline model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = r"E:\temp\tez__ayna\python_calismalar\larger-google-sat2maps-dataset\islenmis\sonuclar_19"
IMAGE_SIZE = 544


def load_pair(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # Rec.601, same as the benchmark
    h, w = gray.shape
    half = w // 2
    src = cv2.resize(gray[:, :half], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    tar = cv2.resize(gray[:, half:], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return src, tar


def informative(tar):
    """autoencoder_unified.keep_informative_tile with the default thresholds."""
    white = float(np.mean(tar >= 245.0))
    dark = float(np.mean(tar <= 10.0))
    return white <= 0.99 or dark >= 0.005


def load_dataset(data_dir, seed, use_filter=True, max_files=0):
    names = sorted(
        n for n in os.listdir(data_dir)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )
    paths = [os.path.join(data_dir, n) for n in names]
    rng = np.random.RandomState(seed)
    rng.shuffle(paths)
    if max_files > 0:
        # Buyuk klasorlerden (output19_ps ~100k) once ornekle, sonra oku.
        paths = paths[:max_files]
    split = int(len(paths) * 0.9)
    with ThreadPoolExecutor(max_workers=16) as pool:
        pairs = list(pool.map(load_pair, paths))
    train, val = [], []
    for i, pair in enumerate(pairs):
        if pair is None:
            continue
        if i < split:
            if not use_filter or informative(pair[1]):
                train.append(pair)
        else:
            val.append(pair)
    stack = lambda items, j: np.stack([p[j] for p in items])  # noqa: E731
    return stack(train, 0), stack(train, 1), stack(val, 0), stack(val, 1)


class SameConv(nn.Module):
    """Stride-1 conv with TensorFlow 'same' padding (extra pad after for even k)."""

    def __init__(self, cin, cout, k, pad_before):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k)
        self.pad = (pad_before, k - 1 - pad_before, pad_before, k - 1 - pad_before)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


def keras_he_normal_(weight, fan_in):
    # keras VarianceScaling(2.0, 'fan_in', 'truncated_normal')
    std = math.sqrt(2.0 / fan_in) / 0.87962566103423978
    with torch.no_grad():
        nn.init.trunc_normal_(weight, 0.0, std, -2 * std, 2 * std)


def keras_glorot_uniform_(weight, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-limit, limit)


class FlatBottleneck(nn.Module):
    """create_flat_bottleneck_model: widths f, f/2, ... then mirrored decoder."""

    def __init__(self, filters=48, kernel=4, depth=3, hidden="elu", outer_kernel=None):
        super().__init__()
        inner = kernel
        outer = outer_kernel or kernel
        widths = [filters // (2 ** i) for i in range(depth)]
        n_enc = len(widths)
        # deneme_6 f32_k5 (2022): ilk conv, son deconv ve cikis dis cekirdekle.
        ks = [outer] + [inner] * (n_enc - 1) + [inner] * (n_enc - 1) + [outer, outer]
        self.act = nn.ELU() if hidden == "elu" else nn.ReLU()
        layers, kinds, cin = [], [], 1
        plan = [("conv", w) for w in widths] + [("deconv", w) for w in reversed(widths)] + [("head", 1)]
        for (kind, w), k in zip(plan, ks):
            enc_pad = (k - 1) // 2          # TF Conv2D 'same'
            dec_pad = k - 1 - enc_pad       # TF Conv2DTranspose(stride 1, 'same') == flipped conv
            layers.append(SameConv(cin, w, k, dec_pad if kind == "deconv" else enc_pad))
            kinds.append(kind)
            cin = w
        self.layers = nn.ModuleList(layers)
        self.kinds = kinds
        self.kernels = ks
        self.meta = {"filters": filters, "kernel": inner, "depth": depth, "hidden": hidden, "kernels": ks}
        for layer, kind in zip(self.layers, self.kinds):
            wgt = layer.conv.weight  # [out, in, k, k]
            out_c, in_c = wgt.shape[0], wgt.shape[1]
            k = wgt.shape[-1]
            if kind == "conv":
                keras_he_normal_(wgt, in_c * k * k)
            elif kind == "deconv":
                # Keras computes fans from the Conv2DTranspose kernel
                # (k, k, filters, in): its "fan_in" is filters * k * k.
                keras_he_normal_(wgt, out_c * k * k)
            else:
                keras_glorot_uniform_(wgt, in_c * k * k, out_c * k * k)
            nn.init.zeros_(layer.conv.bias)

    def forward(self, x):
        for layer, kind in zip(self.layers, self.kinds):
            x = layer(x)
            if kind != "head":
                x = self.act(x)
        return x  # logits


def normalize(u8, mode):
    x = u8.float() / 255.0
    return x * 2.0 - 1.0 if mode == "minus1_1" else x


def parse_lr_schedule(text):
    """"1e-3:4,3e-4:3" -> [(1e-3, 4), (3e-4, 3)] (lr, epoch sayisi)."""
    steps = []
    for part in text.split(","):
        lr, count = part.split(":")
        steps.append((float(lr), int(count)))
    return steps


def lr_for_epoch(schedule, epoch):
    seen = 0
    for lr, count in schedule:
        seen += count
        if epoch <= seen:
            return lr
    return schedule[-1][0]


def _gaussian_kernel(sigma, device):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(t * t) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _motion_kernel(length, angle_deg, device):
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(float(k.sum()), 1e-8)
    return torch.from_numpy(k).to(device)[None, None]


def _filter(img, kernel):
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="reflect"), kernel)


def photometric_augment(u8, prob, gen):
    """Fotometrik IHA kamera bozulmalari (geometri degismez, etiket degismez).

    Aileler benchmark'in hard_v1 profiline benzer (kazanc/ofset/gama, pus,
    hareket ve odak bulanikligi, dusuk kontrast, gurultu) ama parametreler
    kendi rastgele uretecinden gelir; JPEG yoktur. u8: [B,1,H,W] uint8 (GPU).
    """
    x = u8.float()
    for b in range(x.shape[0]):
        if float(torch.rand(1, generator=gen)) >= prob:
            continue
        r = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=gen))  # noqa: E731
        img = x[b : b + 1]
        img = (img * r(0.8, 1.2) + r(-15.0, 15.0)).clamp(0, 255)
        img = 255.0 * (img / 255.0).pow(r(0.8, 1.25))
        pick = float(torch.rand(1, generator=gen))
        if pick < 0.25:  # pus
            a = r(0.1, 0.35)
            img = img * (1 - a) + r(220.0, 255.0) * a
        elif pick < 0.45:  # hareket bulanikligi
            length = int(r(2, 9)) * 2 + 1  # 5..17 tek
            img = _filter(img, _motion_kernel(length, r(0.0, 180.0), img.device))
        elif pick < 0.62:  # odak bulanikligi
            img = _filter(img, _gaussian_kernel(r(0.8, 2.6), img.device))
        elif pick < 0.75:  # dusuk kontrast
            m = img.mean()
            img = m + (img - m) * r(0.5, 0.85)
        noise = r(0.5, 6.0)
        img = img + torch.randn(img.shape, generator=gen, device="cpu").to(img.device) * noise
        x[b : b + 1] = img.clamp(0, 255).round()
    return x.to(torch.uint8)


class WeightEMA:
    """Agirliklarin ustel hareketli ortalamasi (Polyak); kopya model tutar."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)


def load_npz_weights(model, path):
    data = np.load(path, allow_pickle=False)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.conv.weight.copy_(torch.from_numpy(data[f"w{i}"]))
            layer.conv.bias.copy_(torch.from_numpy(data[f"b{i}"]))


def save_checkpoint(model, path, probe):
    arrays = {}
    for i, (layer, kind) in enumerate(zip(model.layers, model.kinds)):
        arrays[f"w{i}"] = layer.conv.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.conv.bias.detach().cpu().numpy()
    # Reference output for the Keras export check: CPU fp32, no TF32/cuDNN noise.
    cpu_model = copy.deepcopy(model).to("cpu").float().eval()
    with torch.no_grad():
        probe_cpu = probe.detach().to("cpu").float().contiguous()
        arrays["probe_x"] = probe_cpu.numpy()
        arrays["probe_y"] = torch.sigmoid(cpu_model(probe_cpu)).numpy()
    del cpu_model
    np.savez(path, kinds=np.array(model.kinds), meta=json.dumps(model.meta), **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--filters", type=int, default=48)
    ap.add_argument("--kernel", type=int, default=4)
    ap.add_argument("--outer-kernel", type=int, default=0, help="ilk conv/son deconv/cikis cekirdegi (0=kernel ile ayni)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hidden", default="elu", choices=("elu", "relu"))
    ap.add_argument("--norm", default="zero_1", choices=("zero_1", "minus1_1"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save-epochs", default="1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,60,70,80,90,100,120,150,200")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (default fp32 like TF)")
    ap.add_argument("--max-train", type=int, default=0, help="egitim kumesini ilk N ornege kirp (0=hepsi)")
    ap.add_argument("--max-files", type=int, default=0, help="okumadan once klasorden rastgele N dosya sec (0=hepsi)")
    ap.add_argument("--no-filter", action="store_true", help="bos etiket filtresini kapat (2022 Kaggle scripti filtrelemiyordu)")
    ap.add_argument("--lr-schedule", default="", help='parca parca sabit lr: "1e-3:4,3e-4:3,1e-4:3" (epoch sayisini da belirler)')
    ap.add_argument("--ema", type=float, default=0.0, help="agirlik EMA bozunumu (adim basina, ornek 0.9998); ema_XXXXX.npz da kaydedilir")
    ap.add_argument("--aug", type=float, default=0.0, help="fotometrik bozulma olasiligi (ornek basina, 0=kapali)")
    ap.add_argument("--init-npz", default="", help="agirliklari bu checkpoint'ten baslat (dusuk lr ile ince ayar icin)")
    args = ap.parse_args()

    schedule = parse_lr_schedule(args.lr_schedule) if args.lr_schedule else [(args.lr, args.epochs)]
    args.epochs = sum(count for _, count in schedule)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    aug_gen = torch.Generator().manual_seed(args.seed + 1000)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    t0 = time.time()
    xs, ys, vxs, vys = load_dataset(args.data_dir, args.seed, use_filter=not args.no_filter, max_files=args.max_files)
    if args.max_train > 0:
        xs, ys = xs[: args.max_train], ys[: args.max_train]
    print(f"veri: egitim={len(xs)} dogrulama={len(vxs)} ({time.time()-t0:.0f}s)", flush=True)
    X = torch.from_numpy(xs).unsqueeze(1)  # uint8 on CPU, pinned batches below
    Y = torch.from_numpy(ys).unsqueeze(1)
    VX = torch.from_numpy(vxs).unsqueeze(1)
    VY = torch.from_numpy(vys).unsqueeze(1)

    model = FlatBottleneck(args.filters, args.kernel, args.depth, args.hidden, outer_kernel=args.outer_kernel or None).to(dev)
    if args.init_npz:
        load_npz_weights(model, args.init_npz)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=schedule[0][0], betas=(0.9, 0.999), eps=1e-7)
    ema = WeightEMA(model, args.ema) if args.ema > 0 else None
    probe = normalize(VX[:2].to(dev), args.norm)
    save_epochs = {int(v) for v in args.save_epochs.split(",") if v}
    (out / "_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log = (out / "_log.jsonl").open("a", encoding="utf-8")

    def val_loss(net):
        vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(VX), 16):
                xb = normalize(VX[s : s + 16].to(dev), args.norm).contiguous(memory_format=torch.channels_last)
                yb = VY[s : s + 16].to(dev).float() / 255.0
                vl = F.binary_cross_entropy_with_logits(net(xb).float(), yb)
                vtotal += float(vl) * len(xb)
                vcount += len(xb)
        return vtotal / vcount

    n = len(X)
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        lr = lr_for_epoch(schedule, epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        perm = torch.randperm(n)
        total, count = 0.0, 0
        for s in range(0, n, args.batch):
            idx = perm[s : s + args.batch]
            xu8 = X[idx].to(dev, non_blocking=True)
            if args.aug > 0:
                xu8 = photometric_augment(xu8, args.aug, aug_gen)
            xb = normalize(xu8, args.norm).contiguous(memory_format=torch.channels_last)
            yb = Y[idx].to(dev, non_blocking=True).float() / 255.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        model.eval()
        rec = {"epoch": epoch, "lr": lr, "loss": total / count, "val_loss": val_loss(model)}
        if ema is not None:
            rec["ema_val_loss"] = val_loss(ema.model)
        model.train()
        rec["seconds"] = time.time() - te
        log.write(json.dumps(rec) + "\n")
        log.flush()
        extra = f" ema_val={rec['ema_val_loss']:.4f}" if ema is not None else ""
        print(f"epoch {epoch} lr={lr:g} loss={rec['loss']:.4f} val_loss={rec['val_loss']:.4f}{extra} ({rec['seconds']:.0f}s)", flush=True)
        if epoch in save_epochs or epoch == args.epochs:
            save_checkpoint(model, out / f"epoch_{epoch:05d}.npz", probe)
            if ema is not None:
                save_checkpoint(ema.model, out / f"ema_{epoch:05d}.npz", probe)
    print("bitti", flush=True)


if __name__ == "__main__":
    main()
