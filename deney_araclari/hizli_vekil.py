#!/usr/bin/env python
"""Fast proxy of the Urgup benchmark on small sub-regions.

Uses the benchmark's own search (coarse_to_fine_search), hard_v1 profile and
tile quality filters, but searches inside 6x6 km crops instead of the full
~500 km2 raster so one model takes minutes instead of ~50 min.

Model spec:  path            -> benchmark normalization (minus1_1)
             path::zero_1    -> feed [0, 1]
             path::affine=a,c-> feed a*x + c with x in [0, 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = Path(r"C:\d_surucusu\visual_navigation\benchmark")
os.environ.setdefault("CUDA_CACHE_PATH", str(BENCH / "outputs" / "cuda_cache"))
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(BENCH))

import h5py  # noqa: F401,E402  (must precede TF in this env)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import rowcol  # noqa: E402
from rasterio.windows import Window  # noqa: E402

import geospatial_model_benchmark as gmb  # noqa: E402

GOOGLE = BENCH / "urgup_cevresi_harita" / "urgup_cevresi_gmap_utm_30cm.tif"
BING = BENCH / "urgup_cevresi_harita" / "urgup_cevresi_bingmap_utm_30cm.tif"
TILE = 544
CROP = 16
OVERLAP = 32
FACTORS = (16, 8, 4, 2, 1)

# 6x6 km crops.  train: north of the margin-gate split (4263375), test: south.
REGIONS = {
    "train_nw": (662000.0, 4270000.0, 668000.0, 4276000.0),
    "train_ne": (672000.0, 4264000.0, 678000.0, 4270000.0),
    "test_s": (664000.0, 4256000.0, 670000.0, 4262000.0),
}
REGION_SETS = {
    "train": ("train_nw", "train_ne"),
    "nw": ("train_nw",),
    "test": ("test_s",),
    "all": ("train_nw", "train_ne", "test_s"),
}


def read_rgba(ds, col0, row0, w, h):
    arr = ds.read([1, 2, 3, 4], window=Window(col0, row0, w, h), boundless=True, fill_value=0)
    return np.moveaxis(arr[:3], 0, -1), arr[3]


def sample_queries(region, count, seed):
    e0, n0, e1, n1 = region
    rng = np.random.default_rng(seed)
    out = []
    margin = 300.0  # keep the 544 px tile (~162 m) well inside the crop
    with rasterio.open(GOOGLE) as g, rasterio.open(BING) as b:
        attempts = 0
        while len(out) < count and attempts < count * 60:
            attempts += 1
            e = float(rng.uniform(e0 + margin, e1 - margin))
            n = float(rng.uniform(n0 + margin, n1 - margin))
            ok = True
            tiles = {}
            for name, ds in (("google", g), ("bing", b)):
                r, c = rowcol(ds.transform, e, n)
                col0, row0 = int(c - TILE // 2), int(r - TILE // 2)
                rgb, alpha = read_rgba(ds, col0, row0, TILE, TILE)
                if int(alpha.min()) < 255:
                    ok = False
                    break
                std, ent, dark = gmb.tile_quality(gmb.to_gray(rgb))
                if std < 12 or ent < 4 or dark > 0.20:
                    ok = False
                    break
                # truth = geometric centre of the extracted even window
                ce, cn = gmb.pixel_center_to_geo(ds.transform, col0 + TILE / 2.0, row0 + TILE / 2.0)
                tiles[name] = (rgb, ce, cn)
            if ok:
                out.append(tiles)
    return out


class ModelRunner:
    def __init__(self, spec):
        import tensorflow as tf

        self.tf = tf
        path, _, opt = spec.partition("::")
        self.path = path
        self.spec = spec
        self.a, self.c = 2.0, -1.0
        if opt == "zero_1":
            self.a, self.c = 1.0, 0.0
        elif opt.startswith("affine="):
            self.a, self.c = (float(v) for v in opt.split("=", 1)[1].split(","))
        elif opt:
            raise ValueError(f"unknown option {opt}")
        self.model = None
        if path != "RAW":
            self.model = tf.keras.models.load_model(path, compile=False)
            act = getattr(self.model.layers[-1], "activation", None)
            name = (getattr(act, "__name__", "") or "").lower()
            self.mode = "tanh" if name == "tanh" else "sigmoid"

    def gray_batch_to_u8(self, grays):
        if self.model is None:
            return list(grays)
        x = np.asarray(grays, dtype=np.float32)[..., None] / 255.0
        x = self.a * x + self.c
        with self.tf.device("/GPU:0"):
            pred = self.model.predict(x, batch_size=16, verbose=0)[..., 0]
        if self.mode == "tanh":
            u8 = np.clip((pred + 1.0) * 127.5, 0, 255)
        else:
            u8 = np.clip(pred * 255.0, 0, 255)
        return list(u8.astype(np.uint8))

    def map_crop(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.model is None:
            return gray
        h, w = gray.shape
        ys = gmb.compute_starts(h, TILE, TILE - OVERLAP)
        xs = gmb.compute_starts(w, TILE, TILE - OVERLAP)
        out = np.zeros_like(gray)
        plan = [(i, j, y, x) for i, y in enumerate(ys) for j, x in enumerate(xs)]
        co = OVERLAP // 2
        for s in range(0, len(plan), 64):
            part = plan[s : s + 64]
            preds = self.gray_batch_to_u8([gray[y : y + TILE, x : x + TILE] for _, _, y, x in part])
            for (i, j, y, x), p in zip(part, preds):
                t = co if i > 0 else 0
                bt = co if i < len(ys) - 1 else 0
                le = co if j > 0 else 0
                ri = co if j < len(xs) - 1 else 0
                hh = min(TILE - bt - t, h - (y + t))
                ww = min(TILE - ri - le, w - (x + le))
                out[y + t : y + t + hh, x + le : x + le + ww] = p[t : t + hh, le : le + ww]
        return out

    def close(self):
        if self.model is not None:
            del self.model
            self.tf.keras.backend.clear_session()


_MAP_CACHE = {}


def map_crop_rgb(mpath, region):
    """Search-map crop, read once per process (a crop is ~1.2 GB decompressed)."""
    key = (str(mpath), region)
    if key not in _MAP_CACHE:
        e0, n0, e1, n1 = region
        with rasterio.open(mpath) as ds:
            r0, c0 = rowcol(ds.transform, e0, n1)
            r1, c1 = rowcol(ds.transform, e1, n0)
            rgb, _ = read_rgba(ds, c0, r0, c1 - c0, r1 - r0)
            _MAP_CACHE[key] = (rgb, r0, c0, ds.transform)
    return _MAP_CACHE[key]


def evaluate(runner, regions, queries_by_region, variants, workers):
    rows = []
    for rname in regions:
        qs = queries_by_region[rname]
        for qside, mside, qpath, mpath in (
            ("google", "bing", GOOGLE, BING),
            ("bing", "google", BING, GOOGLE),
        ):
            rgb, r0, c0, mt = map_crop_rgb(mpath, REGIONS[rname])
            t0 = time.perf_counter()
            mp = runner.map_crop(rgb)
            map_s = time.perf_counter() - t0
            pyramid = gmb.build_pyramid(mp, FACTORS)
            for variant in variants:
                grays = []
                for k, q in enumerate(qs):
                    qrgb, ce, cn = q[qside]
                    if variant == "hard_v1":
                        qid = f"{rname}_{k:04d}_{qside}"
                        qrgb, _ = gmb.augment_hard_v1(qrgb, seed=gmb.hard_v1_seed(42, qid))
                    grays.append(cv2.cvtColor(qrgb, cv2.COLOR_RGB2GRAY))
                outs = runner.gray_batch_to_u8(grays)
                templates = [o[CROP:-CROP, CROP:-CROP] for o in outs]

                def one(k):
                    tpl = templates[k]
                    _, ce, cn = qs[k][qside]
                    if float(np.std(tpl)) < 2.0:
                        return (k, None, "rejected")
                    try:
                        res = gmb.coarse_to_fine_search(
                            pyramid, tpl, factors=FACTORS, top_k=30,
                            refine_radius_full_px=160, nms_radius_full_px=256,
                        )
                    except Exception as exc:  # noqa: BLE001
                        return (k, None, f"error:{exc}")
                    px = res.x + tpl.shape[1] / 2.0 + c0
                    py = res.y + tpl.shape[0] / 2.0 + r0
                    pe, pn = gmb.pixel_center_to_geo(mt, px, py)
                    return (k, math.hypot(pe - ce, pn - cn), "ok", res.top1_score, res.peak_margin)

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(one, range(len(qs))))
                for item in results:
                    rows.append({
                        "region": rname, "direction": f"{qside}2{mside}", "variant": variant,
                        "k": item[0], "error_m": item[1], "status": item[2],
                        "top1": item[3] if len(item) > 3 else None,
                        "margin": item[4] if len(item) > 4 else None,
                        "map_seconds": map_s,
                    })
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["direction"], r["variant"]), []).append(r)
    out = {}
    for key, items in sorted(groups.items()):
        errs = [x["error_m"] for x in items]
        succ = [1.0 if (e is not None and e <= 25.0) else 0.0 for e in errs]
        out[f"{key[0]}/{key[1]}"] = round(float(np.mean(succ)), 4)
    out["mean"] = round(float(np.mean([v for k, v in out.items()])), 4)
    allerr = [r["error_m"] for r in rows]
    out["n"] = len(rows)
    out["s5"] = round(float(np.mean([1.0 if (e is not None and e <= 5) else 0.0 for e in allerr])), 4)
    for region in sorted({r["region"] for r in rows}):
        sub = [r for r in rows if r["region"] == region]
        out[f"region:{region}"] = round(
            float(np.mean([1.0 if (r["error_m"] is not None and r["error_m"] <= 25.0) else 0.0 for r in sub])), 4
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--regions", default="train")
    ap.add_argument("--queries-per-region", type=int, default=100)
    ap.add_argument("--variants", default="clean,hard_v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-raw", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices("GPU"):
        print("UYARI: GPU yok, CPU kullaniliyor", flush=True)

    regions = REGION_SETS.get(args.regions, tuple(args.regions.split(",")))
    variants = tuple(args.variants.split(","))
    t0 = time.perf_counter()
    queries = {}
    for i, rname in enumerate(regions):
        queries[rname] = sample_queries(REGIONS[rname], args.queries_per_region, args.seed + i)
    print(f"sorgular hazir: { {k: len(v) for k, v in queries.items()} } ({time.perf_counter()-t0:.0f}s)", flush=True)

    existing = {}
    if args.out.is_file():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
    specs = ([] if args.no_raw else ["RAW"]) + args.model
    for spec in specs:
        key = f"{spec}|{args.regions}|{args.queries_per_region}|{args.variants}|{args.seed}"
        if key in existing:
            print(f"[atla] {spec}: {existing[key]['summary']}", flush=True)
            continue
        t1 = time.perf_counter()
        runner = ModelRunner(spec)
        rows = evaluate(runner, regions, queries, variants, args.workers)
        runner.close()
        summ = summarize(rows)
        existing[key] = {"spec": spec, "summary": summ, "seconds": round(time.perf_counter() - t1, 1)}
        # Other proxy processes may share this file: merge instead of overwrite.
        if args.out.is_file():
            on_disk = json.loads(args.out.read_text(encoding="utf-8"))
            on_disk.update(existing)
            existing = on_disk
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, args.out)
        print(f"{summ['mean']:.3f}  {summ}  {Path(spec.split('::')[0]).name[:70]} {spec.partition('::')[2]} ({time.perf_counter()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
