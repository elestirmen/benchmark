#!/usr/bin/env python
"""Gercek IHA ucus testi: Karlik Bing haritasi + DJI kareleri (template-matching BENCHMARK modu).

    python ucus_testi.py harita --model M.h5 --out-dir D     # Karlik model haritasi
    python ucus_testi.py harita --model M.h5 --out-dir D --kaynak urgup.tif --etiket urgup
    python ucus_testi.py test   --model M.h5 --harita D/M_urgup_geo.tif --anlik ROTA --dem DEM.tif --out D/M.json
    python ucus_testi.py test   --model M.h5 --harita D/M_geo.tif --out D/M.json
    python ucus_testi.py kimlik --out RAW_kimlik.h5          # RAW icin ~kimlik model

harita: benchmark'in build_model_map'i ile ayni karo plani (544 karo, 32 bindirme,
        16 px kirpma) ve ayni on isleme (minus1_1, enhancement none); yanina bu
        sozlesmeyi .pre.json olarak yazar.
test:   template-matching ana betiginin run_pipeline'i BENCHMARK=True ile (her kare
        bagimsiz, EXIF GPS merkezli 5000 px pencere). Kareler de ayni on islemeyle
        (HAM + minus1_1, sozlesmeler kapali) modelden gecer. Kare basina hata (err_m)
        ve TM skoru (max_val2) 2026-09-12 karlik_top_modeller testindeki geri cagri
        alanlaridir. --eski-protokol ile harita/model sozlesmeleri acik kalir (o testin
        birebir tekrari icin).
kimlik: sigmoid(1.5 x) tek 1x1 conv; RAW'a (modelsiz eslestirme) monoton bir yakinsama.

Sonuc dosyalari (sonuclar.csv vb.) template-matching klasorune degil, --out'un
klasorune yazilir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
TM_ROOT = BENCH.parent / "template-matching"
TM_CORE = "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py"
KARLIK_SOURCE = Path(r"E:\tez__ayna\python_calismalar\AutoEncoder_pix2pix\karlik_30_cm_bingmap_utm.tif")
ROUTE = TM_ROOT / "guzergahlar" / "guzergah_5_to_karain"
DEM = TM_ROOT / "karlik_30_cm_bingmap_utm_elevation.tif"
TILE, OVERLAP = 544, 32
# F1: resmi uydu benchmark'inin on islemesi (her modele minus1_1).
# F2: modellerin egitim on islemesi (zero_1); uyumsuzlugun ucusta etkisini ayirmak icin.
PROTOCOLS = {
    "F1": {"input_normalization": "minus1_1", "enhancement": "none"},
    "F2": {"input_normalization": "zero_1", "enhancement": "none"},
}
MAP_SUFFIX = {"F1": "_geo.tif", "F2": "_geo_z1.tif"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_harita(args):
    sys.path.insert(0, str(BENCH))
    import geospatial_model_benchmark as gmb

    model = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    source = Path(args.kaynak).resolve()
    tag = f"_{args.etiket}" if args.etiket else ""
    meta = gmb.streaming_map_metadata(source, tile_size=TILE, overlap=OVERLAP)
    protocol = PROTOCOLS[args.protokol]
    work = out_dir / "_calisma" / f"{gmb.safe_slug(model.stem)}{tag}_{args.protokol}"
    runtime = gmb.import_loaded_model_runtime().load(
        str(model), color_mode="grayscale", normalization=protocol["input_normalization"],
        enhancement=protocol["enhancement"], require_gpu=not args.cpu, output_value_mode="auto",
    )
    started = time.time()
    geo, _ = gmb.build_model_map(
        model, work, None, meta, source, tile_size=TILE, overlap=OVERLAP,
        batch_size=args.batch_size, normalization=protocol["input_normalization"],
        enhancement=protocol["enhancement"], force=False, keep_intermediate=False,
        model_runtime=runtime, model_sha256=sha256(model),
    )
    target = out_dir / f"{model.stem}{tag}{MAP_SUFFIX[args.protokol]}"
    os.replace(geo, target)
    sidecar = dict(protocol, protokol=args.protokol, color_mode="grayscale", image_size=TILE, model=model.name,
                   source_image=source.name, produced_by="deney_araclari/ucus_testi.py harita")
    target.with_name(target.stem + ".pre.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"HARITA {target}  ({time.time() - started:.0f} s)", flush=True)


def cmd_kimlik(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import h5py  # noqa: F401
    import numpy as np
    import tensorflow as tf

    inp = tf.keras.Input(shape=(TILE, TILE, 1))
    layer = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="model_output")
    model = tf.keras.Model(inp, layer(inp))
    layer.set_weights([np.full((1, 1, 1, 1), 1.5, np.float32), np.zeros(1, np.float32)])
    model.save(args.out)
    print(f"KIMLIK {args.out}")


class _NoIntersectionCounter:
    """stdout'u gecirir; ana betigin "kesişim yok" satirlarini sayar (kareye baglamak icin)."""

    def __init__(self, stream):
        self.stream = stream
        self.count = 0

    def write(self, text):
        self.count += text.count("kesişim yok")
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _Yakalayici:
    """--yakala: kare basina modele giren 3 yama, merkez yamanin model ciktisi, harita
    penceresinin GPS merkezli kesiti ve merkez sablonun eslesme konumu (.npz).

    Ana betige dokunmadan prepare_model_input ve match_three sarilir. BENCHMARK modunda
    5000 px pencere EXIF GPS pikseline oturur: tam pencerede GPS = (2500, 2500), GPS'e
    tam oturan sablonun sol ustu 2500 - 256 = 2244. Kesit [2244 - pay, 2244 + 512 + pay].
    Kayit, kareyi bitiren ERR satirinda (kare no + hata ile) yapilir.
    """

    def __init__(self, tm, out_dir, scale, pay):
        self.tm, self.dir, self.scale, self.pay = tm, Path(out_dir), scale, int(pay)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.inputs, self.pending, self.saved = [], None, 0
        orig_prep, orig_match = tm.prepare_model_input, tm.match_three

        def prep(*a, **k):
            x = orig_prep(*a, **k)
            self.inputs = (self.inputs + [x])[-3:]
            return x

        def match(img, templates):
            res = orig_match(img, templates)
            self._hold(img, templates, res)
            return res

        tm.prepare_model_input, tm.match_three = prep, match

    def _to_u8(self, x):
        x = np.asarray(x, dtype=np.float32)
        u = (x + 1.0) * 127.5 if self.scale == "minus1_1" else x * 255.0
        return np.clip(np.rint(u), 0, 255).astype(np.uint8)

    def _hold(self, img, templates, res):
        import cv2

        if len(self.inputs) != 3:
            self.pending = None
            return
        tsize = int(templates[1].shape[0])
        size = int(self.tm.RUN_CFG["CERCEVE_BOYUTU_BENCHMARK"])
        r0 = size // 2 - tsize // 2 - self.pay
        r1 = r0 + tsize + 2 * self.pay
        full = img.shape[0] >= size and img.shape[1] >= size
        crop = img[r0:r1, r0:r1] if full else np.zeros((r1 - r0, r1 - r0), np.uint8)
        _, val2, _, loc2 = cv2.minMaxLoc(res[1])
        self.pending = {
            "giris": np.stack([self._to_u8(x) for x in self.inputs]),
            "sablon": np.asarray(templates[1], dtype=np.uint8),
            "harita": np.ascontiguousarray(crop, dtype=np.uint8),
            "kesit_ofset": np.array([r0, r0]),
            "cerceve_shape": np.array(img.shape[:2]),
            "tam_cerceve": np.array(full),
            "max_loc2_xy": np.array(loc2),
            "max_val2": np.array(val2),
        }

    def save(self, frame, err_m, tm_score):
        if self.pending is None:
            return
        np.savez_compressed(self.dir / f"kare_{frame:04d}.npz", kare=frame, err_m=err_m, tm_score=tm_score, **self.pending)
        self.pending = None
        self.saved += 1


def cmd_test(args):
    model = Path(args.model).resolve()
    harita = Path(args.harita).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    yakala_dir = Path(args.yakala).resolve() if args.yakala else None  # chdir oncesi
    overrides = {
        "BENCHMARK": True,
        "DEBUG": False,
        "HARITA_DIR": str(harita.parent),
        "MODEL_DIR": str(model.parent),
        "HARITA_DOSYALARI": [str(harita)],
        "MODEL_DOSYALARI": [str(model)],
        "ANLIK_DIR": str(Path(args.anlik).resolve()),
        "DEM_PATH": str(Path(args.dem).resolve()),
        "MAX_FRAMES": int(args.max_frames),
        "SORT_INPUTS": True,
    }
    if not args.eski_protokol:
        scale = PROTOCOLS[args.protokol]["input_normalization"]
        overrides.update(MODEL_INPUT_NORM="HAM", MODEL_INPUT_SCALE=scale, MODEL_INPUT_USE_SIDECAR=False)
        # Kesisim bulunamayan karede tahmin arama MERKEZI olmamali: BENCHMARK modunda
        # merkez EXIF GPS'in kendisidir, hata 0 m cikar (2026-09-25: RAW'da 191 karenin
        # 32'si boyle "basarili" sayildi). False -> tekil sablonun gercek eslesmesi.
        overrides["NO_INTERSECTION_USE_SEARCH_CENTER"] = False
    os.environ["TM_RUN_CFG_JSON"] = json.dumps(overrides)
    os.chdir(out.parent)  # sonuclar.csv / modele_gore_sonuclar.txt buraya
    sys.path.insert(0, str(TM_ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location("tm_core", str(TM_ROOT / TM_CORE))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    yakala = None
    if yakala_dir is not None:
        yakala = _Yakalayici(tm, yakala_dir, PROTOCOLS[args.protokol]["input_normalization"], args.yakala_pay)
    frames = []
    last = {}
    counter = _NoIntersectionCounter(sys.stdout)
    sys.stdout = counter
    seen = {"n": 0}

    # on_frame YOK: ana betik her karede buyuk gorsellestirme goruntusunu kopyalayip
    # on_frame'e verir (res.copy()); ensemble/TPU kosularinda surec 12+ GB'a cikip
    # commit sinirinda sessizce coktu. Kare ozet satiri ayni alanlari tasir.
    pattern = re.compile(r"^\[(\d+)/(\d+)\] ERR=(-?\d+)m\s+ACC=([\d.]+)%\s+TM=([-\d.]+)")

    def on_log(message):
        print(message, flush=True)
        match = pattern.match(str(message))
        if not match:
            return
        frames.append({
            "frame": int(match.group(1)), "err_m": float(match.group(3)), "tm_score": float(match.group(5)),
            "kesisim_yok": counter.count > seen["n"],
        })
        seen["n"] = counter.count
        last.update(total=int(match.group(2)), acc=float(match.group(4)))
        if yakala is not None:
            yakala.save(int(match.group(1)), float(match.group(3)), float(match.group(5)))

    started = time.time()
    tm.run_pipeline(callbacks={"on_log": on_log, "stop_check": lambda: False})
    errs = sorted(f["err_m"] for f in frames)
    n = len(errs)

    def rate(limit):
        return sum(1 for e in errs if e < limit) / n if n else None

    summary = {
        "model": model.name,
        "model_sha256": sha256(model),
        "harita": str(harita),
        "anlik": str(Path(args.anlik).resolve()),
        "protokol": "eski (sozlesmeler acik)" if args.eski_protokol else (
            f"{args.protokol}: {PROTOCOLS[args.protokol]['input_normalization']} + none, kesisimsiz kare = tekil sablon"),
        "kare": n,
        "max_frames": int(args.max_frames),
        "success_25m": rate(25), "success_50m": rate(50), "success_70m": rate(70),
        "tm_basari_yuzde": last.get("acc"),
        "kesisim_yok_kare": sum(1 for f in frames if f["kesisim_yok"]),
        "medyan_m": errs[n // 2] if n else None,
        "ortalama_m": sum(errs) / n if n else None,
        "sure_s": time.time() - started,
        "kareler": frames,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"UCUS {model.name}: kare={n} <=70m={summary['success_70m']} <=25m={summary['success_25m']} medyan={summary['medyan_m']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harita")
    h.add_argument("--model", required=True)
    h.add_argument("--out-dir", required=True)
    h.add_argument("--batch-size", type=int, default=8)
    h.add_argument("--cpu", action="store_true")
    h.add_argument("--protokol", choices=sorted(PROTOCOLS), default="F1")
    h.add_argument("--kaynak", default=str(KARLIK_SOURCE), help="Model haritasinin uretilecegi uydu rasteri")
    h.add_argument("--etiket", default="", help="Cikti adina eklenir (<model>_<etiket>_geo.tif); Karlik disi kaynaklar icin")
    t = sub.add_parser("test")
    t.add_argument("--model", required=True)
    t.add_argument("--harita", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--anlik", default=str(ROUTE))
    t.add_argument("--dem", default=str(DEM))
    t.add_argument("--max-frames", type=int, default=0)
    t.add_argument("--eski-protokol", action="store_true")
    t.add_argument("--protokol", choices=sorted(PROTOCOLS), default="F1")
    t.add_argument("--yakala", default="", help="kare basina egitim verisi (.npz) bu klasore (bkz. _Yakalayici)")
    t.add_argument("--yakala-pay", type=int, default=384, help="harita kesitinin sablon cevresindeki payi (px)")
    k = sub.add_parser("kimlik")
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    {"harita": cmd_harita, "test": cmd_test, "kimlik": cmd_kimlik}[args.cmd](args)


if __name__ == "__main__":
    main()
