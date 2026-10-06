#!/usr/bin/env python
"""Rota karelerinin EXIF/XMP tanisi: yaw kaynagi ve irtifa tutarliligi (GPU/DEM gerekmez).

    python deney_araclari/exif_tani.py [--rotalar ../template-matching/guzergahlar] [--out RAPOR.md]

Olculenler (rota basina):
  * TM'nin kullandigi yaw: MakerNote "FlightDegree" (govde/ucus yawi) / 10.
  * Kameranin gercek yonu: MakerNote "GimbalDegree" (yoksa XMP GimbalYawDegree) / 10.
    Nadir cekimde kare yonunu gimbal yawi belirler; fark, sablonun kuzeye yanlis
    dondurulme acisidir.
  * Irtifa: EXIF GPSAltitude - XMP RelativeAltitude rota icinde sabit olmali (kalkis
    noktasinin mutlak yuksekligi). Sacilim, TM'nin GPSAltitude - DEM ile hesapladigi
    yerden yukseklikteki gurultuyu, dolayisiyla olcek hatasini gosterir.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

TM_ROUTES = Path(__file__).resolve().parents[2] / "template-matching" / "guzergahlar"
_DEG = r"\(Y,P,R\):([+-]?\d+),([+-]?\d+),([+-]?\d+)"


def _angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def read_frame(path: Path):
    with Image.open(path) as im:
        exif = im._getexif() or {}
    named = {TAGS.get(k): v for k, v in exif.items()}
    mk = str(named.get("MakerNote", b""))
    out = {"model": named.get("Model")}
    for key, name in (("FlightDegree", "flight_yaw"), ("GimbalDegree", "gimbal_yaw")):
        m = re.search(key + _DEG, mk)
        out[name] = float(m.group(1)) / 10.0 if m else None
        if key == "GimbalDegree" and m:
            out["gimbal_pitch"] = float(m.group(2)) / 10.0
    gps = named.get("GPSInfo") or {}
    try:
        out["gps_alt"] = float(gps.get(6)) if gps.get(6) is not None else None
    except Exception:
        out["gps_alt"] = None
    head = path.open("rb").read(262144)
    i, j = head.find(b"<x:xmpmeta"), head.find(b"</x:xmpmeta>")
    xmp = head[i:j].decode("utf-8", "ignore") if i >= 0 and j > i else ""
    for key, name in (("RelativeAltitude", "rel_alt"), ("AbsoluteAltitude", "abs_alt"), ("GimbalYawDegree", "xmp_gimbal_yaw")):
        m = re.search(key + r'="([+-]?[\d.]+)"', xmp)
        out[name] = float(m.group(1)) if m else None
    if out["gimbal_yaw"] is None and out["xmp_gimbal_yaw"] is not None:
        out["gimbal_yaw"] = out["xmp_gimbal_yaw"]
    return out


def summarize(route_dir: Path):
    frames = sorted(p for p in route_dir.iterdir() if p.suffix.lower() == ".jpg")
    rows = []
    for p in frames:
        try:
            rows.append(read_frame(p))
        except Exception:
            continue
    if not rows:
        return None
    yaw_d = [_angle_diff(r["gimbal_yaw"], r["flight_yaw"]) for r in rows if r["gimbal_yaw"] is not None and r["flight_yaw"] is not None]
    alt_d = [r["gps_alt"] - r["rel_alt"] for r in rows if r["gps_alt"] is not None and r["rel_alt"] is not None]
    rel = [r["rel_alt"] for r in rows if r["rel_alt"] is not None]
    models = sorted({str(r["model"]) for r in rows})
    res = {"rota": route_dir.name, "kare": len(rows), "kamera": ",".join(models)}
    if yaw_d:
        res.update(yaw_ort=st.mean(yaw_d), yaw_med=st.median(yaw_d), yaw_std=st.pstdev(yaw_d),
                   yaw_abs_p90=sorted(abs(d) for d in yaw_d)[int(0.9 * (len(yaw_d) - 1))],
                   yaw_5ust=sum(1 for d in yaw_d if abs(d) > 5) / len(yaw_d))
    if alt_d:
        res.update(alt_std=st.pstdev(alt_d), alt_aralik=max(alt_d) - min(alt_d))
    if rel:
        res.update(rel_med=st.median(rel))
        if alt_d:
            res["olcek_std_yuzde"] = 100.0 * res["alt_std"] / max(1.0, res["rel_med"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotalar", default=str(TM_ROUTES))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    results = [r for d in sorted(Path(args.rotalar).iterdir()) if d.is_dir() for r in [summarize(d)] if r]
    f = lambda v, n=1: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    lines = ["# EXIF tanisi: yaw kaynagi ve irtifa tutarliligi", "",
             "Yaw farki = gimbal (kamera) yawi - ucus (govde) yawi; TM bugun ucus yawini kullaniyor.",
             "Irtifa sacilimi = rota icinde std(GPSAltitude - RelativeAltitude); olcek std % = bunun yerden yukseklige orani.", "",
             "| Rota | kare | kamera | yaw farki ort | medyan | std | abs p90 | >5 derece | RelAlt medyan (m) | irtifa std (m) | aralik (m) | olcek std % |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['rota']} | {r['kare']} | {r['kamera']} | {f(r.get('yaw_ort'))} | {f(r.get('yaw_med'))} | {f(r.get('yaw_std'))} | "
                     f"{f(r.get('yaw_abs_p90'))} | {f(r.get('yaw_5ust') and 100 * r['yaw_5ust'], 0) if r.get('yaw_5ust') is not None else '-'}% | "
                     f"{f(r.get('rel_med'), 0)} | {f(r.get('alt_std'))} | {f(r.get('alt_aralik'))} | {f(r.get('olcek_std_yuzde'))} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
