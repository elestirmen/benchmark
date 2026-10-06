#!/usr/bin/env python
"""Coklu guzergah ucus testi ozeti: rota ve bolge bazinda basari + eslestirilmis McNemar.

    python deney_araclari/ucus_ozet.py [--kok deneyler/2026-09-25_genis_dogrulama/sonuclar/ucus]

rota_<ad>/<model>.json (ucus_testi.py test ciktisi) ve karlik_f1/ (guzergah_5) okunur.
Bolgeler: Urgup merkezi (tezde_ucus_*, guzergah_1..4) ve Karlik (guzergah_5..8).
Bolge toplaminda her rota-kare cifti bir gozlem; McNemar yalnizca iki modelin de
isledigi kareler uzerinde (kare kumesi TM'nin geometri filtresine bagli, modelden bagimsiz).
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = BENCH / "deneyler" / "2026-09-25_genis_dogrulama" / "sonuclar" / "ucus"
KARLIK = {"guzergah_5_to_karain", "guzergah_6_to_tagar", "guzergah_7_to_gurlevek", "guzergah_8_koy_vadi"}
THRESHOLDS = (25, 50, 70)


def load(root: Path) -> dict[str, dict[str, dict[int, float]]]:
    """{rota: {model: {kare: err_m}}}"""
    data: dict[str, dict[str, dict[int, float]]] = {}
    dirs = [(d.name[len("rota_"):], d) for d in root.glob("rota_*") if d.is_dir()]
    if (root / "karlik_f1").is_dir():
        dirs.append(("guzergah_5_to_karain", root / "karlik_f1"))
    for route, directory in dirs:
        for path in directory.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            # kare=0: TM hic kare islememis (ör. guzergah_8: alcak ucus + genis aci, kare patch'ten kucuk)
            if not payload.get("kareler") or int(payload.get("max_frames") or 0) > 0:
                continue
            data.setdefault(route, {})[Path(payload["model"]).stem] = {f["frame"]: f["err_m"] for f in payload["kareler"]}
    return data


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def pooled(data, routes, models):
    """Bolge toplami; yalniz tum modellerin sonucu olan rota-kareler (adil kiyas)."""
    rows = {m: {(r, k): e for r in routes for k, e in data[r].get(m, {}).items()} for m in models}
    common = set.intersection(*(set(v) for v in rows.values())) if rows else set()
    return {m: {k: v[k] for k in common} for m, v in rows.items()}


def table(title, rows, models):
    print(f"\n## {title}\n")
    print("| Model | kare | ≤25 m | ≤50 m | ≤70 m | medyan |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in sorted(models, key=lambda m: -rate(rows[m], 25)):
        errs = sorted(rows[m].values())
        print(f"| {m} | {len(errs)} | {rate(rows[m], 25):.3f} | {rate(rows[m], 50):.3f} | {rate(rows[m], 70):.3f} | {errs[len(errs) // 2]:.0f} m |")


def rate(errs: dict, limit: float) -> float:
    return sum(1 for e in errs.values() if e < limit) / len(errs) if errs else float("nan")


def pairs(title, rows, models, refs):
    print(f"\n### {title}: eslestirilmis McNemar (yalniz A / yalniz B basarili kare)\n")
    print("| A | B | ≤25 m | p | ≤70 m | p |")
    print("|---|---|---:|---:|---:|---:|")
    seen = set()
    for a in models:
        for b in refs:
            if a == b or frozenset((a, b)) in seen:
                continue
            seen.add(frozenset((a, b)))
            keys = rows[a].keys() & rows[b].keys()
            cells = []
            for t in (25, 70):
                x = sum(1 for k in keys if rows[a][k] < t <= rows[b][k])
                y = sum(1 for k in keys if rows[b][k] < t <= rows[a][k])
                cells += [f"{x}/{y}", f"{mcnemar(x, y):.3f}"]
            print(f"| {a} | {b} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", default=str(DEFAULT_ROOT))
    ap.add_argument("--modeller", nargs="*", default=None, help="Yalniz bu modeller (varsayilan: tum rotalarda olanlar)")
    args = ap.parse_args()
    data = load(Path(args.kok))
    models = args.modeller or sorted(set.intersection(*(set(v) for v in data.values())))
    regions = {
        "Urgup merkezi": sorted(r for r in data if r not in KARLIK),
        "Karlik": sorted(r for r in data if r in KARLIK),
    }
    regions["Tum rotalar"] = regions["Urgup merkezi"] + regions["Karlik"]

    print("# Rota bazinda ≤25 m basari\n")
    print("| Rota | kare | " + " | ".join(models) + " |")
    print("|---|---:|" + "---:|" * len(models))
    for name, routes in regions.items():
        if name == "Tum rotalar":
            continue
        for r in routes:
            n = max(len(data[r].get(m, {})) for m in models)
            cells = [data[r].get(m) for m in models]
            vals = [rate(c, 25) if c else float("nan") for c in cells]
            best = max(v for v in vals if v == v)
            print(f"| {r} ({name}) | {n} | " + " | ".join(
                ("**%.3f**" if v == best else "%.3f") % v for v in vals) + " |")

    for name, routes in regions.items():
        if not routes:
            continue
        rows = pooled(data, routes, models)
        done = sorted({r for k in next(iter(rows.values()), {}) for r in [k[0]]})
        table(f"{name} ({len(done)}/{len(routes)} rota, tum modellerin ortak kareleri)", rows, models)
        ranked = sorted(models, key=lambda m: -rate(rows[m], 25))
        refs = [m for m in ("RAW_kimlik",) if m in models] + ranked[:2]
        pairs(name, rows, ranked, list(dict.fromkeys(refs)))


if __name__ == "__main__":
    main()
