#!/usr/bin/env python
"""Ucus testi A/B karari: ayni modelin iki ayar kolunu kare bazinda eslestirir ve
ONCEDEN yazilmis karar kuralini uygular (bkz. deneyler/2026-10-06_tm_iyilestirme/BENIOKU.md).

    python deney_araclari/ab_karsilastir.py --a KOK_A --b KOK_B --b-adi gsd_area --rapor KARAR.md
    python deney_araclari/ab_karsilastir.py --a KOK_A --b KOK_KONTROL --birebir

KOK, ucus_ozet.load'in okudugu duzendir: rota_<ad>/<model>.json (karlik_f1/ = guzergah_5).
A = mevcut varsayilan (referans), B = aday ayar.

Karar kurali (yalniz birincil model, varsayilan topluluk_x1_2022):
  * Olcut: ortak kareler uzerinde <=25 m ve <=70 m basari; eslestirilmis McNemar.
    GPS'i sabit ~35 m kayik iki rota (guzergah_1_donus, guzergah_2_tezde_ucus_1)
    <=25 m hesabinda disarida, <=70 m'de iceride.
  * KABUL : tum rotalarda <=25 m farki > 0 ve p < 0.05, VE hicbir bolgede
            (tum, Urgup merkezi, Karlik) <=25 m ya da <=70 m'de anlamli dusus yok.
  * RED   : herhangi bir bolgede <=25 m ya da <=70 m'de anlamli dusus (fark < 0, p < 0.05).
  * BELIRSIZ: digerleri. Varsayilan degismez; A yerinde kalir.
--birebir: B'deki her kare A'dakiyle ayni hatayi vermeli (varsayilan davranis
degismedi mi kontrolu); fark varsa cikis kodu 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ucus_ozet import KARLIK, load, mcnemar  # noqa: E402

GPS_KAYMALI = {"guzergah_1_donus", "guzergah_2_tezde_ucus_1"}
ALPHA = 0.05
LIMITS = (25, 50, 70)
KARAR_LIMITLERI = (25, 70)


def _frames(data, route, model):
    return data.get(route, {}).get(model, {})


def paired(a_data, b_data, model, routes, limit, skip_gps=True):
    """Ortak kareler uzerinde A ve B basarisi + eslestirilmis McNemar.

    skip_gps: <=25 m'de GPS'i sabit kayik rotalari disarida birak (karar olcutu)."""
    keys = []
    for r in routes:
        if skip_gps and limit == 25 and r in GPS_KAYMALI:
            continue
        fa, fb = _frames(a_data, r, model), _frames(b_data, r, model)
        keys += [(r, k) for k in fa.keys() & fb.keys()]
    n = len(keys)
    if not n:
        return None
    ea = {k: a_data[k[0]][model][k[1]] for k in keys}
    eb = {k: b_data[k[0]][model][k[1]] for k in keys}
    sa = sum(1 for k in keys if ea[k] < limit)
    sb = sum(1 for k in keys if eb[k] < limit)
    only_b = sum(1 for k in keys if eb[k] < limit <= ea[k])
    only_a = sum(1 for k in keys if ea[k] < limit <= eb[k])
    return {"n": n, "a": sa / n, "b": sb / n, "fark": (sb - sa) / n,
            "yalniz_b": only_b, "yalniz_a": only_a, "p": mcnemar(only_b, only_a)}


def regions(routes):
    urgup = [r for r in routes if r not in KARLIK]
    karlik = [r for r in routes if r in KARLIK]
    return {"Tum rotalar": routes, "Urgup merkezi": urgup, "Karlik": karlik}


def compare(a_data, b_data, model):
    routes = sorted(r for r in set(a_data) & set(b_data) if _frames(a_data, r, model) and _frames(b_data, r, model))
    out = {}
    for name, rs in regions(routes).items():
        for limit in LIMITS:
            res = paired(a_data, b_data, model, rs, limit)
            if res is not None:
                out[(name, limit)] = res
    return routes, out


def decide(results):
    if ("Tum rotalar", 25) not in results:
        return "BELIRSIZ", "ortak kare yok"
    drops = [f"{reg} <={lim} m" for (reg, lim), v in results.items()
             if lim in KARAR_LIMITLERI and v["fark"] < 0 and v["p"] < ALPHA]
    if drops:
        return "RED", "anlamli dusus: " + ", ".join(drops)
    prim = results[("Tum rotalar", 25)]
    if prim["fark"] > 0 and prim["p"] < ALPHA:
        return "KABUL", f"tum rotalar <=25 m +{prim['fark']:.3f} (p = {prim['p']:.3g}), hicbir bolgede anlamli dusus yok"
    return "BELIRSIZ", f"tum rotalar <=25 m {prim['fark']:+.3f} (p = {prim['p']:.3g}); varsayilan degismez"


def report(a_data, b_data, models, primary, b_name):
    lines = [f"# A/B karari: {b_name}", "",
             "A = mevcut varsayilan, B = aday. Ortak kareler; <=25 m'de GPS kaymali iki rota disarida.", ""]
    verdict = None
    for model in models:
        routes, res = compare(a_data, b_data, model)
        if not res:
            lines += [f"## {model}", "", "ortak sonuc yok", ""]
            continue
        lines += [f"## {model}{' (birincil, karar bu modelden)' if model == primary else ''}", "",
                  "| Bolge | esik | kare | A | B | fark | yalniz B / yalniz A | p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for (reg, lim), v in res.items():
            lines.append(f"| {reg} | <={lim} m | {v['n']} | {v['a']:.3f} | {v['b']:.3f} | {v['fark']:+.3f} | "
                         f"{v['yalniz_b']} / {v['yalniz_a']} | {v['p']:.3g} |")
        lines += ["", "| Rota | kare | A <=25 m | B <=25 m | A <=70 m | B <=70 m |", "|---|---:|---:|---:|---:|---:|"]
        for r in routes:
            r70 = paired(a_data, b_data, model, [r], 70)
            r25 = paired(a_data, b_data, model, [r], 25, skip_gps=False)
            lines.append(f"| {r}{' (GPS kaymali, kararda <=25 m disarida)' if r in GPS_KAYMALI else ''} | {r70['n']} | "
                         f"{r25['a']:.3f} | {r25['b']:.3f} | {r70['a']:.3f} | {r70['b']:.3f} |")
        lines.append("")
        if model == primary:
            verdict = decide(res)
    if verdict is None:
        verdict = ("BELIRSIZ", f"birincil model ({primary}) iki kolda da yok")
    lines += ["## Karar", "", f"**{verdict[0]}**: {verdict[1]}", ""]
    if verdict[0] == "KABUL":
        lines.append("Uygulama: varsayilani ayri bir commit ile degistir (geri donus: o commit'i `git revert`).")
    else:
        lines.append("Uygulama: varsayilan degismez; bayrak kodda kalir, etkisi yok.")
    return "\n".join(lines) + "\n", verdict[0]


def birebir(a_data, b_data):
    farkli, toplam, eksik = 0, 0, []
    for r, models in b_data.items():
        for m, frames in models.items():
            ref = _frames(a_data, r, m)
            if not ref:
                eksik.append(f"{r}/{m}")
                continue
            if set(ref) != set(frames):
                farkli += len(set(ref) ^ set(frames))
            for k in set(ref) & set(frames):
                toplam += 1
                if abs(ref[k] - frames[k]) > 1e-9:
                    farkli += 1
    return farkli, toplam, eksik


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="referans (mevcut varsayilan) kok")
    ap.add_argument("--b", required=True, help="aday kol koku")
    ap.add_argument("--b-adi", default="aday")
    ap.add_argument("--modeller", nargs="*", default=None)
    ap.add_argument("--birincil", default="topluluk_x1_2022")
    ap.add_argument("--rapor", default="", help="markdown raporu bu dosyaya da yaz")
    ap.add_argument("--birebir", action="store_true")
    args = ap.parse_args(argv)
    a_data, b_data = load(Path(args.a)), load(Path(args.b))
    if args.birebir:
        farkli, toplam, eksik = birebir(a_data, b_data)
        print(f"BIREBIR: {toplam} kare karsilastirildi, farkli {farkli}" + (f"; A'da olmayan: {', '.join(eksik)}" if eksik else ""))
        return 1 if (farkli or not toplam) else 0
    models = args.modeller or sorted({m for d in b_data.values() for m in d})
    text, verdict = report(a_data, b_data, models, args.birincil, args.b_adi)
    print(text)
    if args.rapor:
        Path(args.rapor).parent.mkdir(parents=True, exist_ok=True)
        Path(args.rapor).write_text(text, encoding="utf-8")
    print(f"KARAR {args.b_adi}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
