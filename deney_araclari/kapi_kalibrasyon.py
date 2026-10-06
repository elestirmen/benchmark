#!/usr/bin/env python
"""Kapi oranlari resmi sonuclari ne kadar iyi siraliyor? (protokol basina Spearman)

    python kapi_kalibrasyon.py <kapi4.json> [--eski <kapi1.json> ...]

Eski tek grup kapi (G2B temiz) varsayilan olarak ayni olcumun g2b_clean grubundan okunur
(ayni merkezler, ayni seed; ayri JSON gerekmez).

Resmi sonuclar sonuc_panosu.collect() ile okunur; eslesme dosya sha256'si uzerinden
(kapi JSON'unda yoksa dosya adi -> model dizini). Olcut: global success_25m'nin ayni
protokolun RAW'ina gore farki (kucuk kapi orani = yuksek fark beklenir, rho < 0 iyi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sonuc_panosu as pano  # noqa: E402


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2.0
        i = j + 1
    return result


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def gate_scores(paths, index, group=None):
    """sha256 -> oran; group verilirse (ör. g2b_clean) o grubun orani (eski tek grup kapi)."""
    scores = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, score in payload.get("results", {}).items():
            if name == "RAW" or not isinstance(score, dict) or "false_peak_ratio" not in score:
                continue
            sha = score.get("model_sha256")
            if not sha:
                candidates = index["by_name"].get(Path(name).name, set())
                sha = next(iter(candidates)) if len(candidates) == 1 else None
            if sha:
                value = score["false_peak_ratio"] if group is None else (score.get("groups") or {}).get(group, {}).get("false_peak_ratio")
                if value is not None:
                    scores[sha] = float(value)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kapi", nargs="+")
    ap.add_argument("--eski", nargs="*", default=[], help="ayri eski kapi JSON'lari (verilmezse ayni olcumun g2b_clean grubu)")
    args = ap.parse_args()
    data = pano.collect()
    index = data["index"]
    new = gate_scores(args.kapi, index)
    old = gate_scores(args.eski, index) if args.eski else gate_scores(args.kapi, index, group="g2b_clean")
    for key, label, _ in pano.protocol_order(data["records"]):
        rows = [
            r for r in data["records"]
            if r["protocol_key"] == key and not r["is_raw"] and r.get("delta_raw") is not None and r.get("sha256") in new
        ]
        if len(rows) < 3:
            continue
        rows.sort(key=lambda r: new[r["sha256"]])
        print(f"\n== {label}: {len(rows)} model")
        print(f"{'kapi4':>6} {'kapi1':>6} {'global':>6} {'dRAW':>7}  model")
        for r in rows:
            k1 = old.get(r["sha256"])
            print(f"{new[r['sha256']]:6.3f} {k1 if k1 is None else round(k1, 3)!s:>6} {r['global']:6.3f} {r['delta_raw']:+7.3f}  {r['name'][:70]}")
        rho4 = spearman([new[r["sha256"]] for r in rows], [r["delta_raw"] for r in rows])
        both = [r for r in rows if r["sha256"] in old]
        rho1 = spearman([old[r["sha256"]] for r in both], [r["delta_raw"] for r in both]) if both else None
        rho4b = spearman([new[r["sha256"]] for r in both], [r["delta_raw"] for r in both]) if both else None
        print(f"Spearman(kapi4, dRAW) = {rho4:.3f}" if rho4 is not None else "Spearman yok")
        if rho1 is not None:
            print(f"ayni {len(both)} modelde: kapi4 {rho4b:.3f} / eski kapi1 {rho1:.3f}  (daha negatif = daha iyi siralama)")


if __name__ == "__main__":
    main()
