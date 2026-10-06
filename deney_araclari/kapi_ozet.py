#!/usr/bin/env python
"""margin_gate.py JSON'larini tek siralı tabloda ozetler (kucuk oran iyi).

    python kapi_ozet.py sonuclar/kapi/*.json [--ilk 40] [--filtre TAM]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = ("g2b_clean", "g2b_hard_v1", "b2g_clean", "b2g_hard_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--ilk", type=int, default=60)
    ap.add_argument("--filtre", default="")
    args = ap.parse_args()
    rows = {}
    raw = None
    for path in args.json:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = payload.get("raw_false_peak_ratio", raw)
        for name, score in payload.get("results", {}).items():
            if not isinstance(score, dict) or "false_peak_ratio" not in score:
                continue
            if args.filtre and args.filtre not in name and name != "RAW":
                continue
            groups = score.get("groups") or {}
            rows[name] = (float(score["false_peak_ratio"]), [groups.get(g, {}).get("false_peak_ratio") for g in GROUPS])
    print(f"{'oran':>6}  {'g2b_c':>6} {'g2b_h':>6} {'b2g_c':>6} {'b2g_h':>6}  model   (RAW baraji {raw:.3f})")
    for name, (ratio, groups) in sorted(rows.items(), key=lambda kv: kv[1][0])[: args.ilk]:
        cells = " ".join(f"{g:6.3f}" if g is not None else "     -" for g in groups)
        print(f"{ratio:6.3f}  {cells}  {name[:90]}")


if __name__ == "__main__":
    main()
