#!/usr/bin/env python
"""Bir modeli deneyler/secili_modeller/ rafina kisa adla kopyalar ve kartini yazar.

    python rafa_ekle.py <model.h5> --ad yeni_topluluk3 --aciklama "..." [--tarif "..."] [--kaynak-deney ...]

Raf, "hangi modeli kullanmaliyim" sorusunun cevabidir: en iyi yeni modeller ve
kiyas icin 2022 kazananlari tek yerde. Kart (<ad>.kart.json) sonuc_panosu.py
tarafindan okunur; panoda uzun dosya adi yerine kisa ad gorunur. Ayni icerik
(sha256) zaten raftaysa yeniden kopyalanmaz, yalniz kart guncellenir.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

RAF = Path(__file__).resolve().parents[1] / "deneyler" / "secili_modeller"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ad", required=True, help="kisa ad (dosya adi olur)")
    ap.add_argument("--aciklama", required=True)
    ap.add_argument("--tarif", default="")
    ap.add_argument("--kaynak-deney", default="")
    args = ap.parse_args()
    source = Path(args.model).resolve()
    RAF.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    target = RAF / f"{args.ad}{source.suffix}"
    if target.exists() and sha256(target) != digest:
        raise SystemExit(f"{target.name} rafta farkli icerikle var; baska bir ad secin")
    if not target.exists():
        shutil.copy2(source, target)
    card = {
        "kisa_ad": args.ad,
        "aciklama": args.aciklama,
        "tarif": args.tarif,
        "kaynak_deney": args.kaynak_deney,
        "kaynak_dosya": str(source),
        "raf_dosyasi": target.name,
        "sha256": digest,
        "rafa_eklenme": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (RAF / f"{args.ad}.kart.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{target.name}  sha256={digest[:12]}")


if __name__ == "__main__":
    main()
