#!/usr/bin/env python
"""Tum resmi benchmark sonuclarini, kapi olcumlerini ve ucus testlerini TEK panoda toplar.

Taranan yerler (tasinmaz, yerinde okunur):
    deneyler/**      yeni duzen: her deney modeller/ sonuclar/ kayitlar/ ile
    outputs/**       eski kosular (urgup_cevresi_models_4000, end_to_end_pipeline_*, ...)
Yazilan:
    deneyler/PANO.md     hizli bakis: protokol basina siralama ve RAW farki
    deneyler/PANO.xlsx   tum satirlar (resmi sonuclar, kapi, ucus, model konumlari)

Model kimligi dosyanin sha256'sidir: ayni model birden cok kosuda ya da kopyada
gorunse de tek satirdir; "konum" sutunu dosyanin BUGUN nerede durdugunu verir
(deneyler/secili_modeller > deneyler > models > models_pipeline > top_modeller).

Protokol: sonucu degistiren benchmark ayarlari (raster ciftleri, sorgu sayisi,
seed, ornekleme, arama ayarlari, normalizasyon). Yalniz ayni protokoldeki
sonuclar ve ayni protokolun RAW'i birbiriyle kiyaslanir.

    python sonuc_panosu.py            # deneyler/PANO.md + PANO.xlsx
    python sonuc_panosu.py --no-xlsx
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DENEYLER = ROOT / "deneyler"
SELECTED_DIR = DENEYLER / "secili_modeller"
RUN_ROOTS = (DENEYLER, ROOT / "outputs")
MODEL_ROOTS = (
    SELECTED_DIR,
    DENEYLER,
    ROOT / "models",
    ROOT / "models_pipeline",
    ROOT.parent / "template-matching" / "top_modeller",
)
MODEL_SUFFIXES = {".h5", ".keras", ".hdf5"}
RAW_MODEL_ID = "RAW_BASELINE"
PRUNE_DIRS = {
    "queries", "raw_tiles", "variants", ".summary_state", "cuda_cache",
    "_epochs", "keras", "__pycache__", "maps", "harita",
}
GLOBAL_MODE = "global"
TABLE_MODES = ("global", "roi_500m", "roi_2000m")
GROUP_ORDER = ("g2b_clean", "g2b_hard_v1", "b2g_clean", "b2g_hard_v1")
GROUP_TITLES = {
    "g2b_clean": "G2B temiz",
    "g2b_hard_v1": "G2B hard",
    "b2g_clean": "B2G temiz",
    "b2g_hard_v1": "B2G hard",
}
# Protokolu tanimlayan (sonucu degistiren) ayarlar. Arama modlari, varyantlar
# ve yon sayisi grup boyutlaridir; farkli mod kumeleri ayni protokolde olabilir.
PROTOCOL_FIELDS = (
    "max_queries", "seed", "query_sampling", "block_size_m", "samples_per_block",
    "min_query_std", "min_query_entropy", "max_dark_fraction", "query_edge_buffer_m",
    "tile_size", "overlap", "crop_border", "pyramid_factors", "top_k",
    "refine_radius_px", "nms_radius_px", "normalization", "enhancement",
    "output_value_mode",
)
FLIGHT_MODE = "benchmark_gps_merkezli"
FLIGHT_NOTES = {
    "outputs/karlik_top_modeller": "2026-09-12 koşusu: eski template-matching kodu ve model başına ön işleme "
    "sözleşmeleri; bugünkü F1/F2 ile kıyaslanmaz (bkz. deneyler/2026-09-25_genis_dogrulama/BENIOKU.md).",
}
HASH_CACHE = DENEYLER / ".pano_hash_cache.json"


# ----------------------------------------------------------------------------
# yardimcilar
# ----------------------------------------------------------------------------
def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def stem(path_text: str) -> str:
    return Path(str(path_text).replace("\\", "/")).stem


def relative(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def short_name(name: str, limit: int = 64) -> str:
    name = str(name)
    if len(name) <= limit:
        return name
    return name[: limit - 15] + "…" + name[-14:]


def walk_dirs(roots: Iterable[Path]) -> Iterable[Path]:
    """Kok altindaki klasorler; agir alt klasorler (sorgu karolari vb.) budanir."""
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirs, _files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
            yield Path(current)


# ----------------------------------------------------------------------------
# model dosya dizini (sha256 -> konumlar)
# ----------------------------------------------------------------------------
def file_sha256(path: Path, cache: dict[str, Any]) -> str:
    stat = path.stat()
    key = str(path.resolve())
    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
        return entry["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}
    return cache[key]["sha256"]


def build_model_index(roots: Iterable[Path] = MODEL_ROOTS, cache_path: Path | None = HASH_CACHE) -> dict[str, Any]:
    cache = read_json(cache_path, {}) if cache_path else {}
    by_sha: dict[str, list[str]] = {}
    by_name: dict[str, set[str]] = {}
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {"queries", "raw_tiles", "variants", ".summary_state", "__pycache__"}]
            for name in files:
                path = Path(current) / name
                if path.suffix.lower() not in MODEL_SUFFIXES:
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    sha = file_sha256(path, cache)
                except OSError:
                    continue
                by_sha.setdefault(sha, []).append(resolved)
                by_name.setdefault(name, set()).add(sha)
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
    cards = {}
    if SELECTED_DIR.is_dir():
        for card_path in SELECTED_DIR.glob("*.kart.json"):
            card = read_json(card_path, {})
            if card.get("sha256"):
                cards[card["sha256"]] = card
    return {"by_sha": by_sha, "by_name": by_name, "cards": cards}


def preferred_location(paths: list[str]) -> str | None:
    if not paths:
        return None

    def rank(path: str) -> tuple[int, str]:
        for index, root in enumerate(MODEL_ROOTS):
            try:
                Path(path).relative_to(root.resolve())
                return index, path
            except ValueError:
                continue
        return len(MODEL_ROOTS), path

    return relative(sorted(paths, key=rank)[0])


# ----------------------------------------------------------------------------
# resmi benchmark kosulari
# ----------------------------------------------------------------------------
def protocol_of(config: dict[str, Any]) -> tuple[str, str]:
    payload = {field: config.get(field) for field in PROTOCOL_FIELDS}
    payload["query_raster"] = stem(config.get("query_raster", ""))
    payload["map_raster"] = stem(config.get("map_raster", ""))
    payload["hard_v1_profile"] = (config.get("hard_v1_profile") or {}).get("profile_revision")
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    area = payload["query_raster"].split("_gmap")[0].split("_bing")[0] or "?"
    label = f"{area} {config.get('max_queries')} sorgu"
    if config.get("seed") != 42 or config.get("query_sampling") not in (None, "balanced_exact"):
        label += f" (seed {config.get('seed')}, {config.get('query_sampling')})"
    return key, label


def direction_labels(config: dict[str, Any]) -> dict[str, str]:
    query, search = stem(config.get("query_raster", "")), stem(config.get("map_raster", ""))
    forward = "g2b" if "gmap" in query or "google" in query else "ileri"
    reverse = "b2g" if forward == "g2b" else "geri"
    return {f"{query}__TO__{search}": forward, f"{search}__TO__{query}": reverse}


def complete_row(row: dict[str, Any], max_queries: int) -> bool:
    total = int(row.get("total_queries") or 0)
    accounted = sum(int(row.get(k) or 0) for k in ("ok_queries", "rejected_queries", "error_queries"))
    return total == max_queries and accounted == total


def read_benchmark_run(run_dir: Path) -> list[dict[str, Any]]:
    config = read_json(run_dir / "run_config.json", {})
    rows = read_json(run_dir / "summary.json", [])
    if not isinstance(rows, list) or not rows or "max_queries" not in config:
        return []
    if any(row.get("search_mode") == FLIGHT_MODE for row in rows):
        return []
    max_queries = int(config["max_queries"])
    protocol_key, protocol_label = protocol_of(config)
    labels = direction_labels(config)
    catalog = read_json(run_dir / "model_catalog.json", {}) or {}
    models = {m.get("model_id"): m for m in catalog.get("models", []) if isinstance(m, dict)}
    grouped: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for row in rows:
        if not complete_row(row, max_queries):
            continue
        direction = labels.get(str(row.get("direction")))
        if direction not in ("g2b", "b2g", "ileri", "geri"):
            continue
        direction = {"ileri": "g2b", "geri": "b2g"}.get(direction, direction)
        group = f"{direction}_{row.get('query_variant')}"
        grouped.setdefault(str(row.get("model_id")), {})[(group, str(row.get("search_mode")))] = row
    records = []
    for model_id, cells in grouped.items():
        entry = models.get(model_id, {})
        record: dict[str, Any] = {
            "protocol_key": protocol_key,
            "protocol": protocol_label,
            "max_queries": max_queries,
            "run_dir": relative(run_dir),
            "model_id": model_id,
            "is_raw": model_id == RAW_MODEL_ID,
            "sha256": entry.get("sha256"),
            "model_file": Path(str(entry.get("relative_path") or entry.get("path") or model_id)).name,
            "catalog_path": entry.get("path"),
            "completed_at": max(
                (str(r.get("result_completed_at_utc") or "") for r in cells.values()), default=""
            ),
        }
        for mode in sorted({m for _, m in cells}):
            values = [cells.get((g, mode), {}).get("success_25m") for g in GROUP_ORDER]
            if all(v is not None for v in values):
                record[mode] = sum(float(v) for v in values) / len(values)
            if mode == GLOBAL_MODE:
                for group, value in zip(GROUP_ORDER, values):
                    record[group] = None if value is None else float(value)
        record["complete_groups"] = sum(1 for g in GROUP_ORDER if (g, GLOBAL_MODE) in cells)
        records.append(record)
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ayni protokolde ayni model (sha256; RAW icin kimlik) tek satir olur."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        identity = RAW_MODEL_ID if record["is_raw"] else (record.get("sha256") or record["model_id"])
        key = (record["protocol_key"], identity)
        current = best.get(key)
        rank = (
            record.get("complete_groups", 0),
            record.get(GLOBAL_MODE) is not None,
            record["run_dir"].startswith("deneyler/"),
            record.get("completed_at", ""),
        )
        if current is None or rank > current["_rank"]:
            best[key] = dict(record, _rank=rank)
    result = []
    for record in best.values():
        record = dict(record)
        record.pop("_rank", None)
        result.append(record)
    return result


def attach_raw_deltas(records: list[dict[str, Any]]) -> None:
    raw = {
        r["protocol_key"]: r
        for r in records
        if r["is_raw"] and r.get(GLOBAL_MODE) is not None
    }
    for record in records:
        base = raw.get(record["protocol_key"])
        record["raw_global"] = base.get(GLOBAL_MODE) if base else None
        if base and record.get(GLOBAL_MODE) is not None and not record["is_raw"]:
            record["delta_raw"] = record[GLOBAL_MODE] - base[GLOBAL_MODE]
        else:
            record["delta_raw"] = None


# ----------------------------------------------------------------------------
# kapi (margin_gate.py JSON) ve ucus testi
# ----------------------------------------------------------------------------
def gate_label(payload: dict[str, Any]) -> str:
    groups = payload.get("groups") or ["g2b_clean"]
    kind = "4 grup" if len(groups) == 4 else "+".join(groups)
    return f"kapı {payload.get('samples')} örnek, {kind}, {payload.get('aoi')}"


def read_gate_files(index: dict[str, Any], roots: Iterable[Path] = RUN_ROOTS) -> list[dict[str, Any]]:
    records = []
    for directory in walk_dirs(roots):
        for path in directory.glob("*.json"):
            if path.stat().st_size > 20_000_000:
                continue
            payload = read_json(path, None)
            if not isinstance(payload, dict) or payload.get("primary_metric") != "false_peak_ratio":
                continue
            label = gate_label(payload)
            raw_ratio = payload.get("raw_false_peak_ratio")
            for name, score in (payload.get("results") or {}).items():
                if name == "RAW" or not isinstance(score, dict) or "false_peak_ratio" not in score:
                    continue
                sha = score.get("model_sha256")
                if not sha:
                    candidates = index["by_name"].get(Path(name).name, set())
                    sha = next(iter(candidates)) if len(candidates) == 1 else None
                groups = score.get("groups") or {}
                records.append({
                    "gate": label,
                    "file": relative(path),
                    "model_file": Path(name).name,
                    "sha256": sha,
                    "ratio": float(score["false_peak_ratio"]),
                    "raw_ratio": raw_ratio,
                    "mtime": path.stat().st_mtime,
                    **{f"kapi_{g}": groups[g]["false_peak_ratio"] for g in groups if "false_peak_ratio" in groups[g]},
                })
    # Ayni kapida ayni model (deterministik) tek satir: en yeni dosya.
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["gate"], record["sha256"] or record["model_file"])
        if key not in best or record["mtime"] > best[key]["mtime"]:
            best[key] = record
    return sorted(best.values(), key=lambda r: (r["gate"], r["ratio"]))


def read_flight_json(path: Path) -> dict[str, Any] | None:
    """deney_araclari/ucus_testi.py test ciktisi (tek model, kare basina hata)."""
    payload = read_json(path, None)
    if not isinstance(payload, dict) or "kareler" not in payload or "success_70m" not in payload:
        return None
    if not payload["kareler"]:
        return None  # TM hic kare islememis (ör. kare patch'ten kucuk); olcum yok
    if int(payload.get("max_frames") or 0) > 0:
        return None  # kisa deneme kosusu (ilk N kare), tam test degil
    return {
        "test": f"{Path(str(payload.get('anlik', ''))).name} · {payload.get('protokol')}",
        "run_dir": relative(path),
        "model_file": payload.get("model"),
        "sha256": payload.get("model_sha256"),
        "frames": payload.get("kare"),
        "success_70m": payload.get("success_70m"),
        "success_50m": payload.get("success_50m"),
        "success_25m": payload.get("success_25m"),
        "median_error_m": payload.get("medyan_m"),
        "no_intersection": payload.get("kesisim_yok_kare"),
    }


def read_flight_runs(roots: Iterable[Path] = RUN_ROOTS) -> list[dict[str, Any]]:
    records = []
    for directory in walk_dirs(roots):
        if (directory.name.startswith("karlik_") or directory.parent.name == "ucus") and not directory.name.startswith("_"):
            for path in directory.glob("*.json"):
                record = read_flight_json(path)
                if record:
                    records.append(record)
        summary = directory / "summary.json"
        if not summary.is_file():
            continue
        rows = read_json(summary, [])
        if not isinstance(rows, list) or not any(r.get("search_mode") == FLIGHT_MODE for r in rows):
            continue
        config = read_json(directory / "run_config.json", {}) or {}
        catalog = read_json(directory / "model_catalog.json", {}) or {}
        shas = {k: v.get("model_sha256") for k, v in catalog.items() if isinstance(v, dict)}
        test = f"{config.get('harita_kaynagi', directory.name)} · {Path(str(config.get('anlik_dir', ''))).name}"
        for row in rows:
            if row.get("search_mode") != FLIGHT_MODE:
                continue
            records.append({
                "test": test,
                "run_dir": relative(directory),
                "model_file": row.get("model_label") or row.get("model_id"),
                "sha256": shas.get(row.get("model_id")),
                "frames": row.get("total_queries"),
                "success_70m": row.get("success_70m"),
                "success_50m": row.get("success_50m"),
                "success_25m": row.get("success_25m"),
                "median_error_m": row.get("median_error_m"),
            })
    return sorted(records, key=lambda r: (r["test"], -(r["success_70m"] or 0)))


# ----------------------------------------------------------------------------
# pano
# ----------------------------------------------------------------------------
def collect(index: dict[str, Any] | None = None) -> dict[str, Any]:
    index = index or build_model_index()
    runs = [d for d in walk_dirs(RUN_ROOTS) if (d / "summary.json").is_file() and (d / "run_config.json").is_file()]
    records = dedupe_records([r for run in runs for r in read_benchmark_run(run)])
    attach_raw_deltas(records)
    gates = read_gate_files(index)
    flights = read_flight_runs()
    decorate(records + gates + flights, index)
    best_gate: dict[str, dict[str, float]] = {}
    for gate in gates:
        if gate["sha256"]:
            best_gate.setdefault(gate["sha256"], {})[gate["gate"]] = gate["ratio"]
    for record in records:
        gate_scores = best_gate.get(record.get("sha256") or "", {})
        four = [v for k, v in gate_scores.items() if "4 grup" in k and "150" in k]
        g2b = [v for k, v in gate_scores.items() if "g2b_clean" in k and "150" in k]
        record["kapi4"] = four[0] if four else None
        record["kapi_g2b"] = g2b[0] if g2b else None
    return {"records": records, "gates": gates, "flights": flights, "runs": runs, "index": index}


def decorate(records: list[dict[str, Any]], index: dict[str, Any]) -> None:
    for record in records:
        sha = record.get("sha256")
        card = index["cards"].get(sha) if sha else None
        if record.get("is_raw"):
            record["name"] = "RAW (model yok)"
            record["location"] = "—"
            continue
        record["name"] = card.get("kisa_ad") if card else short_name(Path(str(record.get("model_file"))).stem)
        record["card"] = card.get("aciklama", "") if card else ""
        location = preferred_location(index["by_sha"].get(sha, [])) if sha else None
        record["location"] = location or f"(dosya bulunamadı) {record.get('catalog_path') or ''}".strip()


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def protocol_order(records: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    seen: dict[str, tuple[str, int]] = {}
    for record in records:
        seen.setdefault(record["protocol_key"], (record["protocol"], 0))
        label, count = seen[record["protocol_key"]]
        seen[record["protocol_key"]] = (label, count + 1)
    # Once Urgup 4000 (2022 protokolu), sonra 1000, sonra digerleri kalabaliga gore.
    def rank(item):
        key, (label, count) = item
        main = 0 if label.startswith("urgup_cevresi 4000") else 1 if label.startswith("urgup_cevresi 1000") else 2
        return (main, -count, label)

    ordered = [(key, label, count) for key, (label, count) in sorted(seen.items(), key=rank)]
    # Ayni etiketli farkli protokoller (or. ayni sorgu sayisi, farkli arama ayari) ayirt edilsin.
    labels = [label for _, label, _ in ordered]
    return [
        (key, f"{label} · protokol {key[:6]}" if labels.count(label) > 1 else label, count)
        for key, label, count in ordered
    ]


def render_markdown(data: dict[str, Any], *, top: int = 25) -> str:
    records = data["records"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Sonuç panosu",
        "",
        f"Otomatik üretildi: `python sonuc_panosu.py` ({now}). Elle düzenlemeyin; "
        "ayrıntılı tüm satırlar `PANO.xlsx` içindedir.",
        "",
        "Ölçüt: global arama `success_25m`, dört grubun ortalaması (G2B/B2G × temiz/hard_v1). "
        "Δ RAW: aynı protokolün (aynı sorgu kümesi) RAW'ına göre fark. "
        "Kapı: `margin_gate.py` yanlış tepe oranı (150 örnek, eğitim bölgesi; küçük iyi).",
        "",
    ]
    for key, label, _count in protocol_order(records):
        rows = [r for r in records if r["protocol_key"] == key and r.get(GLOBAL_MODE) is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: -r[GLOBAL_MODE])
        raw = next((r for r in rows if r["is_raw"]), None)
        main = label.startswith("urgup_cevresi 4000") or label.startswith("urgup_cevresi 1000")
        limit = top if main else 8
        shown = rows[:limit]
        if raw and raw not in shown:
            shown.append(raw)
        lines += [f"## {label}", ""]
        if raw:
            lines += [f"RAW global: **{raw[GLOBAL_MODE]:.3f}** · {len(rows)} sonuç", ""]
        lines += [
            "| # | Model | Global | Δ RAW | G2B temiz | G2B hard | B2G temiz | B2G hard | roi 500 m | roi 2000 m | Kapı 4 grup | Konum |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for rank, record in enumerate(shown, 1):
            name = f"**{record['name']}**" if record["is_raw"] else record["name"]
            lines.append(
                "| {rank} | {name} | {g} | {d} | {a} | {b} | {c} | {e} | {r5} | {r2} | {k} | {loc} |".format(
                    rank=rank if record in rows[:limit] else "",
                    name=name.replace("|", "/"),
                    g=fmt(record.get(GLOBAL_MODE)),
                    d=fmt(record.get("delta_raw"), signed=True),
                    a=fmt(record.get("g2b_clean")),
                    b=fmt(record.get("g2b_hard_v1")),
                    c=fmt(record.get("b2g_clean")),
                    e=fmt(record.get("b2g_hard_v1")),
                    r5=fmt(record.get("roi_500m")),
                    r2=fmt(record.get("roi_2000m")),
                    k=fmt(record.get("kapi4")),
                    loc=f"`{record.get('location')}`" if record.get("location") not in (None, "—") else "—",
                )
            )
        if len(rows) > limit:
            lines.append(f"\n… ve {len(rows) - limit} sonuç daha (PANO.xlsx → Resmi).")
        lines.append("")
    gates = [g for g in data["gates"] if "4 grup" in g["gate"]]
    if gates:
        by_gate: dict[str, list[dict[str, Any]]] = {}
        for gate in gates:
            by_gate.setdefault(gate["gate"], []).append(gate)
        for label, rows in by_gate.items():
            rows.sort(key=lambda r: r["ratio"])
            raw_ratio = rows[0].get("raw_ratio")
            lines += [f"## Kapı: {label} (en iyi {min(top, len(rows))} / {len(rows)})", ""]
            if raw_ratio is not None:
                lines += [f"RAW barajı: **{float(raw_ratio):.3f}** (küçük olan iyi)", ""]
            lines += [
                "| # | Model | Oran | G2B temiz | G2B hard | B2G temiz | B2G hard | Konum |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
            for rank, g in enumerate(rows[:top], 1):
                lines.append(
                    f"| {rank} | {g['name'].replace('|', '/')} | {g['ratio']:.3f} | "
                    f"{fmt(g.get('kapi_g2b_clean'))} | {fmt(g.get('kapi_g2b_hard_v1'))} | "
                    f"{fmt(g.get('kapi_b2g_clean'))} | {fmt(g.get('kapi_b2g_hard_v1'))} | `{g.get('location')}` |"
                )
            lines.append("")
    flights = data["flights"]
    if flights:
        for test in sorted({f["test"] for f in flights}):
            rows = [f for f in flights if f["test"] == test]
            lines += [f"## Uçuş testi: {test} ({rows[0]['frames']} kare, GPS merkezli)", ""]
            note = FLIGHT_NOTES.get(rows[0]["run_dir"].split("/")[0] + "/" + rows[0]["run_dir"].split("/")[1]
                                    if rows[0]["run_dir"].count("/") else rows[0]["run_dir"])
            if note:
                lines += [f"> {note}", ""]
            lines += [
                "| # | Model | ≤70 m | ≤50 m | ≤25 m | Medyan hata (m) | Kesişimsiz kare |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
            for rank, f in enumerate(rows[:top], 1):
                lines.append(
                    f"| {rank} | {f['name'].replace('|', '/')} | {fmt(f['success_70m'])} | "
                    f"{fmt(f['success_50m'])} | {fmt(f['success_25m'])} | {fmt(f['median_error_m'], 1)} | "
                    f"{fmt(f.get('no_intersection'))} |"
                )
            lines.append("")
    lines += [
        "## Nerede ne var",
        "",
        "- `deneyler/<deney>/modeller/` eğitilen modeller, `sonuclar/` benchmark ve kapı çıktıları, "
        "`kayitlar/` günlükler; her deneyin `BENIOKU.md` dosyası amacını ve bulgularını anlatır.",
        "- `deneyler/pipeline_<run_id>/` `end_to_end_pipeline.py` koşuları (aynı düzen).",
        "- `deneyler/secili_modeller/` öne çıkan modellerin kopyaları ve `*.kart.json` kartları.",
        "- Eski düzen (yerinde okunur): `models_pipeline/<run_id>/`, `outputs/end_to_end_pipeline_<run_id>/`, "
        "`outputs/urgup_cevresi_models_4000/`, `models/`.",
        "",
        f"Taranan resmi koşu klasörü: {len(data['runs'])}",
        "",
    ]
    return "\n".join(lines)


def write_xlsx(data: dict[str, Any], path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheets = [
        ("Resmi", data["records"], [
            ("protocol", "Protokol"), ("name", "Model"), (GLOBAL_MODE, "Global"),
            ("delta_raw", "Δ RAW"), ("raw_global", "RAW global"),
            ("g2b_clean", "G2B temiz"), ("g2b_hard_v1", "G2B hard"),
            ("b2g_clean", "B2G temiz"), ("b2g_hard_v1", "B2G hard"),
            ("roi_500m", "roi 500 m"), ("roi_1000m", "roi 1000 m"), ("roi_2000m", "roi 2000 m"),
            ("roi_4000m", "roi 4000 m"), ("roi_8000m", "roi 8000 m"),
            ("kapi4", "Kapı 4 grup"), ("kapi_g2b", "Kapı G2B"),
            ("location", "Konum"), ("model_file", "Dosya adı"), ("run_dir", "Sonuç klasörü"),
            ("completed_at", "Tamamlanma (UTC)"), ("sha256", "sha256"),
        ]),
        ("Kapi", data["gates"], [
            ("gate", "Kapı"), ("name", "Model"), ("ratio", "Oran"), ("raw_ratio", "RAW"),
            ("kapi_g2b_clean", "G2B temiz"), ("kapi_g2b_hard_v1", "G2B hard"),
            ("kapi_b2g_clean", "B2G temiz"), ("kapi_b2g_hard_v1", "B2G hard"),
            ("location", "Konum"), ("model_file", "Dosya adı"), ("file", "Ölçüm dosyası"), ("sha256", "sha256"),
        ]),
        ("Ucus", data["flights"], [
            ("test", "Test"), ("name", "Model"), ("success_70m", "≤70 m"), ("success_50m", "≤50 m"),
            ("success_25m", "≤25 m"), ("median_error_m", "Medyan hata (m)"), ("frames", "Kare"),
            ("no_intersection", "Kesişimsiz kare"),
            ("location", "Konum"), ("run_dir", "Sonuç klasörü"), ("sha256", "sha256"),
        ]),
    ]
    first = True
    for title, rows, columns in sheets:
        sheet = workbook.active if first else workbook.create_sheet()
        first = False
        sheet.title = title
        sheet.append([header for _, header in columns])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        ordered = sorted(
            rows,
            key=lambda r: (str(r.get("protocol") or r.get("gate") or r.get("test") or ""),
                           -(r.get(GLOBAL_MODE) or 0) if title == "Resmi" else
                           (r.get("ratio") or 9) if title == "Kapi" else -(r.get("success_70m") or 0)),
        )
        for row in ordered:
            sheet.append([row.get(key) for key, _ in columns])
        sheet.freeze_panes = "C2"
        for index, (key, header) in enumerate(columns, 1):
            width = 60 if key in ("location", "run_dir", "file", "model_file") else 42 if key == "name" else 12
            sheet.column_dimensions[get_column_letter(index)].width = max(width, len(header) + 2)
    models = workbook.create_sheet("Modeller")
    models.append(["sha256", "Kısa ad", "Açıklama", "Konumlar"])
    for cell in models[1]:
        cell.font = Font(bold=True)
    for sha, paths in sorted(data["index"]["by_sha"].items(), key=lambda kv: kv[1][0]):
        card = data["index"]["cards"].get(sha, {})
        models.append([sha, card.get("kisa_ad", ""), card.get("aciklama", ""), " | ".join(relative(p) for p in paths)])
    models.column_dimensions["A"].width = 20
    models.column_dimensions["B"].width = 30
    models.column_dimensions["D"].width = 120
    tmp = path.with_name(path.stem + ".tmp.xlsx")
    workbook.save(tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tum sonuclari tek panoda toplar (deneyler/PANO.md, PANO.xlsx).")
    parser.add_argument("--out-dir", type=Path, default=DENEYLER)
    parser.add_argument("--no-xlsx", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)
    data = collect()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.out_dir / "PANO.md"
    markdown_path.write_text(render_markdown(data, top=args.top), encoding="utf-8")
    print(f"Pano: {markdown_path}  ({len(data['records'])} resmi sonuç, {len(data['gates'])} kapı ölçümü, {len(data['flights'])} uçuş satırı)")
    if not args.no_xlsx:
        try:
            write_xlsx(data, args.out_dir / "PANO.xlsx")
            print(f"Excel: {args.out_dir / 'PANO.xlsx'}")
        except ImportError:
            print("UYARI: openpyxl yok; PANO.xlsx yazilmadi.", file=sys.stderr)
        except PermissionError:
            print("UYARI: PANO.xlsx acik/kilitli; yazilamadi (Excel'i kapatip tekrar calistirin).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
