#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproducible geospatial benchmark for shared neural image representations.

The benchmark applies the *same* sat-to-map model to both sides:

    model(query tile) -> search inside model(reference raster)

It preserves a RAW baseline, samples query centres in projected coordinates,
uses deterministic spatially stratified sampling, performs multi-candidate
coarse-to-fine normalized cross correlation, checkpoints every query, and
writes machine-readable CSV/JSONL outputs.  It refreshes the Excel workbook at
model boundaries and performs a strict final Excel export at the end.

The expensive map production stages reuse the repository-local
``goruntu_islemleri.py`` copy so the benchmark code is self-contained.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.transform import rowcol
from rasterio.windows import Window


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY = SCRIPT_DIR / "urgup_30_cm_yeni_gmaps_utm.tif"
DEFAULT_MAP = SCRIPT_DIR / "urgup_bingmap_utm_30_cm.tif"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
RAW_MODEL_ID = "RAW_BASELINE"

DEFAULT_SEARCH_MODE_ORDER = {
    "roi_500m": 0,
    "roi_1000m": 1,
    "roi_2000m": 2,
    "roi_4000m": 3,
    "roi_8000m": 4,
    "global": 5,
}
QUERY_VARIANT_ORDER = {"clean": 0, "hard_v1": 1}
HARD_V1_PROFILE = {
    "profile_revision": "uav_camera_v1",
    "scenarios": {
        "clear_light": 0.20,
        "haze": 0.25,
        "motion_blur": 0.20,
        "defocus_blur": 0.15,
        "low_contrast": 0.10,
        "compression_noise": 0.10,
    },
    "gain": [0.82, 1.18],
    "bias": [-14.0, 14.0],
    "gamma": [0.82, 1.22],
    "white_balance_gain": [0.90, 1.10],
    "saturation": [0.78, 1.22],
    "haze_alpha": [0.12, 0.32],
    "haze_airlight": [225.0, 255.0],
    "motion_blur_length_px": [5, 17],
    "defocus_sigma": [1.00, 2.60],
    "low_contrast_factor": [0.55, 0.80],
    "noise_std": [0.8, 4.0],
    "compression_noise_std": [5.0, 10.0],
    "jpeg_quality": [65, 92],
    "compression_jpeg_quality": [35, 60],
    "vignette_probability": 0.18,
    "vignette_strength": [0.08, 0.22],
}

LOG = logging.getLogger("geospatial_benchmark")
_AUTO_EXCEL_ENGINE: str | None = None


def result_group_sort_key(
    item: tuple[tuple[str, str, str, str], Any],
) -> tuple[str, int, str, int, float, str]:
    """Keep summaries in model order and ROI-small-to-global order."""
    (direction, query_variant, search_mode, model_id), _ = item
    explicit_order = DEFAULT_SEARCH_MODE_ORDER.get(search_mode)
    match = re.fullmatch(r"roi_(\d+(?:\.\d+)?)m", search_mode)
    numeric_radius = float(match.group(1)) if match else math.inf
    mode_rank = explicit_order if explicit_order is not None else 4
    return (
        direction,
        QUERY_VARIANT_ORDER.get(query_variant, 99),
        model_id,
        mode_rank,
        numeric_radius,
        search_mode,
    )


RESULT_COLUMNS = [
    "run_id",
    "direction",
    "query_variant",
    "search_mode",
    "roi_radius_m",
    "model_id",
    "model_file",
    "model_sha256",
    "query_id",
    "block_id",
    "query_center_easting_m",
    "query_center_northing_m",
    "expected_center_easting_m",
    "expected_center_northing_m",
    "predicted_center_easting_m",
    "predicted_center_northing_m",
    "error_m",
    "error_px",
    "top1_score",
    "top2_score",
    "peak_margin",
    "psr",
    "success_5m",
    "success_10m",
    "success_25m",
    "success_30m",
    "success_50m",
    "status",
    "reason",
    "query_std",
    "query_entropy",
    "query_dark_fraction",
    "search_seconds",
    "query_inference_seconds",
    "map_build_seconds",
    "pyramid_factors",
    "top_k",
    "normalization",
    "template_size_px",
    "source_query_raster",
    "source_map_raster",
    "created_at_utc",
]


MANIFEST_COLUMNS = [
    "query_id",
    "block_id",
    "center_easting_m",
    "center_northing_m",
    "source_row",
    "source_col",
    "query_std",
    "query_entropy",
    "dark_fraction",
    "raw_tile_file",
]


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    block_id: str
    center_easting_m: float
    center_northing_m: float
    source_row: int
    source_col: int
    query_std: float
    query_entropy: float
    dark_fraction: float
    raw_tile_file: str


@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class SearchOutcome:
    x: int
    y: int
    top1_score: float
    top2_score: float
    peak_margin: float
    psr: float


@dataclass(frozen=True)
class PreparedSearchMap:
    gray: np.ndarray
    transform: Affine
    pyramid: dict[int, np.ndarray] | None


class StageTimer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started = 0.0

    def __enter__(self) -> "StageTimer":
        self.started = time.perf_counter()
        LOG.info("BAŞLADI | %s", self.label)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        elapsed = time.perf_counter() - self.started
        if exc_type is None:
            LOG.info("TAMAMLANDI | %s | süre=%.2f sn", self.label, elapsed)
        else:
            LOG.error("BAŞARISIZ | %s | süre=%.2f sn | hata=%s", self.label, elapsed, exc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str, max_len: int = 120) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    cleaned = cleaned.strip("._") or "item"
    return cleaned[:max_len]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def configure_logging(run_dir: Path, verbose: bool) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "benchmark.log"
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOG.addHandler(console)
    LOG.addHandler(file_handler)
    return log_path


def parse_int_list(value: str) -> tuple[int, ...]:
    # PowerShell/conda run may reconstruct comma-separated CLI values with
    # spaces. Accept both representations so documented commands stay robust.
    items = tuple(int(part) for part in re.split(r"[,\s]+", value.strip()) if part)
    if not items or any(item < 1 for item in items):
        raise argparse.ArgumentTypeError("Pozitif tam sayılardan oluşan liste bekleniyor.")
    if tuple(sorted(items, reverse=True)) != items or items[-1] != 1:
        raise argparse.ArgumentTypeError("Piramit faktörleri büyükten küçüğe inmeli ve 1 ile bitmeli.")
    return items


def parse_patterns(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ("*.h5", "*.keras")
    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(patterns)


def parse_search_modes(value: str) -> tuple[tuple[str, float | None], ...]:
    tokens = [part.lower() for part in re.split(r"[,\s]+", value.strip()) if part]
    if not tokens:
        raise argparse.ArgumentTypeError("En az bir arama modu verilmelidir.")
    modes: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for token in tokens:
        if token == "global":
            name, radius = "global", None
        elif token.startswith("roi"):
            raw_radius = token[3:].lstrip("_:-")
            try:
                radius = float(raw_radius)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"ROI modu 'roi500' biçiminde olmalıdır: {token}"
                ) from exc
            if radius <= 0:
                raise argparse.ArgumentTypeError(f"ROI yarıçapı pozitif olmalıdır: {token}")
            name = f"roi_{radius:g}m"
        else:
            raise argparse.ArgumentTypeError(f"Bilinmeyen arama modu: {token}")
        if name not in seen:
            modes.append((name, radius))
            seen.add(name)
    return tuple(modes)


def parse_query_variants(value: str) -> tuple[str, ...]:
    aliases = {
        "clean": "clean",
        "temiz": "clean",
        "hard": "hard_v1",
        "hard_v1": "hard_v1",
        "zor": "hard_v1",
    }
    variants: list[str] = []
    for token in value.split(","):
        normalized = aliases.get(token.strip().lower())
        if normalized is None:
            raise argparse.ArgumentTypeError(
                f"Bilinmeyen sorgu varyantı: {token}. Geçerli değerler: clean, hard_v1"
            )
        if normalized not in variants:
            variants.append(normalized)
    if not variants:
        raise argparse.ArgumentTypeError("En az bir sorgu varyantı seçilmelidir.")
    return tuple(variants)


def select_models(model_dir: Path, patterns: Sequence[str], max_models: int | None) -> list[Path]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model klasörü bulunamadı: {model_dir}")
    files = [
        path
        for path in sorted(model_dir.iterdir(), key=lambda p: p.name.lower())
        if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
    ]
    if max_models is not None:
        files = files[:max_models]
    if not files:
        raise FileNotFoundError(
            f"Model bulunamadı: dir={model_dir}, patterns={','.join(patterns)}"
        )
    return files


def intersection_bounds(a: BoundingBox, b: BoundingBox) -> BoundingBox:
    result = BoundingBox(
        left=max(a.left, b.left),
        bottom=max(a.bottom, b.bottom),
        right=min(a.right, b.right),
        top=min(a.top, b.top),
    )
    if result.left >= result.right or result.bottom >= result.top:
        raise ValueError("Rasterların ortak coğrafi kesişim alanı yok.")
    return result


def validate_rasters(query_path: Path, map_path: Path) -> dict[str, Any]:
    for path in (query_path, map_path):
        if not path.is_file():
            raise FileNotFoundError(f"Raster bulunamadı: {path}")
    with rasterio.open(query_path) as query_ds, rasterio.open(map_path) as map_ds:
        if query_ds.crs is None or map_ds.crs is None:
            raise ValueError("Her iki raster da CRS taşımalıdır.")
        if query_ds.crs != map_ds.crs:
            raise ValueError(
                f"CRS uyuşmuyor: query={query_ds.crs}, map={map_ds.crs}. "
                "Benchmark öncesi açık bir yeniden projeksiyon adımı gerekir."
            )
        common = intersection_bounds(query_ds.bounds, map_ds.bounds)
        return {
            "crs": str(query_ds.crs),
            "common_bounds": list(common),
            "query_shape": [query_ds.height, query_ds.width],
            "map_shape": [map_ds.height, map_ds.width],
            "query_resolution": [abs(query_ds.transform.a), abs(query_ds.transform.e)],
            "map_resolution": [abs(map_ds.transform.a), abs(map_ds.transform.e)],
            "query_transform": list(query_ds.transform)[:6],
            "map_transform": list(map_ds.transform)[:6],
        }


def read_rgb_window(dataset: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    if dataset.count >= 3:
        band_first = dataset.read([1, 2, 3], window=window)
        return np.moveaxis(band_first, 0, -1)
    single = dataset.read(1, window=window)
    return cv2.cvtColor(single, cv2.COLOR_GRAY2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.uint8, copy=False)
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)


def entropy_u8(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probability = hist[hist > 0] / total
    return float(-(probability * np.log2(probability)).sum())


def tile_quality(gray: np.ndarray) -> tuple[float, float, float]:
    std = float(np.std(gray))
    entropy = entropy_u8(gray)
    dark_fraction = float(np.mean(gray <= 5))
    return std, entropy, dark_fraction


def write_png(path: Path, rgb_or_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_or_gray.ndim == 3:
        data = cv2.cvtColor(rgb_or_gray[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        data = rgb_or_gray
    if not cv2.imwrite(str(path), data):
        raise OSError(f"PNG yazılamadı: {path}")


def csv_write(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_append(path: Path, row: dict[str, Any], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Geçersiz JSONL: {path}:{line_number}: {exc}") from exc
    return rows


def generate_query_manifest(
    query_raster: Path,
    map_raster: Path,
    output_dir: Path,
    *,
    tile_size: int,
    samples_per_block: int,
    block_size_m: float,
    max_queries: int | None,
    seed: int,
    min_std: float,
    min_entropy: float,
    max_dark_fraction: float,
    edge_buffer_m: float | None,
    force: bool,
) -> list[QueryRecord]:
    manifest_json = output_dir / "query_manifest.json"
    manifest_csv = output_dir / "query_manifest.csv"
    raw_dir = output_dir / "raw_tiles"
    with rasterio.open(query_raster) as query_ds, rasterio.open(map_raster) as map_ds:
        max_pixel_size_m = max(
            abs(query_ds.transform.a),
            abs(query_ds.transform.e),
            abs(map_ds.transform.a),
            abs(map_ds.transform.e),
        )
    effective_edge_buffer_m = (
        float(edge_buffer_m) if edge_buffer_m is not None else tile_size * max_pixel_size_m
    )
    selection_config = {
        "query_raster": str(query_raster.resolve()),
        "map_raster": str(map_raster.resolve()),
        "tile_size": tile_size,
        "samples_per_block": samples_per_block,
        "block_size_m": block_size_m,
        "max_queries": max_queries,
        "seed": seed,
        "min_std": min_std,
        "min_entropy": min_entropy,
        "max_dark_fraction": max_dark_fraction,
        "effective_edge_buffer_m": effective_edge_buffer_m,
    }
    if manifest_json.exists() and not force:
        payload = json.loads(manifest_json.read_text(encoding="utf-8"))
        previous_edge_buffer_m = payload.get("effective_edge_buffer_m")
        edge_buffer_matches = previous_edge_buffer_m is not None and math.isclose(
            float(previous_edge_buffer_m), effective_edge_buffer_m, rel_tol=0.0, abs_tol=1e-6
        )
        selection_matches = payload.get("selection_config") == selection_config
        if edge_buffer_matches and selection_matches:
            records = [QueryRecord(**row) for row in payload["queries"]]
            missing = [
                record.raw_tile_file
                for record in records
                if not Path(record.raw_tile_file).is_file()
            ]
            if not missing:
                LOG.info(
                    "Sorgu manifesti yeniden kullanılıyor: %s (%d sorgu)",
                    manifest_json,
                    len(records),
                )
                return records
            LOG.warning(
                "Manifestte %d ham sorgu dosyası eksik; manifest yeniden üretilecek.",
                len(missing),
            )
        elif not edge_buffer_matches:
            LOG.info(
                "Manifest kenar tamponu değişti (eski=%s, yeni=%.2f m); yeniden üretilecek.",
                previous_edge_buffer_m,
                effective_edge_buffer_m,
            )
        else:
            LOG.info("Sorgu örnekleme ayarları değişti; manifest yeniden üretilecek.")

    if force and output_dir.exists():
        LOG.info("Sorgu manifesti --force nedeniyle yeniden üretilecek: %s", output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_dir.glob("Q*.png"):
        old.unlink()

    rng = np.random.default_rng(seed)
    records: list[QueryRecord] = []
    with rasterio.open(query_raster) as query_ds, rasterio.open(map_raster) as map_ds:
        common = intersection_bounds(query_ds.bounds, map_ds.bounds)
        margin_x = (
            (tile_size / 2.0 + 2.0)
            * max(abs(query_ds.transform.a), abs(map_ds.transform.a))
            + effective_edge_buffer_m
        )
        margin_y = (
            (tile_size / 2.0 + 2.0)
            * max(abs(query_ds.transform.e), abs(map_ds.transform.e))
            + effective_edge_buffer_m
        )
        left, right = common.left + margin_x, common.right - margin_x
        bottom, top = common.bottom + margin_y, common.top - margin_y
        if left >= right or bottom >= top:
            raise ValueError("Ortak alan sorgu karo boyutu için yetersiz.")

        block_cols = max(1, int(math.ceil((right - left) / block_size_m)))
        block_rows = max(1, int(math.ceil((top - bottom) / block_size_m)))
        block_order = [(row, col) for row in range(block_rows) for col in range(block_cols)]
        rng.shuffle(block_order)
        LOG.info(
            "Sorgu örnekleme | kullanılabilir alan=%.2f km² | kenar tamponu=%.2f m | "
            "blok=%dx%d | blok boyutu=%.0f m | seed=%d",
            ((right - left) * (top - bottom)) / 1_000_000.0,
            effective_edge_buffer_m,
            block_rows,
            block_cols,
            block_size_m,
            seed,
        )

        for block_row, block_col in block_order:
            bx0 = left + block_col * block_size_m
            bx1 = min(right, bx0 + block_size_m)
            by1 = top - block_row * block_size_m
            by0 = max(bottom, by1 - block_size_m)
            accepted_in_block = 0
            attempts = 0
            max_attempts = max(20, samples_per_block * 30)
            while accepted_in_block < samples_per_block and attempts < max_attempts:
                attempts += 1
                x = float(rng.uniform(bx0, bx1))
                y = float(rng.uniform(by0, by1))
                row, col = rowcol(query_ds.transform, x, y)
                row0 = int(row - tile_size // 2)
                col0 = int(col - tile_size // 2)
                if row0 < 0 or col0 < 0 or row0 + tile_size > query_ds.height or col0 + tile_size > query_ds.width:
                    continue
                rgb = read_rgb_window(query_ds, Window(col0, row0, tile_size, tile_size))
                if rgb.shape[:2] != (tile_size, tile_size):
                    continue
                gray = to_gray(rgb)
                std, ent, dark = tile_quality(gray)
                if std < min_std or ent < min_entropy or dark > max_dark_fraction:
                    continue

                query_id = f"Q{len(records) + 1:05d}"
                block_id = f"B{block_row:02d}_{block_col:02d}"
                tile_path = raw_dir / f"{query_id}.png"
                write_png(tile_path, rgb)
                records.append(
                    QueryRecord(
                        query_id=query_id,
                        block_id=block_id,
                        center_easting_m=x,
                        center_northing_m=y,
                        source_row=int(row),
                        source_col=int(col),
                        query_std=std,
                        query_entropy=ent,
                        dark_fraction=dark,
                        raw_tile_file=str(tile_path.resolve()),
                    )
                )
                accepted_in_block += 1
                if max_queries is not None and len(records) >= max_queries:
                    break
            if max_queries is not None and len(records) >= max_queries:
                break

    if not records:
        raise RuntimeError("Kalite kapılarından geçen hiçbir sorgu üretilemedi.")
    payload = {
        "created_at_utc": utc_now_iso(),
        "query_raster": str(query_raster.resolve()),
        "map_raster": str(map_raster.resolve()),
        "seed": seed,
        "tile_size": tile_size,
        "block_size_m": block_size_m,
        "samples_per_block": samples_per_block,
        "edge_buffer_m_requested": edge_buffer_m,
        "effective_edge_buffer_m": effective_edge_buffer_m,
        "selection_config": selection_config,
        "queries": [asdict(record) for record in records],
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_write(manifest_csv, [asdict(record) for record in records], MANIFEST_COLUMNS)
    LOG.info("Sorgu manifesti üretildi: %d sorgu | %s", len(records), manifest_json)
    return records


def hard_v1_seed(seed: int, query_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|hard_v1|{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def augment_hard_v1(image_rgb: np.ndarray, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a deterministic, probabilistic UAV-camera profile without geometry changes."""
    rng = np.random.default_rng(seed)
    transformed = image_rgb.astype(np.float32) / 255.0

    scenario_names = list(HARD_V1_PROFILE["scenarios"])
    scenario_probabilities = np.asarray(
        list(HARD_V1_PROFILE["scenarios"].values()), dtype=np.float64
    )
    scenario_probabilities /= scenario_probabilities.sum()
    scenario = str(rng.choice(scenario_names, p=scenario_probabilities))

    gain = float(rng.uniform(*HARD_V1_PROFILE["gain"]))
    bias = float(rng.uniform(*HARD_V1_PROFILE["bias"]))
    gamma = float(rng.uniform(*HARD_V1_PROFILE["gamma"]))
    white_balance = rng.uniform(
        *HARD_V1_PROFILE["white_balance_gain"], size=3
    ).astype(np.float32)
    saturation = float(rng.uniform(*HARD_V1_PROFILE["saturation"]))
    transformed = np.clip(
        transformed * gain * white_balance.reshape(1, 1, 3) + bias / 255.0,
        0.0,
        1.0,
    )
    transformed = np.power(transformed, gamma)
    luminance = np.sum(
        transformed * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=2,
        keepdims=True,
    )
    transformed = np.clip(luminance + saturation * (transformed - luminance), 0.0, 1.0)
    transformed = np.clip(transformed * 255.0, 0.0, 255.0).astype(np.uint8)

    effect_parameters: dict[str, Any] = {}
    if scenario == "haze":
        haze_alpha = float(rng.uniform(*HARD_V1_PROFILE["haze_alpha"]))
        airlight = rng.uniform(*HARD_V1_PROFILE["haze_airlight"], size=3).astype(np.float32)
        transformed = np.clip(
            transformed.astype(np.float32) * (1.0 - haze_alpha)
            + airlight.reshape(1, 1, 3) * haze_alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
        effect_parameters.update(
            haze_alpha=haze_alpha,
            haze_airlight_rgb=[float(value) for value in airlight],
        )
    elif scenario == "motion_blur":
        minimum, maximum = HARD_V1_PROFILE["motion_blur_length_px"]
        valid_lengths = np.arange(int(minimum) | 1, int(maximum) + 1, 2)
        motion_length = int(rng.choice(valid_lengths))
        motion_angle = float(rng.uniform(0.0, 180.0))
        kernel = np.zeros((motion_length, motion_length), dtype=np.float32)
        kernel[motion_length // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D(
            (motion_length / 2.0 - 0.5, motion_length / 2.0 - 0.5),
            motion_angle,
            1.0,
        )
        kernel = cv2.warpAffine(kernel, rotation, (motion_length, motion_length))
        kernel /= max(float(kernel.sum()), 1e-8)
        transformed = cv2.filter2D(
            transformed, -1, kernel, borderType=cv2.BORDER_REFLECT_101
        )
        effect_parameters.update(
            motion_blur_length_px=motion_length,
            motion_blur_angle_deg=motion_angle,
        )
    elif scenario == "defocus_blur":
        defocus_sigma = float(rng.uniform(*HARD_V1_PROFILE["defocus_sigma"]))
        transformed = cv2.GaussianBlur(
            transformed,
            (0, 0),
            sigmaX=defocus_sigma,
            sigmaY=defocus_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        effect_parameters["defocus_sigma"] = defocus_sigma
    elif scenario == "low_contrast":
        contrast_factor = float(rng.uniform(*HARD_V1_PROFILE["low_contrast_factor"]))
        channel_mean = transformed.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        transformed = np.clip(
            channel_mean + contrast_factor * (transformed.astype(np.float32) - channel_mean),
            0.0,
            255.0,
        ).astype(np.uint8)
        effect_parameters["low_contrast_factor"] = contrast_factor

    vignette_applied = bool(rng.random() < HARD_V1_PROFILE["vignette_probability"])
    vignette_strength = 0.0
    if vignette_applied:
        vignette_strength = float(rng.uniform(*HARD_V1_PROFILE["vignette_strength"]))
        height, width = transformed.shape[:2]
        yy, xx = np.ogrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        radial = np.clip(xx * xx + yy * yy, 0.0, 1.0).astype(np.float32)
        vignette = 1.0 - vignette_strength * radial
        transformed = np.clip(
            transformed.astype(np.float32) * vignette[..., None], 0.0, 255.0
        ).astype(np.uint8)

    if scenario == "compression_noise":
        noise_std = float(rng.uniform(*HARD_V1_PROFILE["compression_noise_std"]))
        jpeg_range = HARD_V1_PROFILE["compression_jpeg_quality"]
    elif scenario == "clear_light":
        noise_std = float(rng.uniform(0.5, 1.5))
        jpeg_range = [82, 95]
    else:
        noise_std = float(rng.uniform(*HARD_V1_PROFILE["noise_std"]))
        jpeg_range = HARD_V1_PROFILE["jpeg_quality"]
    noise = rng.normal(0.0, noise_std, transformed.shape).astype(np.float32)
    transformed = np.clip(transformed.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    jpeg_quality = int(rng.integers(jpeg_range[0], jpeg_range[1] + 1))
    bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded:
        raise RuntimeError("hard_v1 JPEG bozulması üretilemedi.")
    decoded_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("hard_v1 JPEG bozulması okunamadı.")
    transformed = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    parameters = {
        "seed": seed,
        "profile_revision": HARD_V1_PROFILE["profile_revision"],
        "scenario": scenario,
        "gain": gain,
        "bias": bias,
        "gamma": gamma,
        "white_balance_rgb": [float(value) for value in white_balance],
        "saturation": saturation,
        "vignette_applied": vignette_applied,
        "vignette_strength": vignette_strength,
        "noise_std": noise_std,
        "jpeg_quality": jpeg_quality,
        **effect_parameters,
    }
    return transformed, parameters


def prepare_query_variants(
    queries: Sequence[QueryRecord],
    queries_dir: Path,
    *,
    variants: Sequence[str],
    seed: int,
    force: bool,
) -> dict[str, list[QueryRecord]]:
    prepared: dict[str, list[QueryRecord]] = {}
    for variant in variants:
        if variant == "clean":
            prepared[variant] = list(queries)
            continue
        if variant != "hard_v1":
            raise ValueError(f"Desteklenmeyen sorgu varyantı: {variant}")

        variant_dir = queries_dir / "variants" / variant
        raw_dir = variant_dir / "raw_tiles"
        manifest_path = variant_dir / "query_variant_manifest.json"
        expected_ids = [record.query_id for record in queries]
        if manifest_path.is_file() and not force:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows_by_id = {
                str(row.get("query_id")): row
                for row in payload.get("queries", [])
                if isinstance(row, dict)
            }
            cache_valid = (
                payload.get("schema_version") == 1
                and payload.get("query_variant") == variant
                and payload.get("profile") == HARD_V1_PROFILE
                and int(payload.get("seed", -1)) == int(seed)
                and list(rows_by_id) == expected_ids
                and all(Path(rows_by_id[item]["raw_tile_file"]).is_file() for item in expected_ids)
            )
            if cache_valid:
                cached: list[QueryRecord] = []
                for base in queries:
                    row = rows_by_id[base.query_id]
                    cached.append(
                        QueryRecord(
                            query_id=base.query_id,
                            block_id=base.block_id,
                            center_easting_m=base.center_easting_m,
                            center_northing_m=base.center_northing_m,
                            source_row=base.source_row,
                            source_col=base.source_col,
                            query_std=float(row["query_std"]),
                            query_entropy=float(row["query_entropy"]),
                            dark_fraction=float(row["dark_fraction"]),
                            raw_tile_file=str(row["raw_tile_file"]),
                        )
                    )
                LOG.info("Sorgu varyantı yeniden kullanılıyor | varyant=%s | adet=%d", variant, len(cached))
                prepared[variant] = cached
                continue

        raw_dir.mkdir(parents=True, exist_ok=True)
        for old in raw_dir.glob("Q*.png"):
            old.unlink()
        variant_records: list[QueryRecord] = []
        manifest_rows: list[dict[str, Any]] = []
        for base in queries:
            source_bgr = cv2.imread(base.raw_tile_file, cv2.IMREAD_COLOR)
            if source_bgr is None:
                raise FileNotFoundError(f"Ham sorgu okunamadı: {base.raw_tile_file}")
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            augmented, parameters = augment_hard_v1(
                source_rgb,
                seed=hard_v1_seed(seed, base.query_id),
            )
            target = raw_dir / f"{base.query_id}.png"
            write_png(target, augmented)
            std, entropy, dark_fraction = tile_quality(to_gray(augmented))
            record = QueryRecord(
                query_id=base.query_id,
                block_id=base.block_id,
                center_easting_m=base.center_easting_m,
                center_northing_m=base.center_northing_m,
                source_row=base.source_row,
                source_col=base.source_col,
                query_std=std,
                query_entropy=entropy,
                dark_fraction=dark_fraction,
                raw_tile_file=str(target.resolve()),
            )
            variant_records.append(record)
            manifest_rows.append({**asdict(record), "augmentation": parameters})
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at_utc": utc_now_iso(),
                    "query_variant": variant,
                    "seed": seed,
                    "profile": HARD_V1_PROFILE,
                    "queries": manifest_rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        LOG.info("Sorgu varyantı üretildi | varyant=%s | adet=%d | %s", variant, len(variant_records), variant_dir)
        prepared[variant] = variant_records
    return prepared


def import_image_processor() -> Any:
    try:
        from goruntu_islemleri import ImageProcessor  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "goruntu_islemleri.py yüklenemedi. Kodu visual_navigation_cuda ortamında çalıştırın."
        ) from exc
    return ImageProcessor


def ensure_shared_map_tiles(
    map_raster: Path,
    shared_dir: Path,
    *,
    tile_size: int,
    overlap: int,
    force: bool,
) -> dict[str, Any]:
    metadata_path = shared_dir / "metadata.json"
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = int(metadata["num_frames_x"]) * int(metadata["num_frames_y"])
        actual = len(list(shared_dir.glob("goruntu_*.png")))
        if actual == expected and expected > 0:
            LOG.info("Harita karoları yeniden kullanılıyor: %s (%d karo)", shared_dir, actual)
            return metadata
        LOG.warning("Harita karo önbelleği eksik: expected=%d actual=%d; yeniden üretilecek.", expected, actual)

    shared_dir.mkdir(parents=True, exist_ok=True)
    for old in shared_dir.glob("goruntu_*.png"):
        old.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    ImageProcessor = import_image_processor()
    processor = ImageProcessor(reference_dir=None)
    with StageTimer("Arama haritasını ortak karolara bölme"):
        image = processor.load_image(str(map_raster))
        _, filenames, metadata = processor.split_image(
            image,
            tile_size=tile_size,
            overlap=overlap,
            frame_size=None,
            output_dir=str(shared_dir),
            prefix="goruntu",
            format="png",
            save_metadata=True,
            original_path=str(map_raster),
            # tqdm writes carriage-return updates. Disable it when stdout/stderr
            # is redirected (CI, Codex tool capture, scheduled task); structured
            # stage logs remain available in benchmark.log.
            show_progress=bool(sys.stderr.isatty()),
            keep_in_memory=False,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("Harita karo sayısı: %d", len(filenames))
        del image
    return metadata


def validate_generated_geotiff(path: Path, source_map: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as out_ds, rasterio.open(source_map) as src_ds:
            if out_ds.crs != src_ds.crs:
                return False
            if out_ds.width != src_ds.width or out_ds.height != src_ds.height:
                return False
            delta = np.max(np.abs(np.asarray(out_ds.transform) - np.asarray(src_ds.transform)))
            return bool(delta < 1e-8 and out_ds.count >= 1)
    except Exception:
        return False


def build_model_map(
    model_path: Path,
    model_dir: Path,
    shared_tiles: Path,
    metadata: dict[str, Any],
    source_map: Path,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    normalization: str,
    enhancement: str,
    force: bool,
    keep_intermediate: bool,
) -> tuple[Path, float]:
    model_id = safe_slug(model_path.stem)
    georef_path = model_dir / "04_georef" / f"{model_id}_geo.tif"
    timing_path = model_dir / "map_timing.json"
    if validate_generated_geotiff(georef_path, source_map) and not force:
        elapsed = 0.0
        if timing_path.exists():
            elapsed = float(json.loads(timing_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
        LOG.info("Model haritası yeniden kullanılıyor: %s", georef_path)
        return georef_path, elapsed

    ImageProcessor = import_image_processor()
    processor = ImageProcessor(reference_dir=None)
    tile_out = model_dir / "02_model_tiles"
    merged_dir = model_dir / "03_merged"
    georef_path.parent.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    tile_out.mkdir(parents=True, exist_ok=True)
    (tile_out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    merged_path = merged_dir / f"{model_id}.png"
    started = time.perf_counter()
    with StageTimer(f"Model haritası üretimi | {model_path.name}"):
        produced = processor.process_images_with_model(
            input_dir=str(shared_tiles),
            output_dir=str(tile_out),
            model_path=str(model_path),
            image_size=(tile_size, tile_size),
            color_mode="grayscale",
            batch_size=batch_size,
            normalization=normalization,
            enhancement=enhancement,
        )
        expected = int(metadata["num_frames_x"]) * int(metadata["num_frames_y"])
        if len(produced) != expected:
            raise RuntimeError(f"Eksik model karosu: expected={expected}, produced={len(produced)}")
        merged = processor.merge_images(
            input_dir=str(tile_out),
            output_path=str(merged_path),
            num_frames_x=int(metadata["num_frames_x"]),
            num_frames_y=int(metadata["num_frames_y"]),
            crop_overlap=overlap // 2,
            tile_size=int(metadata["tile_size"]),
            frame_size=int(metadata["frame_size"]),
        )
        # All validated top_modeller models are 1-channel outputs.  The shared
        # merge helper reads PNG files in OpenCV's default colour mode and can
        # therefore materialize three identical bands.  Collapse them before
        # georeferencing; this preserves the representation while reducing the
        # GeoTIFF and in-memory search map to one third of the size.
        if isinstance(merged, np.ndarray) and merged.ndim == 3:
            merged_gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
            if not cv2.imwrite(str(merged_path), merged_gray):
                raise OSError(f"Tek kanallı mozaik yazılamadı: {merged_path}")
            LOG.info(
                "Model mozaiği tek kanala indirildi | shape=%dx%d",
                merged_gray.shape[0],
                merged_gray.shape[1],
            )
            del merged_gray
        del merged
        with rasterio.open(source_map) as src_ds:
            processor.georeference_image(
                input_path=str(merged_path),
                reference_path=str(source_map),
                output_path=str(georef_path),
                target_transform=src_ds.transform,
                target_crs=src_ds.crs,
                force_reference_shape=False,
                transform_grid_shape=(src_ds.height, src_ds.width),
            )
        if not validate_generated_geotiff(georef_path, source_map):
            raise RuntimeError(f"Üretilen GeoTIFF doğrulanamadı: {georef_path}")

    elapsed = time.perf_counter() - started
    timing_path.write_text(
        json.dumps(
            {
                "model": str(model_path.resolve()),
                "elapsed_seconds": elapsed,
                "created_at_utc": utc_now_iso(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not keep_intermediate:
        if tile_out.exists():
            shutil.rmtree(tile_out)
            LOG.info("Yeniden üretilebilir model karo ara çıktıları silindi: %s", tile_out)
        if merged_path.exists():
            merged_path.unlink()
            LOG.info("Yeniden üretilebilir PNG mozaik silindi: %s", merged_path)
    return georef_path, elapsed


def build_model_queries(
    model_path: Path,
    queries: Sequence[QueryRecord],
    output_dir: Path,
    *,
    tile_size: int,
    batch_size: int,
    normalization: str,
    enhancement: str,
    force: bool,
) -> tuple[dict[str, Path], float]:
    expected_names = {f"{record.query_id}.png" for record in queries}
    existing = {path.name for path in output_dir.glob("Q*.png")} if output_dir.exists() else set()
    if existing == expected_names and not force:
        LOG.info("Model sorguları yeniden kullanılıyor: %s (%d sorgu)", output_dir, len(existing))
        return {path.stem: path for path in output_dir.glob("Q*.png")}, 0.0

    raw_dir = Path(queries[0].raw_tile_file).parent
    ImageProcessor = import_image_processor()
    processor = ImageProcessor(reference_dir=None)
    started = time.perf_counter()
    with StageTimer(f"Sorguları modelden geçirme | {model_path.name}"):
        produced = processor.process_images_with_model(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            model_path=str(model_path),
            image_size=(tile_size, tile_size),
            color_mode="grayscale",
            batch_size=batch_size,
            normalization=normalization,
            enhancement=enhancement,
        )
    paths = {Path(path).stem: Path(path) for path in produced}
    missing = sorted(record.query_id for record in queries if record.query_id not in paths)
    if missing:
        raise RuntimeError(f"Eksik model sorguları: {missing[:10]} (toplam {len(missing)})")
    return paths, time.perf_counter() - started


def load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return image


def read_map_gray(path: Path) -> tuple[np.ndarray, Affine, Any]:
    with rasterio.open(path) as dataset:
        gray = dataset.read(1)
        return gray, dataset.transform, dataset.crs


def build_pyramid(image: np.ndarray, factors: Sequence[int]) -> dict[int, np.ndarray]:
    height, width = image.shape[:2]
    pyramid: dict[int, np.ndarray] = {}
    for factor in factors:
        if factor == 1:
            pyramid[factor] = np.ascontiguousarray(image)
        else:
            out_width = max(1, int(round(width / factor)))
            out_height = max(1, int(round(height / factor)))
            pyramid[factor] = cv2.resize(
                image, (out_width, out_height), interpolation=cv2.INTER_AREA
            )
    return pyramid


def resize_template(template: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.ascontiguousarray(template)
    width = max(4, int(round(template.shape[1] / factor)))
    height = max(4, int(round(template.shape[0] / factor)))
    return cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)


def nms_top_candidates(
    response: np.ndarray,
    top_k: int,
    radius: int,
) -> list[Candidate]:
    if response.size == 0:
        return []
    work = response.astype(np.float32, copy=True)
    candidates: list[Candidate] = []
    for _ in range(top_k):
        _, max_value, _, max_location = cv2.minMaxLoc(work)
        if not np.isfinite(max_value):
            break
        x, y = int(max_location[0]), int(max_location[1])
        candidates.append(Candidate(x=x, y=y, score=float(max_value)))
        x0, x1 = max(0, x - radius), min(work.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(work.shape[0], y + radius + 1)
        work[y0:y1, x0:x1] = -np.inf
    return candidates


def deduplicate_candidates(
    candidates: Sequence[Candidate], top_k: int, radius: int
) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(math.hypot(candidate.x - old.x, candidate.y - old.y) > radius for old in kept):
            kept.append(candidate)
        if len(kept) >= top_k:
            break
    return kept


def peak_to_sidelobe(response: np.ndarray, peak_x: int, peak_y: int, exclusion: int) -> float:
    if response.size < 2:
        return 0.0
    mask = np.ones(response.shape, dtype=bool)
    x0, x1 = max(0, peak_x - exclusion), min(response.shape[1], peak_x + exclusion + 1)
    y0, y1 = max(0, peak_y - exclusion), min(response.shape[0], peak_y + exclusion + 1)
    mask[y0:y1, x0:x1] = False
    sidelobe = response[mask]
    if sidelobe.size == 0:
        return 0.0
    std = float(np.std(sidelobe))
    if std <= 1e-12:
        return 0.0
    return float((response[peak_y, peak_x] - float(np.mean(sidelobe))) / std)


def coarse_to_fine_search(
    map_pyramid: dict[int, np.ndarray],
    template: np.ndarray,
    *,
    factors: Sequence[int],
    top_k: int,
    refine_radius_full_px: int,
    nms_radius_full_px: int,
    template_pyramid: dict[int, np.ndarray] | None = None,
) -> SearchOutcome:
    if float(np.std(template)) < 1e-6:
        raise ValueError("Sorgu şablonu sabit/düşük varyanslı.")
    first_factor = factors[0]
    coarse_map = map_pyramid[first_factor]
    coarse_template = (
        template_pyramid[first_factor]
        if template_pyramid is not None
        else resize_template(template, first_factor)
    )
    if coarse_template.shape[0] > coarse_map.shape[0] or coarse_template.shape[1] > coarse_map.shape[1]:
        raise ValueError("Sorgu şablonu kaba haritadan büyük.")
    response = cv2.matchTemplate(coarse_map, coarse_template, cv2.TM_CCOEFF_NORMED)
    coarse_radius = max(1, int(round(nms_radius_full_px / first_factor)))
    candidates = nms_top_candidates(response, top_k=top_k, radius=coarse_radius)
    if not candidates:
        raise RuntimeError("Kaba arama aday üretmedi.")

    previous_factor = first_factor
    for factor in factors[1:]:
        current_map = map_pyramid[factor]
        current_template = (
            template_pyramid[factor]
            if template_pyramid is not None
            else resize_template(template, factor)
        )
        scale = previous_factor / float(factor)
        radius = max(2, int(math.ceil(refine_radius_full_px / factor)))
        refined: list[Candidate] = []
        for candidate in candidates:
            expected_x = int(round(candidate.x * scale))
            expected_y = int(round(candidate.y * scale))
            x0 = max(0, expected_x - radius)
            y0 = max(0, expected_y - radius)
            x1 = min(current_map.shape[1], expected_x + radius + current_template.shape[1] + 1)
            y1 = min(current_map.shape[0], expected_y + radius + current_template.shape[0] + 1)
            roi = current_map[y0:y1, x0:x1]
            if roi.shape[0] < current_template.shape[0] or roi.shape[1] < current_template.shape[1]:
                continue
            local = cv2.matchTemplate(roi, current_template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(local)
            refined.append(
                Candidate(x=x0 + int(location[0]), y=y0 + int(location[1]), score=float(score))
            )
        dedupe_radius = max(1, int(round(nms_radius_full_px / factor)))
        candidates = deduplicate_candidates(refined, top_k=top_k, radius=dedupe_radius)
        if not candidates:
            raise RuntimeError(f"İnce arama adayları tükendi: factor={factor}")
        previous_factor = factor

    final_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = final_candidates[0]
    second_score = float(final_candidates[1].score) if len(final_candidates) > 1 else -1.0
    full_map = map_pyramid[1]
    full_response_radius = max(8, refine_radius_full_px)
    x0 = max(0, best.x - full_response_radius)
    y0 = max(0, best.y - full_response_radius)
    x1 = min(full_map.shape[1], best.x + full_response_radius + template.shape[1] + 1)
    y1 = min(full_map.shape[0], best.y + full_response_radius + template.shape[0] + 1)
    local_map = full_map[y0:y1, x0:x1]
    local_response = cv2.matchTemplate(local_map, template, cv2.TM_CCOEFF_NORMED)
    local_x, local_y = best.x - x0, best.y - y0
    psr = peak_to_sidelobe(
        local_response,
        peak_x=max(0, min(local_x, local_response.shape[1] - 1)),
        peak_y=max(0, min(local_y, local_response.shape[0] - 1)),
        exclusion=max(2, nms_radius_full_px // 4),
    )
    return SearchOutcome(
        x=best.x,
        y=best.y,
        top1_score=float(best.score),
        top2_score=second_score,
        peak_margin=float(best.score - second_score),
        psr=psr,
    )


def pixel_center_to_geo(transform: Affine, x: float, y: float) -> tuple[float, float]:
    east, north = transform * (x, y)
    return float(east), float(north)


def raw_query_template(record: QueryRecord, crop_border: int) -> np.ndarray:
    image = load_gray_image(Path(record.raw_tile_file))
    if crop_border <= 0:
        return image
    if image.shape[0] <= 2 * crop_border or image.shape[1] <= 2 * crop_border:
        raise ValueError("crop_border sorgu boyutundan büyük.")
    return image[crop_border:-crop_border, crop_border:-crop_border]


def model_query_template(path: Path, crop_border: int) -> np.ndarray:
    image = load_gray_image(path)
    if crop_border <= 0:
        return image
    return image[crop_border:-crop_border, crop_border:-crop_border]


def completed_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            str(row.get("direction")),
            str(row.get("query_variant", "clean")),
            str(row.get("search_mode", "global")),
            str(row.get("model_id")),
            str(row.get("query_id")),
        )
        for row in rows
        if row.get("status") in {"ok", "error", "rejected"}
    }


def has_pending_searches(
    *,
    direction: str,
    query_variant: str,
    model_id: str,
    queries: Sequence[QueryRecord],
    search_modes: Sequence[tuple[str, float | None]],
    done: set[tuple[str, str, str, str, str]],
) -> bool:
    return any(
        (direction, query_variant, mode_name, model_id, query.query_id) not in done
        for query in queries
        for mode_name, _ in search_modes
    )


def search_in_mode(
    *,
    map_gray: np.ndarray,
    global_pyramid: dict[int, np.ndarray] | None,
    map_transform: Affine,
    template: np.ndarray,
    center_easting_m: float,
    center_northing_m: float,
    mode_name: str,
    roi_radius_m: float | None,
    factors: Sequence[int],
    top_k: int,
    refine_radius_px: int,
    nms_radius_px: int,
    template_pyramid: dict[int, np.ndarray] | None = None,
) -> SearchOutcome:
    if mode_name == "global":
        if global_pyramid is None:
            raise RuntimeError("Global arama piramidi hazırlanmadı.")
        return coarse_to_fine_search(
            global_pyramid,
            template,
            factors=factors,
            top_k=top_k,
            refine_radius_full_px=refine_radius_px,
            nms_radius_full_px=nms_radius_px,
            template_pyramid=template_pyramid,
        )
    if roi_radius_m is None:
        raise ValueError(f"ROI yarıçapı eksik: {mode_name}")
    expected_row, expected_col = rowcol(map_transform, center_easting_m, center_northing_m)
    radius_x = int(math.ceil(roi_radius_m / abs(map_transform.a)))
    radius_y = int(math.ceil(roi_radius_m / abs(map_transform.e)))
    half_w = template.shape[1] // 2
    half_h = template.shape[0] // 2
    x0 = max(0, int(expected_col) - radius_x - half_w)
    y0 = max(0, int(expected_row) - radius_y - half_h)
    x1 = min(map_gray.shape[1], int(expected_col) + radius_x + half_w + 1)
    y1 = min(map_gray.shape[0], int(expected_row) + radius_y + half_h + 1)
    roi = map_gray[y0:y1, x0:x1]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        raise ValueError(f"ROI şablondan küçük: mode={mode_name}, shape={roi.shape}")
    local = coarse_to_fine_search(
        build_pyramid(roi, factors),
        template,
        factors=factors,
        top_k=top_k,
        refine_radius_full_px=refine_radius_px,
        nms_radius_full_px=nms_radius_px,
        template_pyramid=template_pyramid,
    )
    return SearchOutcome(
        x=local.x + x0,
        y=local.y + y0,
        top1_score=local.top1_score,
        top2_score=local.top2_score,
        peak_margin=local.peak_margin,
        psr=local.psr,
    )


def prepare_search_map(
    map_path: Path,
    *,
    model_id: str,
    factors: Sequence[int],
    search_modes: Sequence[tuple[str, float | None]],
) -> PreparedSearchMap:
    """Load one read-only search map shared by all variants and workers."""
    LOG.info("Harita belleğe alınıyor | model=%s | path=%s", model_id, map_path)
    map_gray, map_transform, _ = read_map_gray(map_path)
    LOG.info(
        "Harita hazır | shape=%dx%d | bellek=%.1f MiB | piramit=%s",
        map_gray.shape[0],
        map_gray.shape[1],
        map_gray.nbytes / (1024 * 1024),
        ",".join(str(item) for item in factors),
    )
    pyramid: dict[int, np.ndarray] | None = None
    if any(mode_name == "global" for mode_name, _ in search_modes):
        with StageTimer(f"Harita piramidi | {model_id}"):
            pyramid = build_pyramid(map_gray, factors)
    return PreparedSearchMap(map_gray, map_transform, pyramid)


def run_searches_for_representation(
    *,
    run_id: str,
    direction: str,
    query_variant: str,
    model_id: str,
    model_file: str,
    model_sha256: str,
    map_path: Path,
    prepared_map: PreparedSearchMap,
    map_build_seconds: float,
    queries: Sequence[QueryRecord],
    query_paths: dict[str, Path] | None,
    query_inference_total_seconds: float,
    crop_border: int,
    search_modes: Sequence[tuple[str, float | None]],
    factors: Sequence[int],
    top_k: int,
    refine_radius_px: int,
    nms_radius_px: int,
    normalization: str,
    source_query_raster: Path,
    source_map_raster: Path,
    results_csv: Path,
    results_jsonl: Path,
    done: set[tuple[str, str, str, str, str]],
    search_workers: int,
) -> None:
    map_gray = prepared_map.gray
    map_transform = prepared_map.transform
    pyramid = prepared_map.pyramid

    pending = [
        (query, mode_name, radius)
        for query in queries
        for mode_name, radius in search_modes
        if (direction, query_variant, mode_name, model_id, query.query_id) not in done
    ]
    total = len(pending)
    per_query_inference = query_inference_total_seconds / max(1, len(queries))
    started = time.perf_counter()
    template_cache: dict[
        str, tuple[np.ndarray, float, dict[int, np.ndarray]]
    ] = {}
    unique_pending = {record.query_id: record for record, _, _ in pending}
    cache_bytes = 0
    for record in unique_pending.values():
        try:
            if model_id == RAW_MODEL_ID:
                template = raw_query_template(record, crop_border)
            else:
                if query_paths is None or record.query_id not in query_paths:
                    continue
                template = model_query_template(query_paths[record.query_id], crop_border)
            template = np.ascontiguousarray(template)
            template_std = float(np.std(template))
            if template_std < 2.0:
                continue
            template_levels = {
                factor: resize_template(template, factor) for factor in factors
            }
            template_cache[record.query_id] = (
                template,
                template_std,
                template_levels,
            )
            cache_bytes += sum(level.nbytes for level in template_levels.values())
        except Exception:
            # Preserve the existing per-task error/rejection behavior below.
            LOG.debug(
                "Şablon önbelleğe alınamadı | %s | %s",
                record.query_id,
                traceback.format_exc(),
            )
    LOG.info(
        "ŞABLON ÖNBELLEĞİ | model=%s | varyant=%s | %d/%d sorgu | %.1f MiB",
        model_id,
        query_variant,
        len(template_cache),
        len(unique_pending),
        cache_bytes / (1024 * 1024),
    )
    LOG.info(
        "ARAMA BAŞLADI | model=%s | varyant=%s | bekleyen=%d | toplam=%d | workers=%d",
        model_id,
        query_variant,
        total,
        len(queries),
        search_workers,
    )
    def evaluate_task(
        task: tuple[QueryRecord, str, float | None],
    ) -> tuple[QueryRecord, str, dict[str, Any]]:
        record, mode_name, roi_radius_m = task
        row_started = time.perf_counter()
        status = "ok"
        reason = "ok"
        result: dict[str, Any] = {
            "run_id": run_id,
            "direction": direction,
            "query_variant": query_variant,
            "search_mode": mode_name,
            "roi_radius_m": roi_radius_m,
            "model_id": model_id,
            "model_file": model_file,
            "model_sha256": model_sha256,
            "query_id": record.query_id,
            "block_id": record.block_id,
            "query_center_easting_m": record.center_easting_m,
            "query_center_northing_m": record.center_northing_m,
            "expected_center_easting_m": record.center_easting_m,
            "expected_center_northing_m": record.center_northing_m,
            "query_std": record.query_std,
            "query_entropy": record.query_entropy,
            "query_dark_fraction": record.dark_fraction,
            "query_inference_seconds": per_query_inference if model_id != RAW_MODEL_ID else 0.0,
            "map_build_seconds": map_build_seconds,
            "pyramid_factors": ",".join(str(item) for item in factors),
            "top_k": top_k,
            "normalization": normalization if model_id != RAW_MODEL_ID else "RAW",
            "template_size_px": 0,
            "source_query_raster": str(source_query_raster.resolve()),
            "source_map_raster": str(source_map_raster.resolve()),
            "created_at_utc": utc_now_iso(),
        }
        try:
            cached_template = template_cache.get(record.query_id)
            if cached_template is not None:
                template, template_std, template_levels = cached_template
            else:
                if model_id == RAW_MODEL_ID:
                    template = raw_query_template(record, crop_border)
                else:
                    if query_paths is None or record.query_id not in query_paths:
                        raise FileNotFoundError(f"Model sorgusu bulunamadı: {record.query_id}")
                    template = model_query_template(query_paths[record.query_id], crop_border)
                template = np.ascontiguousarray(template)
                template_std = float(np.std(template))
                template_levels = {
                    factor: resize_template(template, factor) for factor in factors
                }
            result["template_size_px"] = int(template.shape[0])
            if template_std < 2.0:
                status = "rejected"
                reason = "low_model_template_variance"
                raise ValueError(reason)
            outcome = search_in_mode(
                map_gray=map_gray,
                global_pyramid=pyramid,
                map_transform=map_transform,
                template=template,
                center_easting_m=record.center_easting_m,
                center_northing_m=record.center_northing_m,
                mode_name=mode_name,
                roi_radius_m=roi_radius_m,
                factors=factors,
                top_k=top_k,
                refine_radius_px=refine_radius_px,
                nms_radius_px=nms_radius_px,
                template_pyramid=template_levels,
            )
            predicted_center_x = outcome.x + template.shape[1] / 2.0
            predicted_center_y = outcome.y + template.shape[0] / 2.0
            pred_e, pred_n = pixel_center_to_geo(
                map_transform, predicted_center_x, predicted_center_y
            )
            expected_row, expected_col = rowcol(
                map_transform, record.center_easting_m, record.center_northing_m
            )
            error_px = math.hypot(
                predicted_center_x - float(expected_col),
                predicted_center_y - float(expected_row),
            )
            error_m = math.hypot(pred_e - record.center_easting_m, pred_n - record.center_northing_m)
            result.update(
                {
                    "predicted_center_easting_m": pred_e,
                    "predicted_center_northing_m": pred_n,
                    "error_m": error_m,
                    "error_px": error_px,
                    "top1_score": outcome.top1_score,
                    "top2_score": outcome.top2_score,
                    "peak_margin": outcome.peak_margin,
                    "psr": outcome.psr,
                    "success_5m": int(error_m <= 5.0),
                    "success_10m": int(error_m <= 10.0),
                    "success_25m": int(error_m <= 25.0),
                    "success_30m": int(error_m <= 30.0),
                    "success_50m": int(error_m <= 50.0),
                }
            )
        except Exception as exc:
            if status != "rejected":
                status = "error"
                reason = f"{type(exc).__name__}: {exc}"
            LOG.debug("Sorgu başarısız | %s | %s", record.query_id, traceback.format_exc())
        result["status"] = status
        result["reason"] = reason
        result["search_seconds"] = time.perf_counter() - row_started
        for column in RESULT_COLUMNS:
            result.setdefault(column, None)
        return record, mode_name, result

    executor: ThreadPoolExecutor | None = None
    if search_workers == 1:
        evaluated = map(evaluate_task, pending)
    else:
        executor = ThreadPoolExecutor(
            max_workers=search_workers,
            thread_name_prefix=f"search-{model_id[:24]}",
        )
        # executor.map computes concurrently but yields in input order. Numeric
        # result/checkpoint ordering therefore stays identical to serial mode.
        evaluated = executor.map(evaluate_task, pending)

    try:
        for index, (record, mode_name, result) in enumerate(evaluated, start=1):
            status = str(result["status"])
            reason = str(result["reason"])
            # Only this coordinator thread writes checkpoints; worker threads
            # share read-only map arrays and never write CSV/JSONL.
            csv_append(results_csv, result, RESULT_COLUMNS)
            jsonl_append(results_jsonl, result)
            done.add((direction, query_variant, mode_name, model_id, record.query_id))

            elapsed = time.perf_counter() - started
            rate = index / max(elapsed, 1e-9)
            eta = (total - index) / max(rate, 1e-9)
            error_text = (
                f"hata={result['error_m']:.2f} m skor={result['top1_score']:.4f}"
                if result.get("error_m") is not None
                else f"durum={status} neden={reason}"
            )
            LOG.info(
                "İLERLEME | model=%s | varyant=%s | mod=%s | %d/%d (%%%0.1f) | %s | %s | ETA=%.1f dk",
                model_id,
                query_variant,
                mode_name,
                index,
                total,
                100.0 * index / max(total, 1),
                record.query_id,
                error_text,
                eta / 60.0,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    LOG.info(
        "ARAMA TAMAMLANDI | model=%s | varyant=%s | süre=%.2f dk",
        model_id,
        query_variant,
        (time.perf_counter() - started) / 60.0,
    )


def percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def block_bootstrap_intervals(
    rows: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    all_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, float | None]:
    empty_result = {
        "median_error_ci95_low": None,
        "median_error_ci95_high": None,
        "success_10m_ci95_low": None,
        "success_10m_ci95_high": None,
        "success_30m_ci95_low": None,
        "success_30m_ci95_high": None,
    }
    all_rows = list(all_rows) if all_rows is not None else list(rows)
    if iterations <= 0 or not all_rows:
        return empty_result
    if not rows:
        success_30_by_block: dict[str, list[float]] = {}
        for row in all_rows:
            success_30_by_block.setdefault(
                str(row.get("block_id") or "__all__"), []
            ).append(0.0)
        all_blocks = sorted(success_30_by_block)
        rng = np.random.default_rng(seed)
        successes_30m = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled_all = rng.choice(all_blocks, size=len(all_blocks), replace=True)
            success_flags = np.asarray(
                [
                    value
                    for block in sampled_all
                    for value in success_30_by_block[str(block)]
                ],
                dtype=np.float64,
            )
            successes_30m[index] = np.mean(success_flags)
        empty_result["success_30m_ci95_low"] = float(
            np.percentile(successes_30m, 2.5)
        )
        empty_result["success_30m_ci95_high"] = float(
            np.percentile(successes_30m, 97.5)
        )
        return empty_result
    by_block: dict[str, list[float]] = {}
    for row in rows:
        by_block.setdefault(str(row.get("block_id") or "__all__"), []).append(float(row["error_m"]))
    blocks = sorted(by_block)
    if not blocks:
        return empty_result

    success_30_by_block: dict[str, list[float]] = {}
    for row in all_rows:
        is_success = (
            row.get("status") == "ok"
            and row.get("error_m") is not None
            and float(row["error_m"]) <= 30.0
        )
        success_30_by_block.setdefault(
            str(row.get("block_id") or "__all__"), []
        ).append(float(is_success))
    all_blocks = sorted(success_30_by_block)

    rng = np.random.default_rng(seed)
    medians = np.empty(iterations, dtype=np.float64)
    successes_10m = np.empty(iterations, dtype=np.float64)
    successes_30m = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        errors = np.asarray(
            [error for block in sampled for error in by_block[str(block)]], dtype=np.float64
        )
        medians[index] = np.median(errors)
        successes_10m[index] = np.mean(errors <= 10.0)
        sampled_all = rng.choice(all_blocks, size=len(all_blocks), replace=True)
        success_flags = np.asarray(
            [
                value
                for block in sampled_all
                for value in success_30_by_block[str(block)]
            ],
            dtype=np.float64,
        )
        successes_30m[index] = np.mean(success_flags)
    return {
        "median_error_ci95_low": float(np.percentile(medians, 2.5)),
        "median_error_ci95_high": float(np.percentile(medians, 97.5)),
        "success_10m_ci95_low": float(np.percentile(successes_10m, 2.5)),
        "success_10m_ci95_high": float(np.percentile(successes_10m, 97.5)),
        "success_30m_ci95_low": float(np.percentile(successes_30m, 2.5)),
        "success_30m_ci95_high": float(np.percentile(successes_30m, 97.5)),
    }


def aggregate_results(
    rows: Sequence[dict[str, Any]], *, bootstrap_iterations: int = 1000, seed: int = 42
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["direction"]),
                str(row.get("query_variant", "clean")),
                str(row.get("search_mode", "global")),
                str(row["model_id"]),
            ),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (direction, query_variant, search_mode, model_id), group in sorted(
        groups.items(), key=result_group_sort_key
    ):
        ok = [row for row in group if row.get("status") == "ok" and row.get("error_m") is not None]
        errors = np.asarray([float(row["error_m"]) for row in ok], dtype=np.float64)
        search_times = np.asarray(
            [float(row["search_seconds"]) for row in group if row.get("search_seconds") is not None],
            dtype=np.float64,
        )
        scores = np.asarray(
            [float(row["top1_score"]) for row in ok if row.get("top1_score") is not None],
            dtype=np.float64,
        )
        stable_group_seed = int.from_bytes(
            hashlib.sha256(
                f"{direction}|{query_variant}|{search_mode}|{model_id}".encode("utf-8")
            ).digest()[:8],
            "big",
        ) ^ int(seed)
        confidence_intervals = block_bootstrap_intervals(
            ok,
            iterations=bootstrap_iterations,
            seed=stable_group_seed,
            all_rows=group,
        )
        errors_under_30m = errors[errors <= 30.0]
        success_30m_count = int(errors_under_30m.size)
        success_30m = success_30m_count / max(1, len(group))
        auc_30m = (
            float(np.sum(np.clip(1.0 - errors / 30.0, 0.0, 1.0)))
            / max(1, len(group))
            if errors.size
            else 0.0
        )
        summary.append(
            {
                "direction": direction,
                "query_variant": query_variant,
                "search_mode": search_mode,
                "model_id": model_id,
                "total_queries": len(group),
                "ok_queries": len(ok),
                "rejected_queries": sum(row.get("status") == "rejected" for row in group),
                "error_queries": sum(row.get("status") == "error" for row in group),
                "coverage": len(ok) / max(1, len(group)),
                "success_30m_queries": success_30m_count,
                "success_30m": success_30m,
                "success_30m_failure_rate": 1.0 - success_30m,
                "auc_30m": auc_30m,
                "mean_error_under_30m": (
                    float(np.mean(errors_under_30m)) if errors_under_30m.size else None
                ),
                "median_error_under_30m": percentile(errors_under_30m, 50),
                "mean_error_m": float(np.mean(errors)) if errors.size else None,
                "median_error_m": percentile(errors, 50),
                **confidence_intervals,
                "p90_error_m": percentile(errors, 90),
                "p95_error_m": percentile(errors, 95),
                "success_5m": float(np.mean(errors <= 5.0)) if errors.size else None,
                "success_10m": float(np.mean(errors <= 10.0)) if errors.size else None,
                "success_25m": float(np.mean(errors <= 25.0)) if errors.size else None,
                "success_50m": float(np.mean(errors <= 50.0)) if errors.size else None,
                "mean_top1_score": float(np.mean(scores)) if scores.size else None,
                "mean_search_seconds": float(np.mean(search_times)) if search_times.size else None,
                "total_search_seconds": float(np.sum(search_times)) if search_times.size else 0.0,
            }
        )
    return summary


def write_summary_files(run_dir: Path) -> tuple[Path, Path]:
    result_rows = read_jsonl(run_dir / "results.jsonl")
    # JSONL is the checkpoint source of truth. Rebuild CSV on every summary
    # pass so an interruption between two append operations cannot leave a
    # duplicated or incomplete CSV.
    csv_write(run_dir / "results.csv", result_rows, RESULT_COLUMNS)
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    summary_metadata_path = run_dir / "summary_metadata.json"
    bootstrap_iterations = int(config.get("bootstrap_iterations", 1000))
    seed = int(config.get("seed", 42))
    current_metadata = {
        "schema_version": 3,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
    }

    previous_summary: list[dict[str, Any]] = []
    previous_metadata: dict[str, Any] = {}
    if summary_json.is_file():
        try:
            candidate = json.loads(summary_json.read_text(encoding="utf-8"))
            if isinstance(candidate, list):
                previous_summary = [row for row in candidate if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            LOG.warning("Önceki summary.json okunamadı; özetler yeniden hesaplanacak.")
    if summary_metadata_path.is_file():
        try:
            candidate = json.loads(summary_metadata_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                previous_metadata = candidate
        except (OSError, json.JSONDecodeError):
            LOG.warning("Önceki summary_metadata.json okunamadı; özetler yeniden hesaplanacak.")

    previous_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if previous_metadata == current_metadata:
        previous_by_key = {
            (
                str(row.get("direction", "")),
                str(row.get("query_variant", "clean")),
                str(row.get("search_mode", "global")),
                str(row.get("model_id", "")),
            ): row
            for row in previous_summary
        }

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in result_rows:
        key = (
            str(row.get("direction", "")),
            str(row.get("query_variant", "clean")),
            str(row.get("search_mode", "global")),
            str(row.get("model_id", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    reused_groups = 0
    recomputed_groups = 0
    for key, group in sorted(grouped.items(), key=result_group_sort_key):
        ok_count = sum(
            row.get("status") == "ok" and row.get("error_m") is not None for row in group
        )
        rejected_count = sum(row.get("status") == "rejected" for row in group)
        error_count = sum(row.get("status") == "error" for row in group)
        previous = previous_by_key.get(key)
        if (
            previous is not None
            and previous.get("total_queries") == len(group)
            and previous.get("ok_queries") == ok_count
            and previous.get("rejected_queries") == rejected_count
            and previous.get("error_queries") == error_count
        ):
            summary.append(previous)
            reused_groups += 1
        else:
            summary.extend(
                aggregate_results(
                    group,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                )
            )
            recomputed_groups += 1

    LOG.info(
        "ÖZET CHECKPOINT | grup=%d | yeniden_kullanılan=%d | hesaplanan=%d",
        len(summary),
        reused_groups,
        recomputed_groups,
    )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = (
        list(summary[0].keys())
        if summary
        else ["direction", "query_variant", "search_mode", "model_id"]
    )
    csv_write(summary_csv, summary, columns)
    summary_metadata_path.write_text(
        json.dumps(current_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_json, summary_csv


def invoke_excel_report(
    run_dir: Path,
    strict: bool,
    engine: str = "auto",
    *,
    incremental: bool = False,
    validation_mode: str = "deep",
) -> Path | None:
    global _AUTO_EXCEL_ENGINE

    artifact_builder = SCRIPT_DIR / "build_benchmark_excel.mjs"
    openpyxl_builder = SCRIPT_DIR / "build_benchmark_excel_openpyxl.py"
    output = run_dir / "benchmark_results.xlsx"
    failures: list[str] = []
    requested_engine = engine
    if incremental:
        if engine == "artifact":
            LOG.info(
                "Model checkpointinde gerçek artımlı güncelleme için openpyxl motoru kullanılacak."
            )
        engine = "openpyxl"
    if engine == "auto" and _AUTO_EXCEL_ENGINE is not None:
        engine = _AUTO_EXCEL_ENGINE
        LOG.info("Excel auto motoru yeniden kullanılıyor: %s", engine)

    def run_reporter(label: str, command: list[str]) -> Path | None:
        LOG.info("Excel motoru deneniyor | motor=%s | çıktı=%s", label, output)
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        actual_output: Path | None = None
        for line in completed.stdout.splitlines():
            if line.strip():
                LOG.info("Excel/%s | %s", label, line.strip())
            if line.startswith("WORKBOOK_READY_JSON: "):
                try:
                    payload = json.loads(line.removeprefix("WORKBOOK_READY_JSON: "))
                    candidate = Path(payload["path"]).resolve()
                    if candidate.is_file() and candidate.suffix.lower() == ".xlsx":
                        actual_output = candidate
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    LOG.warning("Excel/%s | çıktı yolu işaretçisi okunamadı.", label)
        if (
            label == "artifact"
            and completed.returncode != 0
            and "ERR_MODULE_NOT_FOUND" in completed.stderr
            and "@oai/artifact-tool" in completed.stderr
        ):
            LOG.warning("Excel/artifact | @oai/artifact-tool paketi bulunamadı.")
        else:
            for line in completed.stderr.splitlines():
                if line.strip():
                    log_method = LOG.warning if completed.returncode != 0 else LOG.info
                    log_method("Excel/%s | %s", label, line.strip())
        if completed.returncode == 0:
            actual_output = actual_output or output
        if (
            completed.returncode == 0
            and actual_output.is_file()
            and actual_output.stat().st_size > 0
        ):
            LOG.info("Excel raporu hazır | motor=%s | dosya=%s", label, actual_output)
            return actual_output
        failures.append(f"{label}: exit={completed.returncode}")
        return None

    artifact_result: Path | None = None
    if engine in {"auto", "artifact"}:
        node = shutil.which("node")
        if node is None:
            failures.append("artifact: Node.js bulunamadı")
            LOG.warning("Artifact Excel motoru atlandı: Node.js bulunamadı.")
        else:
            artifact_result = run_reporter(
                "artifact",
                [node, str(artifact_builder), "--run-dir", str(run_dir), "--output", str(output)],
            )
        if node is not None and artifact_result is not None:
            if requested_engine == "auto":
                _AUTO_EXCEL_ENGINE = "artifact"
            return artifact_result
        if engine == "auto":
            LOG.warning("Artifact motoru kullanılamadı; onaylı openpyxl yedeğine geçiliyor.")

    if engine in {"auto", "openpyxl"} or (engine == "artifact" and failures):
        command = [
            sys.executable,
            str(openpyxl_builder),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--validation-mode",
            validation_mode,
        ]
        if incremental:
            command.append("--incremental")
        openpyxl_result = run_reporter(
            "openpyxl",
            command,
        )
        if openpyxl_result is not None:
            if requested_engine == "auto":
                _AUTO_EXCEL_ENGINE = "openpyxl"
            return openpyxl_result

    message = "Excel raporu üretilemedi | " + " | ".join(failures)
    if strict:
        raise RuntimeError(message)
    LOG.error(message)
    return None


def refresh_excel_after_model(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    position: int,
    total: int,
    model_name: str,
    model_status: str,
) -> Path | None:
    """Create a non-fatal, model-boundary Excel checkpoint."""
    if args.excel_update != "model":
        return None
    started = time.perf_counter()
    LOG.info(
        "ARA EXCEL BAŞLADI | model=%d/%d | durum=%s | %s",
        position,
        total,
        model_status,
        model_name,
    )
    try:
        summary_json, summary_csv = write_summary_files(run_dir)
        workbook = invoke_excel_report(
            run_dir,
            strict=False,
            engine=args.excel_engine,
            incremental=True,
            validation_mode="checkpoint",
        )
    except Exception:
        LOG.exception(
            "ARA EXCEL HATASI | benchmark devam edecek | model=%d/%d | %s",
            position,
            total,
            model_name,
        )
        return None
    if workbook is None:
        LOG.warning(
            "ARA EXCEL ÜRETİLEMEDİ | benchmark devam edecek | model=%d/%d | %s",
            position,
            total,
            model_name,
        )
        return None
    LOG.info(
        "ARA EXCEL TAMAMLANDI | model=%d/%d | süre=%.2f sn | excel=%s | özet=%s | csv=%s",
        position,
        total,
        time.perf_counter() - started,
        workbook,
        summary_json,
        summary_csv,
    )
    return workbook


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
    }
    try:
        import tensorflow as tf  # type: ignore

        info["tensorflow"] = tf.__version__
        info["tensorflow_cuda_build"] = bool(tf.test.is_built_with_cuda())
        info["tensorflow_gpus"] = [str(item) for item in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        info["tensorflow_error"] = str(exc)
    return info


def direction_name(query_raster: Path, map_raster: Path) -> str:
    return f"{query_raster.stem}__TO__{map_raster.stem}"


def run_direction(args: argparse.Namespace, run_dir: Path, query_raster: Path, map_raster: Path) -> None:
    direction = direction_name(query_raster, map_raster)
    direction_dir = run_dir / safe_slug(direction)
    direction_dir.mkdir(parents=True, exist_ok=True)
    raster_info = validate_rasters(query_raster, map_raster)
    LOG.info("YÖN | %s", direction)
    LOG.info(
        "RASTER | CRS=%s | query=%s | map=%s | common_bounds=%s",
        raster_info["crs"],
        raster_info["query_shape"],
        raster_info["map_shape"],
        raster_info["common_bounds"],
    )

    queries = generate_query_manifest(
        query_raster,
        map_raster,
        direction_dir / "queries",
        tile_size=args.tile_size,
        samples_per_block=args.samples_per_block,
        block_size_m=args.block_size_m,
        max_queries=args.max_queries,
        seed=args.seed,
        min_std=args.min_query_std,
        min_entropy=args.min_query_entropy,
        max_dark_fraction=args.max_dark_fraction,
        edge_buffer_m=args.query_edge_buffer_m,
        force=args.force_queries,
    )
    query_variants = prepare_query_variants(
        queries,
        direction_dir / "queries",
        variants=args.query_variants,
        seed=args.seed,
        force=args.force_queries,
    )

    shared_tiles = direction_dir / "map_shared_tiles"
    metadata: dict[str, Any] | None = None
    if args.include_models:
        metadata = ensure_shared_map_tiles(
            map_raster,
            shared_tiles,
            tile_size=args.tile_size,
            overlap=args.overlap,
            force=args.force_maps,
        )

    results_csv = run_dir / "results.csv"
    results_jsonl = run_dir / "results.jsonl"
    existing_rows = read_jsonl(results_jsonl)
    done = completed_keys(existing_rows)

    if args.include_raw:
        pending_raw_variants = [
            (query_variant, variant_queries)
            for query_variant, variant_queries in query_variants.items()
            if has_pending_searches(
                direction=direction,
                query_variant=query_variant,
                model_id=RAW_MODEL_ID,
                queries=variant_queries,
                search_modes=args.search_modes,
                done=done,
            )
        ]
        if not pending_raw_variants:
            LOG.info("RAW CHECKPOINT TAMAM | yön=%s | harita yüklenmeden atlandı", direction)
        else:
            raw_search_map = prepare_search_map(
                map_raster,
                model_id=RAW_MODEL_ID,
                factors=args.pyramid_factors,
                search_modes=args.search_modes,
            )
            try:
                for query_variant, variant_queries in pending_raw_variants:
                    run_searches_for_representation(
                        run_id=args.run_id,
                        direction=direction,
                        query_variant=query_variant,
                        model_id=RAW_MODEL_ID,
                        model_file="",
                        model_sha256="",
                        map_path=map_raster,
                        prepared_map=raw_search_map,
                        map_build_seconds=0.0,
                        queries=variant_queries,
                        query_paths=None,
                        query_inference_total_seconds=0.0,
                        crop_border=args.crop_border,
                        search_modes=args.search_modes,
                        factors=args.pyramid_factors,
                        top_k=args.top_k,
                        refine_radius_px=args.refine_radius_px,
                        nms_radius_px=args.nms_radius_px,
                        normalization="RAW",
                        source_query_raster=query_raster,
                        source_map_raster=map_raster,
                        results_csv=results_csv,
                        results_jsonl=results_jsonl,
                        done=done,
                        search_workers=args.search_workers,
                    )
            finally:
                del raw_search_map

    if not args.include_models:
        return
    assert metadata is not None
    patterns = parse_patterns(args.models)
    models = select_models(args.model_dir, patterns, args.max_models)
    LOG.info("MODEL LİSTESİ | adet=%d", len(models))
    for position, model_path in enumerate(models, start=1):
        model_id = safe_slug(model_path.stem)
        model_status = "başarısız"
        LOG.info("MODEL BAŞLIYOR | %d/%d | %s", position, len(models), model_path.name)
        model_sha = sha256_file(model_path)
        previous_model_shas = {
            str(row.get("model_sha256"))
            for row in existing_rows
            if row.get("direction") == direction
            and row.get("model_id") == model_id
            and row.get("model_sha256")
        }
        if previous_model_shas and previous_model_shas != {model_sha}:
            raise RuntimeError(
                f"Model dosyası önceki checkpointten sonra değişmiş: {model_path.name}. "
                "Sonuçların karışmaması için yeni bir --run-id kullanın."
            )
        model_root = direction_dir / "models" / model_id
        excel_refresh_needed = True
        try:
            pending_model_variants = [
                (query_variant, variant_queries)
                for query_variant, variant_queries in query_variants.items()
                if has_pending_searches(
                    direction=direction,
                    query_variant=query_variant,
                    model_id=model_id,
                    queries=variant_queries,
                    search_modes=args.search_modes,
                    done=done,
                )
            ]
            if not pending_model_variants:
                model_status = "checkpoint tamam"
                excel_refresh_needed = False
                LOG.info(
                    "MODEL CHECKPOINT TAMAM | %d/%d | %s | inference ve harita yükleme atlandı",
                    position,
                    len(models),
                    model_path.name,
                )
                continue
            model_map, map_seconds = build_model_map(
                model_path,
                model_root,
                shared_tiles,
                metadata,
                map_raster,
                tile_size=args.tile_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                normalization=args.normalization,
                enhancement=args.enhancement,
                force=args.force_maps,
                keep_intermediate=args.keep_intermediate,
            )
            model_search_map = prepare_search_map(
                model_map,
                model_id=model_id,
                factors=args.pyramid_factors,
                search_modes=args.search_modes,
            )
            try:
                for query_variant, variant_queries in pending_model_variants:
                    query_paths, query_seconds = build_model_queries(
                        model_path,
                        variant_queries,
                        model_root / "query_model_tiles" / query_variant,
                        tile_size=args.tile_size,
                        batch_size=args.batch_size,
                        normalization=args.normalization,
                        enhancement=args.enhancement,
                        force=args.force_queries,
                    )
                    run_searches_for_representation(
                        run_id=args.run_id,
                        direction=direction,
                        query_variant=query_variant,
                        model_id=model_id,
                        model_file=str(model_path.resolve()),
                        model_sha256=model_sha,
                        map_path=model_map,
                        prepared_map=model_search_map,
                        map_build_seconds=map_seconds,
                        queries=variant_queries,
                        query_paths=query_paths,
                        query_inference_total_seconds=query_seconds,
                        crop_border=args.crop_border,
                        search_modes=args.search_modes,
                        factors=args.pyramid_factors,
                        top_k=args.top_k,
                        refine_radius_px=args.refine_radius_px,
                        nms_radius_px=args.nms_radius_px,
                        normalization=args.normalization,
                        source_query_raster=query_raster,
                        source_map_raster=map_raster,
                        results_csv=results_csv,
                        results_jsonl=results_jsonl,
                        done=done,
                        search_workers=args.search_workers,
                    )
            finally:
                del model_search_map
        except Exception as exc:
            LOG.error("MODEL BAŞARISIZ | %s | %s", model_path.name, exc)
            LOG.debug("Model traceback:\n%s", traceback.format_exc())
            error_record = {
                "created_at_utc": utc_now_iso(),
                "direction": direction,
                "model_id": model_id,
                "model_file": str(model_path.resolve()),
                "model_sha256": model_sha,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            jsonl_append(run_dir / "model_errors.jsonl", error_record)
            if args.fail_fast:
                raise
        else:
            model_status = "tamamlandı"
            LOG.info("MODEL TAMAMLANDI | %d/%d | %s", position, len(models), model_path.name)
        finally:
            if excel_refresh_needed:
                refresh_excel_after_model(
                    run_dir,
                    args,
                    position=position,
                    total=len(models),
                    model_name=model_path.name,
                    model_status=model_status,
                )
            else:
                LOG.info(
                    "ARA EXCEL ATLANDI | model=%d/%d | checkpoint zaten güncel | %s",
                    position,
                    len(models),
                    model_path.name,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aynı sinir ağı temsilinde jeoreferanslı global template-matching benchmarkı."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query-raster", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--map-raster", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--models",
        action="append",
        help="Model dosya globu; birden çok kez verilebilir. Örn: --models '*f48*'",
    )
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run", type=Path, default=None)
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "İki rasterı sırayla sorgu ve arama haritası yap (varsayılan: açık). "
            "Yalnız ilk yön için --no-bidirectional kullanın."
        ),
    )
    parser.add_argument("--include-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--query-variants",
        type=parse_query_variants,
        default=("clean", "hard_v1"),
        help=(
            "Virgülle ayrılmış sorgu koşulları. clean mevcut benchmarkı korur; "
            "hard_v1 geometriyi koruyan deterministik sensör/görüntü bozulmaları ekler."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=544)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--crop-border", type=int, default=16)
    parser.add_argument("--block-size-m", type=float, default=1000.0)
    parser.add_argument("--samples-per-block", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-query-std", type=float, default=12.0)
    parser.add_argument("--min-query-entropy", type=float, default=4.0)
    parser.add_argument("--max-dark-fraction", type=float, default=0.20)
    parser.add_argument(
        "--query-edge-buffer-m",
        type=float,
        default=None,
        help=(
            "Ortak raster sınırında sorgu alınmayacak ek tampon (metre). "
            "Varsayılan: bir tam sorgu karosu genişliği."
        ),
    )
    parser.add_argument(
        "--search-modes",
        type=parse_search_modes,
        default=(
            ("roi_500m", 500.0),
            ("roi_1000m", 1000.0),
            ("roi_2000m", 2000.0),
            ("roi_4000m", 4000.0),
            ("roi_8000m", 8000.0),
            ("global", None),
        ),
        help="Virgülle ayrılmış global ve ROI modları; örn. global,roi250,roi500",
    )
    parser.add_argument("--pyramid-factors", type=parse_int_list, default=(16, 8, 4, 2, 1))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--refine-radius-px", type=int, default=160)
    parser.add_argument("--nms-radius-px", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--search-workers",
        type=int,
        default=8,
        help=(
            "Aynı haritayı salt-okunur paylaşan paralel arama işçisi. "
            "Kalite karşılaştırması/seri çalışma için 1 kullanın."
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--normalization",
        choices=["minus1_1", "zero_1", "raw", "zscore"],
        default="minus1_1",
    )
    parser.add_argument("--enhancement", choices=["none", "hist_eq", "clahe"], default="none")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--force-queries", action="store_true")
    parser.add_argument("--force-maps", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--strict-excel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--excel-engine",
        choices=["auto", "artifact", "openpyxl"],
        default="auto",
        help="Excel rapor motoru; auto önce Artifact, sonra onaylı openpyxl yedeğini dener.",
    )
    parser.add_argument(
        "--excel-update",
        choices=["model", "end"],
        default="model",
        help="Excel raporunu her model denemesinden sonra veya yalnız koşu sonunda günceller.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0 or args.tile_size % 2 != 0:
        raise ValueError("tile-size pozitif ve çift olmalıdır.")
    if args.overlap < 0 or args.overlap >= args.tile_size or args.overlap % 2 != 0:
        raise ValueError("overlap çift, >=0 ve tile-size değerinden küçük olmalıdır.")
    if args.crop_border < 0 or args.crop_border * 2 >= args.tile_size:
        raise ValueError("crop-border geçersiz.")
    if args.crop_border != args.overlap // 2:
        raise ValueError("Tutarlılık için crop-border = overlap / 2 olmalıdır.")
    if args.samples_per_block < 1 or (args.max_queries is not None and args.max_queries < 1):
        raise ValueError("Sorgu sayıları pozitif olmalıdır.")
    if args.top_k < 2:
        raise ValueError("Belirsizlik ölçümü için top-k en az 2 olmalıdır.")
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap-iterations negatif olamaz.")
    if args.search_workers < 1 or args.search_workers > 8:
        raise ValueError("search-workers 1 ile 8 arasında olmalıdır.")
    if args.query_edge_buffer_m is not None and args.query_edge_buffer_m < 0:
        raise ValueError("query-edge-buffer-m negatif olamaz.")
    if not args.include_raw and not args.include_models:
        raise ValueError("En az RAW veya model kanalı etkin olmalıdır.")


def resume_signature_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Return only settings that can change scientific benchmark results."""

    def raster_identity(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        "schema_version": 1,
        "query_raster": raster_identity(args.query_raster),
        "map_raster": raster_identity(args.map_raster),
        "model_dir": str(args.model_dir.resolve()),
        "models": list(args.models or []),
        "max_models": args.max_models,
        "bidirectional": args.bidirectional,
        "include_raw": args.include_raw,
        "include_models": args.include_models,
        "query_variants": list(args.query_variants),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "crop_border": args.crop_border,
        "block_size_m": args.block_size_m,
        "samples_per_block": args.samples_per_block,
        "max_queries": args.max_queries,
        "seed": args.seed,
        "min_query_std": args.min_query_std,
        "min_query_entropy": args.min_query_entropy,
        "max_dark_fraction": args.max_dark_fraction,
        "query_edge_buffer_m": args.query_edge_buffer_m,
        "search_modes": [
            {"name": name, "roi_radius_m": radius} for name, radius in args.search_modes
        ],
        "pyramid_factors": list(args.pyramid_factors),
        "top_k": args.top_k,
        "refine_radius_px": args.refine_radius_px,
        "nms_radius_px": args.nms_radius_px,
        "batch_size": args.batch_size,
        "normalization": args.normalization,
        "enhancement": args.enhancement,
        "hard_v1_profile": HARD_V1_PROFILE if "hard_v1" in args.query_variants else None,
    }


def signature_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    args.query_raster = args.query_raster.resolve()
    args.map_raster = args.map_raster.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.resume_run:
        run_dir = args.resume_run.resolve()
        args.run_id = run_dir.name
    else:
        args.run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = args.output_root / safe_slug(args.run_id)
    log_path = configure_logging(run_dir, args.verbose)
    LOG.info("=" * 88)
    LOG.info("JEOREFERANSLI ORTAK-TEMSİL BENCHMARKI")
    LOG.info("=" * 88)
    LOG.info("Run klasörü: %s", run_dir)
    LOG.info("Log: %s", log_path)
    LOG.info("Sistem: %s", json.dumps(system_info(), ensure_ascii=False))

    config_path = run_dir / "run_config.json"
    results_path = run_dir / "results.jsonl"
    resume_payload = resume_signature_payload(args)
    resume_signature = signature_sha256(resume_payload)
    if config_path.is_file() and results_path.is_file():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_signature = previous_config.get("resume_signature")
        if previous_signature is not None and previous_signature != resume_signature:
            raise RuntimeError(
                "Bu koşu klasöründeki bilimsel ayarlar/raster kimliği mevcut komutla "
                "uyuşmuyor. Eski ve yeni sonuçların karışmaması için aynı ayarları "
                "kullanın veya yeni bir --run-id verin."
            )
        if (
            previous_signature is None
            and "hard_v1" in args.query_variants
            and previous_config.get("hard_v1_profile") != HARD_V1_PROFILE
        ):
            raise RuntimeError(
                "Bu eski koşudaki hard_v1 profili mevcut İHA profiliyle uyuşmuyor. "
                "Bilimsel sonuçların karışmaması için yeni bir --run-id kullanın."
            )
        if previous_signature is None:
            LOG.warning(
                "Eski koşuda resume_signature yok; mevcut uyumlu ayarlar yeni imzayla kaydedilecek."
            )

    config = vars(args).copy()
    config["query_raster"] = str(args.query_raster)
    config["map_raster"] = str(args.map_raster)
    config["model_dir"] = str(args.model_dir)
    config["output_root"] = str(args.output_root)
    config["resume_run"] = str(args.resume_run) if args.resume_run else None
    config["pyramid_factors"] = list(args.pyramid_factors)
    config["search_modes"] = [
        {"name": name, "roi_radius_m": radius} for name, radius in args.search_modes
    ]
    config["hard_v1_profile"] = HARD_V1_PROFILE
    config["resume_signature_payload"] = resume_payload
    config["resume_signature"] = resume_signature
    config["created_at_utc"] = utc_now_iso()
    config["system"] = system_info()
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    started = time.perf_counter()
    try:
        run_direction(args, run_dir, args.query_raster, args.map_raster)
        if args.bidirectional:
            run_direction(args, run_dir, args.map_raster, args.query_raster)
        summary_json, summary_csv = write_summary_files(run_dir)
        LOG.info("Özet dosyaları: %s | %s", summary_json, summary_csv)
        workbook = invoke_excel_report(
            run_dir, strict=args.strict_excel, engine=args.excel_engine
        )
        LOG.info("=" * 88)
        LOG.info("BENCHMARK TAMAMLANDI | toplam süre=%.2f dk", (time.perf_counter() - started) / 60.0)
        LOG.info("Sonuç JSONL: %s", run_dir / "results.jsonl")
        LOG.info("Sonuç CSV: %s", run_dir / "results.csv")
        LOG.info("Excel: %s", workbook or "üretilemedi")
        LOG.info("=" * 88)
        return 0
    except KeyboardInterrupt:
        LOG.warning("Kullanıcı tarafından durduruldu. Checkpointler korundu; --resume-run ile devam edilebilir.")
        write_summary_files(run_dir)
        if args.excel_update == "end":
            try:
                invoke_excel_report(run_dir, strict=False, engine=args.excel_engine)
            except Exception:
                LOG.exception("Kesinti sonrası Excel raporu üretilemedi.")
        else:
            LOG.info(
                "KESİNTİ EXCEL | model checkpoint Excel'i korunuyor; ikinci tam export atlandı."
            )
        return 130
    except Exception:
        LOG.exception("BENCHMARK BAŞARISIZ. Mevcut checkpointler korunmuştur.")
        try:
            write_summary_files(run_dir)
            invoke_excel_report(run_dir, strict=False, engine=args.excel_engine)
        except Exception:
            LOG.exception("Hata sonrası kısmi rapor üretilemedi.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
