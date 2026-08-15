#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select successful training lineages and benchmark every epoch on 2,000 queries.

This module is deliberately an orchestration layer.  All raster truth, neural
inference, matching, error and metric calculations remain in
``geospatial_model_benchmark.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import traceback
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import geospatial_model_benchmark as core


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "epoch_sweep_2000"
DEFAULT_STAGING_DIR = SCRIPT_DIR / "models_epoch_sweep"
SWEEP_QUERY_COUNT = 2000
TOP_MODEL_COUNT = 10
SWEEP_SCHEMA_VERSION = 1
ALLOWED_MODEL_SUFFIXES = {".h5", ".hdf5", ".keras"}
EPOCH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])epoch(?P<sep>[_ -]*)(?P<number>\d+)(?![A-Za-z0-9])"
)
NON_EPOCH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:step|batch)(?=[_ -]*\d)"
)
REQUIRED_SOURCE_METRICS = {
    "success_25m",
    "auc_25m",
    "success_5m",
    "success_10m",
    "median_error_under_25m",
}
EPOCH_RESULT_COLUMNS = [
    "training_lineage",
    "lineage_rank",
    "epoch",
    "model_file",
    "model_sha256",
    "status",
    "clean_success_25m",
    "hard_success_25m",
    "mean_success_25m",
    "clean_auc_25m",
    "hard_auc_25m",
    "mean_auc_25m",
    "clean_success_5m",
    "hard_success_5m",
    "mean_success_5m",
    "clean_success_10m",
    "hard_success_10m",
    "clean_hard_gap",
    "clean_median_error_under_25m",
    "hard_median_error_under_25m",
    "mean_median_error_under_25m",
    "evaluated_model_id",
    "canonical_epoch",
    "duplicate_sha_alias",
]


@dataclass(frozen=True)
class RankedModel:
    rank: int
    model_id: str
    model_file: str
    model_sha256: str
    source_series_id: str
    source_series_key: str
    source_relative_path: str
    source_checkpoint_number: int | None
    completed_directions: tuple[str, ...]
    clean_success_25m: float
    hard_success_25m: float
    mean_success_25m: float
    clean_auc_25m: float
    hard_auc_25m: float
    mean_auc_25m: float
    clean_success_5m: float
    hard_success_5m: float
    mean_success_5m: float
    median_error_under_25m: float | None
    source_duplicate_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchiveCheckpoint:
    path: Path
    relative_path: str
    epoch: int
    sha256: str
    normalized_stem: str
    lineage_key: str


@dataclass
class ResolvedLineage:
    lineage_key: str
    lineage_id: str
    checkpoints: list[ArchiveCheckpoint]
    selected_models: list[RankedModel] = field(default_factory=list)
    mirror_alias_keys: list[str] = field(default_factory=list)

    @property
    def first_rank(self) -> int:
        return min(item.rank for item in self.selected_models)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Gerekli dosya bulunamadı: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Geçersiz JSON: {path}: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mean(values: Iterable[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def required_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Tamamlanmış özet hücresinde {key} eksik: {row}")
    return float(value)


def configured_directions(config: dict[str, Any]) -> tuple[str, ...]:
    query = Path(str(config["query_raster"]))
    search_map = Path(str(config["map_raster"]))
    directions = [core.direction_name(query, search_map)]
    if bool(config.get("bidirectional", True)):
        directions.append(core.direction_name(search_map, query))
    return tuple(directions)


def _source_has_completion_evidence(run_dir: Path) -> bool:
    log_path = run_dir / "benchmark.log"
    return log_path.is_file() and "BENCHMARK TAMAMLANDI" in log_path.read_text(
        encoding="utf-8", errors="replace"
    )


def _catalog_identity(run_dir: Path) -> dict[str, dict[str, Any]]:
    catalog_path = run_dir / "model_catalog.json"
    if not catalog_path.is_file():
        return {}
    payload = load_json(catalog_path)
    return {
        str(row["model_id"]): dict(row)
        for row in payload.get("models", [])
        if isinstance(row, dict) and row.get("model_id")
    }


def _recompute_legacy_summary(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = core.read_jsonl(run_dir / "results.jsonl")
    global_rows = [row for row in rows if row.get("search_mode", "global") == "global"]
    if not global_rows:
        raise ValueError(f"Global sonuç bulunamadı: {run_dir / 'results.jsonl'}")
    identities: dict[str, dict[str, Any]] = {}
    for row in global_rows:
        model_id = str(row.get("model_id", ""))
        if model_id and model_id != core.RAW_MODEL_ID and model_id not in identities:
            identities[model_id] = {
                "model_id": model_id,
                "path": str(row.get("model_file") or ""),
                "relative_path": Path(str(row.get("model_file") or model_id)).name,
                "sha256": str(row.get("model_sha256") or ""),
                "series_id": "",
                "series_key": "",
                "checkpoint_number": None,
            }
    summary = core.aggregate_results(global_rows, bootstrap_iterations=0, seed=42)
    return summary, identities


def load_source_run(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    run_dir = run_dir.resolve()
    config = load_json(run_dir / "run_config.json")
    if not isinstance(config, dict):
        raise ValueError("run_config.json bir nesne olmalıdır.")
    if not _source_has_completion_evidence(run_dir):
        raise ValueError(
            "Kaynak benchmark tamamlanmış görünmüyor: benchmark.log içinde final "
            "BENCHMARK TAMAMLANDI işareti bulunamadı."
        )
    variants = set(config.get("query_variants") or [])
    if not {"clean", "hard_v1"}.issubset(variants):
        raise ValueError("Kaynak koşuda clean ve hard_v1 birlikte tamamlanmış olmalıdır.")
    modes = {
        str(item.get("name")) if isinstance(item, dict) else str(item[0])
        for item in config.get("search_modes", [])
    }
    if "global" not in modes:
        raise ValueError("Kaynak koşuda global arama sonucu bulunmalıdır.")

    summary = load_json(run_dir / "summary.json")
    if not isinstance(summary, list):
        raise ValueError("summary.json bir liste olmalıdır.")
    catalog = _catalog_identity(run_dir)
    global_rows = [row for row in summary if row.get("search_mode", "global") == "global"]
    if not global_rows or not REQUIRED_SOURCE_METRICS.issubset(global_rows[0]):
        global_rows, legacy_identities = _recompute_legacy_summary(run_dir)
        catalog.update({key: value for key, value in legacy_identities.items() if key not in catalog})

    expected_queries = int(config.get("max_queries") or 0)
    if expected_queries < 1:
        raise ValueError("Kaynak max_queries değeri geçersiz.")
    expected_directions = set(configured_directions(config))
    observed_directions = {str(row.get("direction")) for row in global_rows}
    if not expected_directions.issubset(observed_directions):
        raise ValueError(
            f"Kaynak yönleri eksik: beklenen={sorted(expected_directions)}, "
            f"görülen={sorted(observed_directions)}"
        )
    if not any(int(row.get("total_queries") or 0) == expected_queries for row in global_rows):
        raise ValueError("Kaynak özetinde tamamlanmış sorgu hücresi bulunamadı.")
    return config, global_rows, catalog


def rank_top_models(
    config: dict[str, Any],
    summary: Sequence[dict[str, Any]],
    identities: dict[str, dict[str, Any]],
    *,
    limit: int = TOP_MODEL_COUNT,
) -> list[RankedModel]:
    expected_queries = int(config["max_queries"])
    rows_by_model: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in summary:
        if row.get("search_mode", "global") != "global":
            continue
        model_id = str(row.get("model_id", ""))
        if not model_id or model_id == core.RAW_MODEL_ID:
            continue
        key = (str(row.get("direction")), str(row.get("query_variant", "clean")))
        if key in rows_by_model[model_id]:
            raise ValueError(f"Tekrarlanan kaynak özet hücresi: model={model_id}, hücre={key}")
        rows_by_model[model_id][key] = dict(row)

    candidates: list[RankedModel] = []
    required_directions = configured_directions(config)
    for model_id, cells in rows_by_model.items():
        complete_directions: list[str] = []
        for direction in required_directions:
            clean = cells.get((direction, "clean"))
            hard = cells.get((direction, "hard_v1"))
            if clean is None or hard is None:
                continue
            if (
                int(clean.get("total_queries") or 0) == expected_queries
                and int(hard.get("total_queries") or 0) == expected_queries
            ):
                complete_directions.append(direction)
        if tuple(complete_directions) != required_directions:
            continue
        identity = identities.get(model_id)
        if identity is None:
            raise ValueError(f"Model kimliği katalogda/sonuçlarda bulunamadı: {model_id}")
        sha = str(identity.get("sha256") or identity.get("model_sha256") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            model_path = Path(str(identity.get("path") or ""))
            if not model_path.is_file():
                raise ValueError(f"Model SHA256 ve dosyası bulunamadı: {model_id}")
            sha = core.sha256_file(model_path)

        clean_rows = [cells[(direction, "clean")] for direction in complete_directions]
        hard_rows = [cells[(direction, "hard_v1")] for direction in complete_directions]
        clean_s25 = mean(required_float(row, "success_25m") for row in clean_rows)
        hard_s25 = mean(required_float(row, "success_25m") for row in hard_rows)
        clean_auc = mean(required_float(row, "auc_25m") for row in clean_rows)
        hard_auc = mean(required_float(row, "auc_25m") for row in hard_rows)
        clean_s5 = mean(required_float(row, "success_5m") for row in clean_rows)
        hard_s5 = mean(required_float(row, "success_5m") for row in hard_rows)
        medians = [
            row.get("median_error_under_25m") for row in [*clean_rows, *hard_rows]
        ]
        combined_median = mean(medians) if all(value is not None for value in medians) else None
        assert None not in {clean_s25, hard_s25, clean_auc, hard_auc, clean_s5, hard_s5}
        candidates.append(
            RankedModel(
                rank=0,
                model_id=model_id,
                model_file=str(identity.get("path") or ""),
                model_sha256=sha.lower(),
                source_series_id=str(identity.get("series_id") or ""),
                source_series_key=str(identity.get("series_key") or ""),
                source_relative_path=str(identity.get("relative_path") or Path(str(identity.get("path") or model_id)).name),
                source_checkpoint_number=(
                    int(identity["checkpoint_number"])
                    if identity.get("checkpoint_number") is not None
                    else None
                ),
                completed_directions=tuple(complete_directions),
                clean_success_25m=float(clean_s25),
                hard_success_25m=float(hard_s25),
                mean_success_25m=float((clean_s25 + hard_s25) / 2.0),
                clean_auc_25m=float(clean_auc),
                hard_auc_25m=float(hard_auc),
                mean_auc_25m=float((clean_auc + hard_auc) / 2.0),
                clean_success_5m=float(clean_s5),
                hard_success_5m=float(hard_s5),
                mean_success_5m=float((clean_s5 + hard_s5) / 2.0),
                median_error_under_25m=combined_median,
                source_duplicate_paths=tuple(
                    str(path) for path in identity.get("duplicate_paths", [])
                ),
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.mean_success_25m,
            -item.mean_auc_25m,
            -item.mean_success_5m,
            item.median_error_under_25m
            if item.median_error_under_25m is not None
            else math.inf,
            item.model_id.casefold(),
        )
    )
    if len(candidates) < limit:
        raise ValueError(f"En az {limit} tamamlanmış model gerekli; bulunan={len(candidates)}")
    return [
        RankedModel(**{**asdict(item), "rank": rank})
        for rank, item in enumerate(candidates[:limit], start=1)
    ]


def normalized_epoch_stem(path: Path) -> tuple[int, str] | None:
    stem = path.stem
    if NON_EPOCH_RE.search(stem):
        return None
    matches = list(EPOCH_RE.finditer(stem))
    if len(matches) != 1:
        return None
    match = matches[0]
    epoch = int(match.group("number"))
    normalized = stem[: match.start("number")] + "{EPOCH}" + stem[match.end("number") :]
    return epoch, unicodedata.normalize("NFC", normalized).casefold()


def archive_lineage_key(relative_parent: Path, normalized_stem: str) -> str:
    parent = unicodedata.normalize("NFC", relative_parent.as_posix()).casefold()
    return f"{parent}/{normalized_stem}"


def lineage_id_for_key(key: str) -> str:
    readable = core.safe_slug(key.replace("/", "__"), max_len=66)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{digest}"


def scan_epoch_archive(
    archive_root: Path,
) -> tuple[dict[str, list[ArchiveCheckpoint]], dict[str, set[str]]]:
    archive_root = archive_root.resolve()
    groups: dict[str, list[ArchiveCheckpoint]] = defaultdict(list)
    sha_to_lineages: dict[str, set[str]] = defaultdict(set)
    candidates = sorted(
        (
            path
            for path in archive_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in ALLOWED_MODEL_SUFFIXES
        ),
        key=lambda path: path.relative_to(archive_root).as_posix().casefold(),
    )
    for path in candidates:
        parsed = normalized_epoch_stem(path)
        if parsed is None:
            continue
        epoch, normalized_stem = parsed
        relative = path.relative_to(archive_root)
        key = archive_lineage_key(relative.parent, normalized_stem)
        sha = core.sha256_file(path)
        checkpoint = ArchiveCheckpoint(
            path=path.resolve(),
            relative_path=relative.as_posix(),
            epoch=epoch,
            sha256=sha,
            normalized_stem=normalized_stem,
            lineage_key=key,
        )
        groups[key].append(checkpoint)
        sha_to_lineages[sha].add(key)
    for checkpoints in groups.values():
        checkpoints.sort(key=lambda item: (item.epoch, item.relative_path.casefold()))
    if not groups:
        raise ValueError(f"Arşivde epoch checkpoint bulunamadı: {archive_root}")
    return dict(groups), dict(sha_to_lineages)


def _lineage_pairs(checkpoints: Sequence[ArchiveCheckpoint]) -> set[tuple[int, str]]:
    return {(item.epoch, item.sha256) for item in checkpoints}


def _validate_epoch_collisions(
    lineage_key: str, checkpoints: Sequence[ArchiveCheckpoint]
) -> None:
    by_epoch: dict[int, set[str]] = defaultdict(set)
    for checkpoint in checkpoints:
        by_epoch[checkpoint.epoch].add(checkpoint.sha256)
    conflicts = {
        epoch: sorted(shas) for epoch, shas in by_epoch.items() if len(shas) > 1
    }
    if conflicts:
        raise ValueError(
            "Aynı lineage ve epoch için farklı SHA256 bulundu; sessiz seçim yapılmadı: "
            f"lineage={lineage_key}, çakışmalar={conflicts}"
        )


def resolve_archive_lineage(
    ranked: RankedModel,
    groups: dict[str, list[ArchiveCheckpoint]],
    sha_to_lineages: dict[str, set[str]],
) -> tuple[str, list[str]]:
    contenders = sorted(sha_to_lineages.get(ranked.model_sha256, set()))
    if not contenders:
        raise ValueError(
            "Seçilen modelin SHA256 ankrajı için epoch lineage bulunamadı: "
            f"rank={ranked.rank}, model={ranked.model_id}, sha={ranked.model_sha256}"
        )
    for key in contenders:
        _validate_epoch_collisions(key, groups[key])
    if len(contenders) == 1:
        return contenders[0], []

    pair_sets = {key: _lineage_pairs(groups[key]) for key in contenders}
    source_path = Path(ranked.model_file).resolve() if ranked.model_file else None
    exact_path_keys = []
    if source_path is not None:
        exact_path_keys = [
            key
            for key in contenders
            if any(item.path == source_path for item in groups[key])
        ]
    documented_paths = {
        Path(path).as_posix().casefold()
        for path in (
            ranked.source_relative_path,
            *ranked.source_duplicate_paths,
        )
        if path
    }
    documented_keys = {
        key
        for key in contenders
        if any(
            item.relative_path.casefold() in documented_paths
            for item in groups[key]
        )
    }
    evidence_keys = sorted(documented_keys.union(exact_path_keys))
    dominant = [
        key
        for key in evidence_keys
        if len(pair_sets[key]) > 1
        if all(pair_sets[other] <= pair_sets[key] for other in evidence_keys)
        and any(
            pair_sets[other] < pair_sets[key]
            for other in evidence_keys
            if other != key
        )
        and all(
            groups[other][0].normalized_stem == groups[key][0].normalized_stem
            for other in evidence_keys
        )
    ]
    if len(dominant) == 1:
        chosen = dominant[0]
        aliases = [
            key
            for key in evidence_keys
            if key != chosen and pair_sets[key] <= pair_sets[chosen]
        ]
        return chosen, aliases
    if len(exact_path_keys) == 1:
        return exact_path_keys[0], []
    if len(evidence_keys) == 1:
        return evidence_keys[0], []

    raise ValueError(
        "Aynı model SHA256 için birden çok bağımsız/eşit lineage bulundu; kaynak klasör "
        f"kimliği korunamadığı için seçim yapılmadı: model={ranked.model_id}, "
        f"lineages={contenders}"
    )


def resolve_selected_lineages(
    ranked_models: Sequence[RankedModel],
    groups: dict[str, list[ArchiveCheckpoint]],
    sha_to_lineages: dict[str, set[str]],
) -> list[ResolvedLineage]:
    resolved_by_key: dict[str, ResolvedLineage] = {}
    for ranked in ranked_models:
        key, aliases = resolve_archive_lineage(ranked, groups, sha_to_lineages)
        lineage = resolved_by_key.get(key)
        if lineage is None:
            lineage = ResolvedLineage(
                lineage_key=key,
                lineage_id=lineage_id_for_key(key),
                checkpoints=list(groups[key]),
                mirror_alias_keys=list(aliases),
            )
            resolved_by_key[key] = lineage
        lineage.selected_models.append(ranked)
        lineage.mirror_alias_keys = sorted(set(lineage.mirror_alias_keys).union(aliases))
    # The top ten is selected first. Repeated lineages collapse here and are not
    # backfilled from rank 11 onward.
    return sorted(resolved_by_key.values(), key=lambda item: item.first_rank)


def lineage_checkpoint_plan(lineage: ResolvedLineage) -> list[dict[str, Any]]:
    by_epoch: dict[int, list[ArchiveCheckpoint]] = defaultdict(list)
    for checkpoint in lineage.checkpoints:
        by_epoch[checkpoint.epoch].append(checkpoint)
    plan: list[dict[str, Any]] = []
    canonical_epoch_by_sha: dict[str, int] = {}
    for epoch in sorted(by_epoch):
        candidates = by_epoch[epoch]
        shas = {item.sha256 for item in candidates}
        if len(shas) != 1:
            _validate_epoch_collisions(lineage.lineage_key, lineage.checkpoints)
        sha = next(iter(shas))
        canonical_epoch = canonical_epoch_by_sha.setdefault(sha, epoch)
        plan.append(
            {
                "epoch": epoch,
                "sha256": sha,
                "canonical_epoch": canonical_epoch,
                "duplicate_sha_alias": epoch != canonical_epoch,
                "source_paths": [str(item.path) for item in candidates],
                "source_relative_paths": [item.relative_path for item in candidates],
                "original_filenames": [item.path.name for item in candidates],
                "source_path": str(candidates[0].path),
                "original_filename": candidates[0].path.name,
                "suffix": candidates[0].path.suffix.casefold(),
            }
        )
    return plan


def plan_payload(lineages: Sequence[ResolvedLineage]) -> dict[str, Any]:
    rows = []
    for lineage in lineages:
        checkpoints = lineage_checkpoint_plan(lineage)
        rows.append(
            {
                "training_lineage": lineage.lineage_id,
                "lineage_key": lineage.lineage_key,
                "lineage_rank": lineage.first_rank,
                "selected_models": [asdict(item) for item in sorted(lineage.selected_models, key=lambda item: item.rank)],
                "mirror_alias_keys": lineage.mirror_alias_keys,
                "checkpoints": checkpoints,
                "checkpoint_count": len(checkpoints),
                "unique_sha256_count": len({item["sha256"] for item in checkpoints}),
            }
        )
    return {
        "top_model_count": TOP_MODEL_COUNT,
        "unique_lineage_count": len(rows),
        "unique_model_sha256_count": len(
            {item["sha256"] for row in rows for item in row["checkpoints"]}
        ),
        "lineages": rows,
    }


def print_plan(ranked_models: Sequence[RankedModel], lineages: Sequence[ResolvedLineage]) -> None:
    print("İlk 10 model:")
    for item in ranked_models:
        print(
            f"  {item.rank:02d}. {item.model_id} | mean_success_25m="
            f"{item.mean_success_25m:.6f} | sha={item.model_sha256[:12]}"
        )
    print(f"Benzersiz training lineage: {len(lineages)} (top 10 sonrası tekilleştirildi)")
    total_logical = 0
    unique_shas: set[str] = set()
    for index, lineage in enumerate(lineages, start=1):
        checkpoints = lineage_checkpoint_plan(lineage)
        total_logical += len(checkpoints)
        unique_shas.update(item["sha256"] for item in checkpoints)
        epochs = ", ".join(str(item["epoch"]) for item in checkpoints)
        print(
            f"  {index:02d}. {lineage.lineage_id} | seçilen_rank="
            f"{','.join(str(item.rank) for item in lineage.selected_models)} | "
            f"epoch={len(checkpoints)} [{epochs}]"
        )
    print(
        f"Toplam mantıksal epoch checkpoint: {total_logical}; "
        f"SHA tekilleştirmesi sonrası test modeli: {len(unique_shas)}"
    )


def _atomic_copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = core.sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Staging dosyası beklenen SHA256 ile uyuşmuyor: {destination}; "
                f"beklenen={expected_sha256}, görülen={actual}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        actual = core.sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Kopyalanan checkpoint SHA256 doğrulaması başarısız: {source}; "
                f"beklenen={expected_sha256}, görülen={actual}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_staging_targets(
    lineages: Sequence[ResolvedLineage], staging_root: Path
) -> dict[Path, str]:
    targets_by_sha: dict[str, Path] = {}
    expected: dict[Path, str] = {}
    for index, lineage in enumerate(lineages, start=1):
        folder = staging_root / f"{index:02d}_{lineage.lineage_id}"
        for entry in lineage_checkpoint_plan(lineage):
            sha = str(entry["sha256"])
            target = targets_by_sha.get(sha)
            if target is None:
                target = folder / f"epoch_{int(entry['canonical_epoch']):05d}{entry['suffix']}"
                targets_by_sha[sha] = target
            expected[target.resolve()] = sha
    return expected


def _validate_staging_model_allowlist(
    staging_root: Path,
    expected: dict[Path, str],
    *,
    require_all: bool,
) -> None:
    actual: dict[Path, Path] = {}
    if staging_root.is_dir():
        for path in staging_root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in ALLOWED_MODEL_SUFFIXES:
                if path.is_symlink():
                    raise RuntimeError(f"Staging model symlink olamaz: {path}")
                actual[path.resolve()] = path
    unexpected = sorted(set(actual).difference(expected), key=str)
    if unexpected:
        shown = ", ".join(str(path) for path in unexpected[:5])
        raise RuntimeError(
            "Staging klasöründe güncel epoch planına ait olmayan model dosyaları var; "
            f"benchmark başlatılmadı: {shown}"
        )
    missing = sorted(set(expected).difference(actual), key=str)
    if require_all and missing:
        shown = ", ".join(str(path) for path in missing[:5])
        raise RuntimeError(f"Staging checkpoint eksik: {shown}")
    for resolved_path, source_path in actual.items():
        expected_sha = expected[resolved_path]
        actual_sha = core.sha256_file(source_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Staging checkpoint SHA256 değişmiş: {source_path}; "
                f"beklenen={expected_sha}, görülen={actual_sha}"
            )


def prepare_staging(
    lineages: Sequence[ResolvedLineage], staging_root: Path
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    expected_targets = _expected_staging_targets(lineages, staging_root)
    target_by_sha = {sha: path for path, sha in expected_targets.items()}
    source_by_sha: dict[str, Path] = {}
    planned_lineages: list[dict[str, Any]] = []
    expected_manifest_paths: set[Path] = set()
    for index, lineage in enumerate(lineages, start=1):
        folder = staging_root / f"{index:02d}_{lineage.lineage_id}"
        logical_entries = lineage_checkpoint_plan(lineage)
        for entry in logical_entries:
            sha = str(entry["sha256"])
            target = target_by_sha[sha]
            source_by_sha.setdefault(sha, Path(str(entry["source_path"])))
            entry["copied_file"] = str(target)
            entry["copied_relative_path"] = target.relative_to(staging_root).as_posix()
        manifest = {
            "schema_version": SWEEP_SCHEMA_VERSION,
            "training_lineage": lineage.lineage_id,
            "lineage_key": lineage.lineage_key,
            "lineage_rank": lineage.first_rank,
            "selected_models": [
                asdict(item)
                for item in sorted(lineage.selected_models, key=lambda item: item.rank)
            ],
            "mirror_alias_keys": lineage.mirror_alias_keys,
            "checkpoints": logical_entries,
            "checkpoint_count": len(logical_entries),
            "unique_sha256_count": len(
                {item["sha256"] for item in logical_entries}
            ),
        }
        manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
        manifest_path = (folder / "lineage_manifest.json").resolve()
        expected_manifest_paths.add(manifest_path)
        if manifest_path.exists() and load_json(manifest_path) != manifest:
            raise RuntimeError(
                "Mevcut lineage_manifest.json yeni arşiv planıyla uyuşmuyor; "
                f"hiçbir staging dosyası değiştirilmedi: {manifest_path}"
            )
        planned_lineages.append(
            {**manifest, "manifest_file": str(manifest_path)}
        )
    actual_manifests = (
        {path.resolve() for path in staging_root.rglob("lineage_manifest.json")}
        if staging_root.is_dir()
        else set()
    )
    unexpected_manifests = sorted(actual_manifests - expected_manifest_paths, key=str)
    if unexpected_manifests:
        raise RuntimeError(
            "Staging klasöründe güncel plana ait olmayan lineage manifesti var; "
            f"hiçbir dosya değiştirilmedi: {unexpected_manifests[0]}"
        )
    # Fail before copying anything if a shared staging root contains stale or
    # tampered model files from another sweep plan.
    _validate_staging_model_allowlist(
        staging_root, expected_targets, require_all=False
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    for target, sha in expected_targets.items():
        _atomic_copy_verified(source_by_sha[sha], target, sha)
    for lineage in planned_lineages:
        manifest_path = Path(str(lineage["manifest_file"]))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            core.atomic_write_json(
                manifest_path,
                {key: value for key, value in lineage.items() if key != "manifest_file"},
            )

    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "staging_root": str(staging_root),
        "unique_staged_model_count": len(expected_targets),
        "lineages": planned_lineages,
    }


def load_existing_staging(
    lineages: Sequence[ResolvedLineage], staging_root: Path
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    loaded: list[dict[str, Any]] = []
    for index, lineage in enumerate(lineages, start=1):
        folder = staging_root / f"{index:02d}_{lineage.lineage_id}"
        manifest_path = folder / "lineage_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Resume reddedildi: staging lineage manifesti eksik: {manifest_path}"
            )
        manifest = load_json(manifest_path)
        if manifest.get("training_lineage") != lineage.lineage_id or manifest.get("lineage_key") != lineage.lineage_key:
            raise RuntimeError(f"Resume staging lineage kimliği değişmiş: {manifest_path}")
        expected = lineage_checkpoint_plan(lineage)
        observed_pairs = [
            (int(item["epoch"]), str(item["sha256"]))
            for item in manifest.get("checkpoints", [])
        ]
        expected_pairs = [(int(item["epoch"]), str(item["sha256"])) for item in expected]
        if observed_pairs != expected_pairs:
            raise RuntimeError(f"Resume staging checkpoint inventory değişmiş: {manifest_path}")
        loaded.append({**manifest, "manifest_file": str(manifest_path)})
    payload = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "staging_root": str(staging_root),
        "unique_staged_model_count": len(
            {item["sha256"] for lineage in loaded for item in lineage.get("checkpoints", [])}
        ),
        "lineages": loaded,
    }
    validate_staging(payload)
    return payload


def validate_staging(staging_payload: dict[str, Any]) -> None:
    expected_models: dict[Path, str] = {}
    for lineage in staging_payload.get("lineages", []):
        manifest_path = Path(str(lineage["manifest_file"]))
        if load_json(manifest_path) != {key: value for key, value in lineage.items() if key != "manifest_file"}:
            raise RuntimeError(f"Staging lineage manifesti değişmiş: {manifest_path}")
        for checkpoint in lineage.get("checkpoints", []):
            copied = Path(str(checkpoint["copied_file"]))
            resolved = copied.resolve()
            sha = str(checkpoint["sha256"])
            previous_sha = expected_models.setdefault(resolved, sha)
            if previous_sha != sha:
                raise RuntimeError(
                    f"Aynı staging dosyasına birden çok SHA256 atanmış: {copied}"
                )
    _validate_staging_model_allowlist(
        Path(str(staging_payload["staging_root"])).resolve(),
        expected_models,
        require_all=True,
    )


def _selected_lineage_rows(staging_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineage in staging_payload.get("lineages", []):
        selected = sorted(lineage.get("selected_models", []), key=lambda item: int(item["rank"]))
        rows.append(
            {
                "training_lineage": lineage["training_lineage"],
                "lineage_rank": lineage["lineage_rank"],
                "lineage_key": lineage["lineage_key"],
                "selected_top10_ranks": ";".join(str(item["rank"]) for item in selected),
                "selected_model_ids": ";".join(str(item["model_id"]) for item in selected),
                "selected_model_sha256": ";".join(str(item["model_sha256"]) for item in selected),
                "epoch_checkpoint_count": lineage["checkpoint_count"],
                "unique_sha256_count": lineage["unique_sha256_count"],
                "lineage_manifest": lineage["manifest_file"],
                "mirror_alias_keys": ";".join(lineage.get("mirror_alias_keys", [])),
            }
        )
    return rows


def write_selected_lineages(output_dir: Path, staging_payload: dict[str, Any]) -> None:
    rows = _selected_lineage_rows(staging_payload)
    columns = list(rows[0]) if rows else ["training_lineage"]
    core.csv_write(output_dir / "selected_lineages.csv", rows, columns)


def scientific_contract(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if int(config.get("scientific_semantics_version", -1)) != int(
        core.SCIENTIFIC_SEMANTICS_VERSION
    ):
        raise ValueError(
            "Kaynak scientific semantics sürümü güncel benchmark ile uyuşmuyor: "
            f"kaynak={config.get('scientific_semantics_version')!r}, "
            f"güncel={core.SCIENTIFIC_SEMANTICS_VERSION}"
        )
    if config.get("hard_v1_profile") != core.HARD_V1_PROFILE:
        raise ValueError(
            "Kaynak hard_v1 profili güncel benchmark profiliyle uyuşmuyor."
        )
    expected = {
        "tile_size": 544,
        "crop_border": 16,
        "normalization": "minus1_1",
        "top_k": 30,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Kaynak bilimsel ayarı sweep sözleşmesiyle uyuşmuyor: "
                f"{key}={config.get(key)!r}, beklenen={value!r}"
            )
    factors = tuple(int(value) for value in config.get("pyramid_factors", []))
    if factors != (16, 8, 4, 2, 1):
        raise ValueError(f"Kaynak pyramid_factors değişmiş: {factors}")
    overlap = int(config.get("overlap", 32))
    if overlap != 32 or int(config["tile_size"]) - 2 * int(config["crop_border"]) != 512:
        raise ValueError("544/32/16 -> 512 tile/template sözleşmesi bozulmuş.")

    def raster_identity(value: Any) -> dict[str, Any]:
        path = Path(str(value)).resolve()
        stat = path.stat()
        return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    return {
        "scientific_semantics_version": core.SCIENTIFIC_SEMANTICS_VERSION,
        "query_manifest_schema_version": core.QUERY_MANIFEST_SCHEMA_VERSION,
        "query_raster": raster_identity(config["query_raster"]),
        "map_raster": raster_identity(config["map_raster"]),
        "bidirectional": bool(config.get("bidirectional", True)),
        "query_variants": ["clean", "hard_v1"],
        "search_modes": [{"name": "global", "roi_radius_m": None}],
        "tile_size": 544,
        "overlap": 32,
        "crop_border": 16,
        "template_size": 512,
        "normalization": "minus1_1",
        "pyramid_factors": [16, 8, 4, 2, 1],
        "top_k": 30,
        "matching_method": "TM_CCOEFF_NORMED",
        "query_sampling": "balanced_exact",
        "max_queries": SWEEP_QUERY_COUNT,
        "block_size_m": float(config.get("block_size_m", 1000.0)),
        "samples_per_block_source": int(config.get("samples_per_block", 5)),
        "seed": int(config.get("seed", 42)),
        "min_query_std": float(config.get("min_query_std", 12.0)),
        "min_query_entropy": float(config.get("min_query_entropy", 4.0)),
        "max_dark_fraction": float(config.get("max_dark_fraction", 0.20)),
        "query_edge_buffer_m": config.get("query_edge_buffer_m"),
        "refine_radius_px": int(config.get("refine_radius_px", 160)),
        "nms_radius_px": int(config.get("nms_radius_px", 256)),
        "output_value_mode": str(config.get("output_value_mode", "auto")),
        "enhancement": str(config.get("enhancement", "none")),
        "hard_v1_profile_sha256": canonical_sha256(core.HARD_V1_PROFILE),
        "model_plan_sha256": canonical_sha256(plan),
        "raw_baseline": True,
        "model_applied_to_query_and_map": True,
    }


def _scientific_manifest_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scientific_manifest_value(item)
            for key, item in value.items()
            if key not in {"created_at_utc", "raw_tile_file"}
        }
    if isinstance(value, list):
        return [_scientific_manifest_value(item) for item in value]
    return value


def manifest_fingerprint(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest nesne olmalıdır: {path}")
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError(f"Manifest queries listesi geçersiz: {path}")
    tile_hashes: list[dict[str, str]] = []
    for row in queries:
        if not isinstance(row, dict):
            raise ValueError(f"Manifest sorgu satırı geçersiz: {path}")
        tile_path = Path(str(row.get("raw_tile_file", "")))
        if not tile_path.is_file():
            raise RuntimeError(f"Manifest sorgu tile dosyası eksik: {tile_path}")
        tile_hashes.append(
            {
                "query_id": str(row.get("query_id", "")),
                "sha256": core.sha256_file(tile_path),
            }
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": core.sha256_file(path),
        "scientific_sha256": canonical_sha256(_scientific_manifest_value(payload)),
        "tiles_sha256": canonical_sha256(tile_hashes),
        "query_count": len(queries),
    }


def _record_centre(record: core.QueryRecord) -> tuple[Any, ...]:
    return (
        record.query_id,
        record.block_id,
        record.center_easting_m,
        record.center_northing_m,
        record.source_row,
        record.source_col,
    )


def _verify_manifest_tiles(path: Path) -> None:
    payload = load_json(path)
    queries = payload.get("queries", [])
    ids = [str(row.get("query_id")) for row in queries]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Manifestte tekrarlanan query_id bulundu: {path}")
    missing = [
        str(row.get("raw_tile_file"))
        for row in queries
        if not Path(str(row.get("raw_tile_file", ""))).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Manifestte {len(missing)} sorgu tile dosyası eksik: {path}; "
            "sonuçlar varken otomatik yeniden üretim yapılmadı."
        )


def direction_specs(config: dict[str, Any]) -> list[tuple[Path, Path]]:
    query = Path(str(config["query_raster"])).resolve()
    search_map = Path(str(config["map_raster"])).resolve()
    specs = [(query, search_map)]
    if bool(config.get("bidirectional", True)):
        specs.append((search_map, query))
    return specs


def prepare_query_manifest_index(
    output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    direction_entries: list[dict[str, Any]] = []
    for query_raster, map_raster in direction_specs(config):
        direction = core.direction_name(query_raster, map_raster)
        queries_dir = output_dir / core.safe_slug(direction) / "queries"
        records = core.generate_query_manifest(
            query_raster,
            map_raster,
            queries_dir,
            tile_size=544,
            samples_per_block=int(config.get("samples_per_block", 5)),
            block_size_m=float(config.get("block_size_m", 1000.0)),
            max_queries=SWEEP_QUERY_COUNT,
            seed=int(config.get("seed", 42)),
            min_std=float(config.get("min_query_std", 12.0)),
            min_entropy=float(config.get("min_query_entropy", 4.0)),
            max_dark_fraction=float(config.get("max_dark_fraction", 0.20)),
            edge_buffer_m=config.get("query_edge_buffer_m"),
            force=False,
            sampling_strategy="balanced_exact",
        )
        if len(records) != SWEEP_QUERY_COUNT:
            raise RuntimeError(
                f"{direction} için tam {SWEEP_QUERY_COUNT} sorgu gerekli; "
                f"üretilen={len(records)}"
            )
        variants = core.prepare_query_variants(
            records,
            queries_dir,
            variants=("clean", "hard_v1"),
            seed=int(config.get("seed", 42)),
            force=False,
        )
        clean_centres = [_record_centre(item) for item in variants["clean"]]
        hard_centres = [_record_centre(item) for item in variants["hard_v1"]]
        if clean_centres != hard_centres:
            raise RuntimeError(f"clean ve hard_v1 merkezleri ayrıştı: {direction}")
        base_manifest = queries_dir / "query_manifest.json"
        hard_manifest = queries_dir / "variants" / "hard_v1" / "query_variant_manifest.json"
        block_counts: dict[str, int] = defaultdict(int)
        for record in records:
            block_counts[record.block_id] += 1
        direction_entries.append(
            {
                "direction": direction,
                "query_raster": str(query_raster),
                "map_raster": str(map_raster),
                "query_count": len(records),
                "block_count": len(block_counts),
                "min_queries_per_block": min(block_counts.values()),
                "max_queries_per_block": max(block_counts.values()),
                "centres_sha256": canonical_sha256(clean_centres),
                "base_manifest": manifest_fingerprint(base_manifest),
                "hard_v1_manifest": manifest_fingerprint(hard_manifest),
            }
        )
    index = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "max_queries_per_direction": SWEEP_QUERY_COUNT,
        "seed": int(config.get("seed", 42)),
        "sampling_strategy": "balanced_exact",
        "query_variants": ["clean", "hard_v1"],
        "directions": direction_entries,
    }
    core.atomic_write_json(output_dir / "query_manifest_2000.json", index)
    return index


def validate_query_manifest_index(output_dir: Path) -> dict[str, Any]:
    index_path = output_dir / "query_manifest_2000.json"
    index = load_json(index_path)
    if index.get("max_queries_per_direction") != SWEEP_QUERY_COUNT:
        raise RuntimeError("query_manifest_2000.json sorgu sayısı değişmiş.")
    if index.get("sampling_strategy") != "balanced_exact":
        raise RuntimeError("query_manifest_2000.json örnekleme stratejisi değişmiş.")
    for direction in index.get("directions", []):
        if int(direction.get("query_count", 0)) != SWEEP_QUERY_COUNT:
            raise RuntimeError(f"Yön manifesti 2000 sorgu içermiyor: {direction}")
        for key in ("base_manifest", "hard_v1_manifest"):
            recorded = direction[key]
            path = Path(str(recorded["path"]))
            current = manifest_fingerprint(path)
            if current != recorded:
                raise RuntimeError(
                    "Resume reddedildi: sorgu manifesti içeriği değişmiş: "
                    f"{path}; kayıtlı={recorded}, güncel={current}"
                )
            _verify_manifest_tiles(path)
    return index


def source_fingerprint(run_dir: Path) -> dict[str, Any]:
    files = [run_dir / "run_config.json", run_dir / "summary.json"]
    for optional in ("model_catalog.json", "summary_metadata.json"):
        candidate = run_dir / optional
        if candidate.is_file():
            files.append(candidate)
    return {
        "run_dir": str(run_dir.resolve()),
        "files": {path.name: core.sha256_file(path) for path in files},
    }


def sweep_resume_identity(
    *,
    source: dict[str, Any],
    contract: dict[str, Any],
    plan: dict[str, Any],
    staging_payload: dict[str, Any],
    query_index: dict[str, Any],
) -> dict[str, Any]:
    query_science = {
        "max_queries_per_direction": query_index["max_queries_per_direction"],
        "seed": query_index["seed"],
        "sampling_strategy": query_index["sampling_strategy"],
        "query_variants": query_index["query_variants"],
        "directions": [
            {
                "direction": item["direction"],
                "query_raster": item["query_raster"],
                "map_raster": item["map_raster"],
                "query_count": item["query_count"],
                "centres_sha256": item["centres_sha256"],
                "base_scientific_sha256": item["base_manifest"]["scientific_sha256"],
                "hard_scientific_sha256": item["hard_v1_manifest"]["scientific_sha256"],
                "base_tiles_sha256": item["base_manifest"]["tiles_sha256"],
                "hard_tiles_sha256": item["hard_v1_manifest"]["tiles_sha256"],
            }
            for item in query_index["directions"]
        ],
    }
    staged_inventory = [
        {
            "training_lineage": lineage["training_lineage"],
            "checkpoints": [
                {
                    "epoch": item["epoch"],
                    "sha256": item["sha256"],
                    "copied_relative_path": item["copied_relative_path"],
                }
                for item in lineage["checkpoints"]
            ],
        }
        for lineage in staging_payload["lineages"]
    ]
    return {
        "source": source,
        "scientific_contract": contract,
        "model_plan_sha256": canonical_sha256(plan),
        "staged_inventory": staged_inventory,
        "query_manifests": query_science,
    }


def write_or_validate_sweep_config(
    output_dir: Path,
    identity: dict[str, Any],
    *,
    results_exist: bool,
) -> dict[str, Any]:
    path = output_dir / "epoch_sweep_config.json"
    identity_sha = canonical_sha256(identity)
    if path.is_file():
        previous = load_json(path)
        if previous.get("resume_identity_sha256") != identity_sha or previous.get("resume_identity") != identity:
            message = (
                "Epoch sweep resume reddedildi: kaynak sonuç, model inventory, sorgu "
                "manifesti veya bilimsel yapılandırma değişmiş. Yeni bir --output-dir kullanın."
            )
            if results_exist:
                raise RuntimeError(message)
            raise RuntimeError(message + " Mevcut hazırlık dosyalarının üzerine yazılmadı.")
        return previous
    if results_exist:
        raise RuntimeError(
            "results.jsonl var ancak epoch_sweep_config.json yok; güvenli resume yapılamaz."
        )
    payload = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "resume_identity": identity,
        "resume_identity_sha256": identity_sha,
        "status": "prepared",
        "status_history": [{"status": "prepared", "at_utc": utc_now_iso()}],
    }
    core.atomic_write_json(path, payload)
    return payload


def update_sweep_status(output_dir: Path, status: str, **details: Any) -> None:
    path = output_dir / "epoch_sweep_config.json"
    payload = load_json(path)
    payload["status"] = status
    payload["status_history"] = [
        *payload.get("status_history", []),
        {"status": status, "at_utc": utc_now_iso(), **details},
    ]
    core.atomic_write_json(path, payload)


def build_core_argv(
    *,
    output_dir: Path,
    staging_root: Path,
    source_config: dict[str, Any],
    batch_size: int,
    search_workers: int,
    keep_maps: bool,
    fail_fast: bool,
) -> list[str]:
    argv = [
        "--query-raster", str(Path(str(source_config["query_raster"])).resolve()),
        "--map-raster", str(Path(str(source_config["map_raster"])).resolve()),
        "--model-dir", str(staging_root.resolve()),
        "--model-sampling", "full",
        "--resume-run", str(output_dir.resolve()),
        "--include-raw", "--include-models",
        "--query-variants", "clean,hard_v1",
        "--search-modes", "global",
        "--tile-size", "544", "--overlap", "32", "--crop-border", "16",
        "--block-size-m", str(float(source_config.get("block_size_m", 1000.0))),
        "--samples-per-block", str(int(source_config.get("samples_per_block", 5))),
        "--max-queries", str(SWEEP_QUERY_COUNT),
        "--query-sampling", "balanced_exact",
        "--seed", str(int(source_config.get("seed", 42))),
        "--min-query-std", str(float(source_config.get("min_query_std", 12.0))),
        "--min-query-entropy", str(float(source_config.get("min_query_entropy", 4.0))),
        "--max-dark-fraction", str(float(source_config.get("max_dark_fraction", 0.20))),
        "--pyramid-factors", "16,8,4,2,1", "--top-k", "30",
        "--refine-radius-px", str(int(source_config.get("refine_radius_px", 160))),
        "--nms-radius-px", str(int(source_config.get("nms_radius_px", 256))),
        "--batch-size", str(batch_size), "--search-workers", str(search_workers),
        "--bootstrap-iterations", str(int(source_config.get("bootstrap_iterations", 1000))),
        "--normalization", "minus1_1",
        "--output-value-mode", str(source_config.get("output_value_mode", "auto")),
        "--enhancement", str(source_config.get("enhancement", "none")),
        # Match the proven core benchmark checkpoint strategy: refresh the
        # incremental summary and Excel workbook after every completed model.
        "--excel-update", "model", "--excel-report", "--no-results-csv",
    ]
    edge_buffer = source_config.get("query_edge_buffer_m")
    if edge_buffer is not None:
        argv.extend(["--query-edge-buffer-m", str(float(edge_buffer))])
    argv.append("--bidirectional" if bool(source_config.get("bidirectional", True)) else "--no-bidirectional")
    if not keep_maps:
        argv.append("--cleanup-maps")
    if fail_fast:
        argv.append("--fail-fast")
    return argv


def _variant_metrics(
    summary_cells: dict[tuple[str, str, str], dict[str, Any]],
    *, model_id: str, variant: str, directions: Sequence[str],
) -> dict[str, float | None] | None:
    rows: list[dict[str, Any]] = []
    for direction in directions:
        row = summary_cells.get((model_id, direction, variant))
        if row is None or int(row.get("total_queries") or 0) != SWEEP_QUERY_COUNT:
            return None
        rows.append(row)
    medians = [row.get("median_error_under_25m") for row in rows]
    return {
        "success_25m": mean(required_float(row, "success_25m") for row in rows),
        "auc_25m": mean(required_float(row, "auc_25m") for row in rows),
        "success_5m": mean(required_float(row, "success_5m") for row in rows),
        "success_10m": mean(required_float(row, "success_10m") for row in rows),
        "median_error_under_25m": mean(medians) if all(value is not None for value in medians) else None,
    }


def pooled_medians_under_25m(
    results_path: Path,
    *,
    model_ids: set[str],
    directions: Sequence[str],
) -> dict[tuple[str, str], float | None]:
    direction_set = set(directions)
    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Geçersiz epoch sweep results JSONL: {results_path}:{line_number}"
                ) from exc
            model_id = str(row.get("model_id", ""))
            variant = str(row.get("query_variant", "clean"))
            if (
                model_id not in model_ids
                or variant not in {"clean", "hard_v1"}
                or str(row.get("direction")) not in direction_set
                or str(row.get("search_mode", "global")) != "global"
                or row.get("status") != "ok"
                or row.get("error_m") is None
            ):
                continue
            error_m = float(row["error_m"])
            if error_m <= 25.0:
                errors[(model_id, variant)].append(error_m)
    return {
        (model_id, variant): (
            float(statistics.median(errors.get((model_id, variant), [])))
            if errors.get((model_id, variant))
            else None
        )
        for model_id in model_ids
        for variant in ("clean", "hard_v1")
    }


def build_epoch_result_rows(
    output_dir: Path, staging_payload: dict[str, Any], source_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(output_dir / "summary.json")
    catalog = load_json(output_dir / "model_catalog.json")
    sha_to_model_id = {
        str(item["sha256"]): str(item["model_id"])
        for item in catalog.get("models", []) if item.get("sha256") and item.get("model_id")
    }
    summary_cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary:
        if row.get("search_mode", "global") != "global":
            continue
        key = (str(row.get("model_id")), str(row.get("direction")), str(row.get("query_variant", "clean")))
        summary_cells[key] = row
    directions = configured_directions(source_config)
    pooled_medians = pooled_medians_under_25m(
        output_dir / "results.jsonl",
        model_ids={*sha_to_model_id.values(), core.RAW_MODEL_ID},
        directions=directions,
    )
    result_rows: list[dict[str, Any]] = []
    for lineage in staging_payload.get("lineages", []):
        for checkpoint in lineage.get("checkpoints", []):
            sha = str(checkpoint["sha256"])
            model_id = sha_to_model_id.get(sha, "")
            clean = _variant_metrics(summary_cells, model_id=model_id, variant="clean", directions=directions) if model_id else None
            hard = _variant_metrics(summary_cells, model_id=model_id, variant="hard_v1", directions=directions) if model_id else None
            complete = clean is not None and hard is not None
            clean_s25 = clean["success_25m"] if clean else None
            hard_s25 = hard["success_25m"] if hard else None
            clean_auc = clean["auc_25m"] if clean else None
            hard_auc = hard["auc_25m"] if hard else None
            clean_s5 = clean["success_5m"] if clean else None
            hard_s5 = hard["success_5m"] if hard else None
            medians = [
                pooled_medians.get((model_id, "clean")) if complete else None,
                pooled_medians.get((model_id, "hard_v1")) if complete else None,
            ]
            result_rows.append({
                "training_lineage": lineage["training_lineage"], "lineage_rank": lineage["lineage_rank"],
                "epoch": checkpoint["epoch"], "model_file": checkpoint["copied_file"], "model_sha256": sha,
                "status": "complete" if complete else "incomplete_or_model_error",
                "clean_success_25m": clean_s25, "hard_success_25m": hard_s25,
                "mean_success_25m": ((float(clean_s25) + float(hard_s25)) / 2.0 if clean_s25 is not None and hard_s25 is not None else None),
                "clean_auc_25m": clean_auc, "hard_auc_25m": hard_auc,
                "mean_auc_25m": ((float(clean_auc) + float(hard_auc)) / 2.0 if clean_auc is not None and hard_auc is not None else None),
                "clean_success_5m": clean_s5, "hard_success_5m": hard_s5,
                "mean_success_5m": ((float(clean_s5) + float(hard_s5)) / 2.0 if clean_s5 is not None and hard_s5 is not None else None),
                "clean_success_10m": clean["success_10m"] if clean else None,
                "hard_success_10m": hard["success_10m"] if hard else None,
                "clean_hard_gap": (float(clean_s25) - float(hard_s25) if clean_s25 is not None and hard_s25 is not None else None),
                "clean_median_error_under_25m": medians[0], "hard_median_error_under_25m": medians[1],
                "mean_median_error_under_25m": mean(medians) if all(value is not None for value in medians) else None,
                "evaluated_model_id": model_id, "canonical_epoch": checkpoint["canonical_epoch"],
                "duplicate_sha_alias": checkpoint["duplicate_sha_alias"],
            })
    result_rows.sort(key=lambda row: (int(row["lineage_rank"]), int(row["epoch"])))

    raw_clean = _variant_metrics(summary_cells, model_id=core.RAW_MODEL_ID, variant="clean", directions=directions)
    raw_hard = _variant_metrics(summary_cells, model_id=core.RAW_MODEL_ID, variant="hard_v1", directions=directions)
    raw_rows: list[dict[str, Any]] = []
    if raw_clean is not None and raw_hard is not None:
        raw_rows.append({
            "model_id": core.RAW_MODEL_ID,
            "clean_success_25m": raw_clean["success_25m"], "hard_success_25m": raw_hard["success_25m"],
            "mean_success_25m": (float(raw_clean["success_25m"]) + float(raw_hard["success_25m"])) / 2.0,
            "clean_auc_25m": raw_clean["auc_25m"], "hard_auc_25m": raw_hard["auc_25m"],
            "clean_success_5m": raw_clean["success_5m"], "hard_success_5m": raw_hard["success_5m"],
            "clean_success_10m": raw_clean["success_10m"], "hard_success_10m": raw_hard["success_10m"],
        })
    return result_rows, raw_rows


def best_epoch_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "complete":
            grouped[str(row["training_lineage"])].append(dict(row))
    best: list[dict[str, Any]] = []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (
            -float(row["mean_success_25m"]), -float(row["mean_auc_25m"]), -float(row["mean_success_5m"]),
            float(row["clean_hard_gap"]),
            float(row["mean_median_error_under_25m"]) if row.get("mean_median_error_under_25m") is not None else math.inf,
            int(row["epoch"]),
        ))
        winner = candidates[0]
        winner["lineage_epoch_count_complete"] = len(candidates)
        winner["selection_priority"] = "mean_success_25m,mean_auc_25m,mean_success_5m,clean_hard_gap,median_error,epoch"
        best.append(winner)
    best.sort(key=lambda row: int(row["lineage_rank"]))
    return best


def write_epoch_plots(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["training_lineage"])].append(dict(row))
    for lineage, lineage_rows in sorted(grouped.items()):
        lineage_rows.sort(key=lambda row: int(row["epoch"]))
        epochs = [int(row["epoch"]) for row in lineage_rows]
        clean = [row.get("clean_success_25m", math.nan) for row in lineage_rows]
        hard = [row.get("hard_success_25m", math.nan) for row in lineage_rows]
        combined = [row.get("mean_success_25m", math.nan) for row in lineage_rows]
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        axis.plot(epochs, clean, marker="o", linewidth=1.5, label="clean success_25m")
        axis.plot(epochs, hard, marker="o", linewidth=1.5, label="hard_v1 success_25m")
        axis.plot(epochs, combined, marker="o", linewidth=2.0, label="mean success_25m")
        axis.set_title(lineage)
        axis.set_xlabel("epoch")
        axis.set_ylabel("success_25m")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / f"{core.safe_slug(lineage, max_len=100)}_success25.png", dpi=150)
        plt.close(figure)


def write_epoch_reports(
    output_dir: Path,
    staging_payload: dict[str, Any],
    source_config: dict[str, Any],
) -> tuple[int, int, int]:
    rows, raw_rows = build_epoch_result_rows(output_dir, staging_payload, source_config)
    core.csv_write(output_dir / "all_epoch_results.csv", rows, EPOCH_RESULT_COLUMNS)
    best = best_epoch_rows(rows)
    best_columns = list(best[0]) if best else [*EPOCH_RESULT_COLUMNS, "selection_priority"]
    core.csv_write(output_dir / "best_epoch_per_lineage.csv", best, best_columns)
    raw_columns = list(raw_rows[0]) if raw_rows else ["model_id"]
    core.csv_write(output_dir / "raw_baseline_results.csv", raw_rows, raw_columns)
    write_epoch_plots(output_dir, rows)
    return len(rows), sum(row.get("status") == "complete" for row in rows), len(best)


def classify_sweep_outcome(
    *,
    core_return_code: int,
    execution_error: BaseException | None,
    expected_epoch_rows: int,
    complete_epoch_rows: int,
) -> tuple[str, int]:
    if isinstance(execution_error, KeyboardInterrupt) or core_return_code == 130:
        return "interrupted", 130
    if execution_error is not None or core_return_code != 0:
        return "failed", core_return_code if core_return_code != 0 else 1
    if expected_epoch_rows < 1 or complete_epoch_rows < 1:
        return "failed", 1
    if complete_epoch_rows < expected_epoch_rows:
        return "complete_with_errors", 0
    return "complete", 0


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sync_epoch_errors(output_dir: Path, staging_payload: dict[str, Any]) -> int:
    aliases_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lineage in staging_payload.get("lineages", []):
        for checkpoint in lineage.get("checkpoints", []):
            aliases_by_sha[str(checkpoint["sha256"])].append(
                {
                    "training_lineage": lineage["training_lineage"],
                    "epoch": checkpoint["epoch"],
                    "canonical_epoch": checkpoint["canonical_epoch"],
                    "model_file": checkpoint["copied_file"],
                }
            )
    source = output_dir / "model_errors.jsonl"
    rows = core.read_jsonl(source) if source.is_file() else []
    enriched: list[dict[str, Any]] = []
    for row in rows:
        sha = str(row.get("model_sha256") or "")
        enriched.append({**row, "epoch_aliases": aliases_by_sha.get(sha, [])})
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched)
    _atomic_write_text(output_dir / "epoch_sweep_errors.jsonl", text)
    return len(enriched)


def write_top_models(output_dir: Path, ranked_models: Sequence[RankedModel]) -> None:
    rows = [asdict(item) for item in ranked_models]
    columns = list(rows[0]) if rows else ["rank", "model_id"]
    core.csv_write(output_dir / "selected_top10_models.csv", rows, columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tamamlanmış benchmarkın ilk 10 modelinden training lineage seçer, "
            "tüm epoch checkpointlerini sabit/dengeli 2000 sorguda çalıştırır."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-run", type=Path, required=True)
    parser.add_argument("--model-archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--staging-model-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--search-workers", type=int, default=None)
    parser.add_argument("--keep-maps", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or _is_within(first, second) or _is_within(second, first)


def _directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))


def run(args: argparse.Namespace) -> int:
    benchmark_run = args.benchmark_run.resolve()
    archive_root = args.model_archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    staging_root = args.staging_model_dir.resolve()
    if not benchmark_run.is_dir():
        raise FileNotFoundError(f"Benchmark run klasörü bulunamadı: {benchmark_run}")
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Model arşivi bulunamadı: {archive_root}")
    if _paths_overlap(output_dir, benchmark_run):
        raise ValueError("Epoch sweep output klasörü kaynak benchmark run ile çakışamaz.")
    if _paths_overlap(output_dir, archive_root):
        raise ValueError("Epoch sweep output klasörü kaynak model arşiviyle çakışamaz.")
    if _paths_overlap(staging_root, archive_root):
        raise ValueError("Staging model klasörü model arşiviyle çakışamaz.")
    if _paths_overlap(staging_root, benchmark_run):
        raise ValueError("Staging model klasörü kaynak benchmark run ile çakışamaz.")
    if _paths_overlap(output_dir, staging_root):
        raise ValueError("Epoch sweep output ve staging model klasörleri çakışamaz.")

    source_config, summary, identities = load_source_run(benchmark_run)
    ranked_models = rank_top_models(source_config, summary, identities)
    archive_groups, sha_to_lineages = scan_epoch_archive(archive_root)
    lineages = resolve_selected_lineages(ranked_models, archive_groups, sha_to_lineages)
    print_plan(ranked_models, lineages)
    plan = plan_payload(lineages)
    contract = scientific_contract(source_config, plan)
    if args.dry_run:
        print("Dry-run: kopyalama, manifest üretimi ve benchmark çalıştırma yapılmadı.")
        return 0

    batch_size = int(
        args.batch_size if args.batch_size is not None else source_config.get("batch_size", 16)
    )
    search_workers = int(
        args.search_workers
        if args.search_workers is not None
        else source_config.get("search_workers", 8)
    )
    if batch_size < 1:
        raise ValueError("batch-size pozitif olmalıdır.")
    if not 1 <= search_workers <= 8:
        raise ValueError("search-workers 1 ile 8 arasında olmalıdır.")

    sweep_config_exists = (output_dir / "epoch_sweep_config.json").is_file()
    results_exist = (output_dir / "results.jsonl").is_file()
    if not sweep_config_exists and _directory_has_files(output_dir):
        raise RuntimeError(
            "Epoch sweep output klasöründe config olmadan hazırlanmış dosyalar var; "
            "güvenli biçimde üzerine yazılmadı. Yeni bir --output-dir kullanın."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if sweep_config_exists:
        # Prepared runs are always validated read-only first, even when the
        # first result row has not yet been written.
        staging_payload = load_existing_staging(lineages, staging_root)
        query_index = validate_query_manifest_index(output_dir)
    else:
        if results_exist:
            raise RuntimeError(
                "results.jsonl var ancak epoch_sweep_config.json yok; "
                "güvenli resume yapılamaz."
            )
        staging_payload = prepare_staging(lineages, staging_root)
        query_index = prepare_query_manifest_index(output_dir, source_config)
    validate_staging(staging_payload)
    identity = sweep_resume_identity(
        source=source_fingerprint(benchmark_run),
        contract=contract,
        plan=plan,
        staging_payload=staging_payload,
        query_index=query_index,
    )
    write_or_validate_sweep_config(output_dir, identity, results_exist=results_exist)
    write_top_models(output_dir, ranked_models)
    write_selected_lineages(output_dir, staging_payload)

    core_argv = build_core_argv(
        output_dir=output_dir,
        staging_root=staging_root,
        source_config=source_config,
        batch_size=batch_size,
        search_workers=search_workers,
        keep_maps=bool(args.keep_maps),
        fail_fast=bool(args.fail_fast),
    )
    if args.verbose:
        core_argv.append("--verbose")
    update_sweep_status(output_dir, "running", core_argv=core_argv)
    return_code = 1
    core_error: BaseException | None = None
    sync_error: BaseException | None = None
    error_count = 0
    try:
        return_code = core.main(core_argv)
    except BaseException as exc:
        core_error = exc
    finally:
        try:
            error_count = sync_epoch_errors(output_dir, staging_payload)
        except BaseException as exc:
            sync_error = exc

    expected_epoch_rows = sum(
        len(lineage.get("checkpoints", []))
        for lineage in staging_payload.get("lineages", [])
    )
    epoch_rows = 0
    complete_epoch_rows = 0
    best_rows = 0
    report_error: BaseException | None = None
    if (output_dir / "summary.json").is_file() and (output_dir / "model_catalog.json").is_file():
        try:
            epoch_rows, complete_epoch_rows, best_rows = write_epoch_reports(
                output_dir, staging_payload, source_config
            )
        except BaseException as exc:
            report_error = exc
    outcome_error = next(
        (
            error
            for error in (core_error, sync_error, report_error)
            if error is not None
        ),
        None,
    )
    status, effective_return_code = classify_sweep_outcome(
        core_return_code=return_code,
        execution_error=outcome_error,
        expected_epoch_rows=expected_epoch_rows,
        complete_epoch_rows=complete_epoch_rows,
    )
    update_sweep_status(
        output_dir,
        status,
        core_return_code=return_code,
        effective_return_code=effective_return_code,
        expected_epoch_rows=expected_epoch_rows,
        epoch_result_rows=epoch_rows,
        complete_epoch_rows=complete_epoch_rows,
        best_lineage_rows=best_rows,
        error_records=error_count,
    )
    print(
        f"Epoch sweep durum={status}; tamamlanan_epoch={complete_epoch_rows}/"
        f"{expected_epoch_rows}; all_epoch_results={epoch_rows}; "
        f"best_lineage={best_rows}; hata_kaydı={error_count}"
    )
    if core_error is not None:
        raise core_error
    if sync_error is not None:
        raise sync_error
    if report_error is not None:
        raise report_error
    return effective_return_code


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Epoch sweep kullanıcı tarafından durduruldu; checkpointler korundu.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Epoch sweep başarısız: {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
