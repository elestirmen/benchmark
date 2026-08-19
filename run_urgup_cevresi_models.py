#!/usr/bin/env python
"""Run the existing benchmark on the wider Urgup Google/Bing raster pair."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import rasterio


ROOT = Path(__file__).resolve().parent
QUERY_RASTER = ROOT / "urgup_cevresi_harita" / "urgup_cevresi_gmap_utm_30cm.tif"
MAP_RASTER = ROOT / "urgup_cevresi_harita" / "urgup_cevresi_bingmap_utm_30cm.tif"
MODEL_DIR = ROOT / "models"
DEFAULT_RUN_DIR = ROOT / "outputs" / "urgup_cevresi_models_3000_utm"
MODEL_SUFFIXES = {".h5", ".keras", ".hdf5"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Geniş Ürgüp Google/Bing haritalarında models/ klasörünü test eder."
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    result.add_argument("--max-queries", type=int, default=3000)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--search-workers", type=int, default=8)
    result.add_argument("--keep-maps", action="store_true")
    result.add_argument("--fail-fast", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--verbose", action="store_true")
    result.add_argument(
        "--isolate-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Her model/yönü ayrı Python sürecinde çalıştırarak birikimli belleği boşaltır.",
    )
    return result


def model_files() -> list[Path]:
    return sorted(
        path
        for path in MODEL_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )


def raster_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Raster bulunamadı: {path}")
    with rasterio.open(path) as dataset:
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError(f"Raster metre tabanlı projeksiyonda değil: {path}")
        return {
            "path": str(path),
            "crs": dataset.crs.to_string(),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "resolution": [abs(dataset.res[0]), abs(dataset.res[1])],
            "bounds": list(dataset.bounds),
        }


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    if args.max_queries < 1 or args.batch_size < 1:
        raise ValueError("max-queries ve batch-size pozitif olmalıdır.")
    if not 1 <= args.search_workers <= 8:
        raise ValueError("search-workers 1 ile 8 arasında olmalıdır.")
    models = model_files()
    if not models:
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_DIR}")
    query = raster_metadata(QUERY_RASTER)
    search_map = raster_metadata(MAP_RASTER)
    if query["crs"] != search_map["crs"]:
        raise ValueError(f"Raster CRS değerleri farklı: {query['crs']} != {search_map['crs']}")
    qb, mb = query["bounds"], search_map["bounds"]
    overlap_width = min(qb[2], mb[2]) - max(qb[0], mb[0])
    overlap_height = min(qb[3], mb[3]) - max(qb[1], mb[1])
    if overlap_width <= 0 or overlap_height <= 0:
        raise ValueError("Google ve Bing rasterlarının ortak alanı yok.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not (
        output_dir / "run_config.json"
    ).is_file():
        raise RuntimeError(
            f"Çıktı klasörü dolu ancak güvenli resume bilgisi yok: {output_dir}"
        )
    return {
        "query_raster": query,
        "map_raster": search_map,
        "overlap_m": [overlap_width, overlap_height],
        "model_count": len(models),
        "output_dir": str(output_dir),
        "resume": (output_dir / "run_config.json").is_file(),
    }


def core_arguments(args: argparse.Namespace, *, resume: bool) -> list[str]:
    output_dir = args.output_dir.resolve()
    argv = [
        "--query-raster", str(QUERY_RASTER),
        "--map-raster", str(MAP_RASTER),
        "--model-dir", str(MODEL_DIR),
        "--model-sampling", "full",
        "--include-raw", "--include-models", "--bidirectional",
        "--query-variants", "clean,hard_v1",
        "--search-modes", "roi500,roi1000,roi2000,roi4000,roi8000,global",
        "--tile-size", "544", "--overlap", "32", "--crop-border", "16",
        "--block-size-m", "1000", "--samples-per-block", "5",
        "--max-queries", str(args.max_queries),
        "--query-sampling", "balanced_exact", "--seed", "42",
        "--min-query-std", "12", "--min-query-entropy", "4",
        "--max-dark-fraction", "0.20",
        "--pyramid-factors", "16,8,4,2,1", "--top-k", "30",
        "--refine-radius-px", "160", "--nms-radius-px", "256",
        "--batch-size", str(args.batch_size),
        "--search-workers", str(args.search_workers),
        "--bootstrap-iterations", "1000",
        "--normalization", "minus1_1", "--output-value-mode", "auto",
        "--enhancement", "none",
        "--excel-engine", "openpyxl", "--excel-update", "model",
        "--excel-report", "--no-results-csv",
    ]
    if resume:
        argv.extend(["--resume-run", str(output_dir)])
    else:
        argv.extend(["--output-root", str(output_dir.parent), "--run-id", output_dir.name])
    if not args.keep_maps:
        argv.append("--cleanup-maps")
    if args.fail_fast:
        argv.append("--fail-fast")
    if args.verbose:
        argv.append("--verbose")
    return argv


def completed_worker_keys(
    output_dir: Path,
    *,
    max_queries: int,
    expected_groups: int = 12,
) -> set[tuple[str, str]]:
    """Return direction/model checkpoints completed at the summary boundary."""
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return set()
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in rows:
        total = int(row.get("total_queries") or 0)
        accounted = sum(
            int(row.get(name) or 0)
            for name in ("ok_queries", "rejected_queries", "error_queries")
        )
        if total != max_queries or accounted != total:
            continue
        key = (str(row.get("direction") or ""), str(row.get("model_id") or ""))
        groups.setdefault(key, set()).add(
            (str(row.get("query_variant") or ""), str(row.get("search_mode") or ""))
        )
    return {key for key, values in groups.items() if len(values) >= expected_groups}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    plan = validate_inputs(args)
    benchmark_argv = core_arguments(args, resume=bool(plan["resume"]))
    print("WIDE_URGUP_PLAN_JSON:", json.dumps(plan, ensure_ascii=False))
    print("Komut:", "python geospatial_model_benchmark.py", " ".join(benchmark_argv))
    if args.dry_run:
        print("Dry-run: benchmark başlatılmadı ve çıktı oluşturulmadı.")
        return 0
    import geospatial_model_benchmark as benchmark

    if not args.isolate_models:
        return benchmark.main(benchmark_argv)

    catalog = benchmark.build_model_catalog(
        MODEL_DIR,
        benchmark.parse_patterns(None),
        None,
        "full",
    )
    directions = [
        (
            "forward",
            benchmark.direction_name(QUERY_RASTER, MAP_RASTER),
        ),
        (
            "reverse",
            benchmark.direction_name(MAP_RASTER, QUERY_RASTER),
        ),
    ]
    total_workers = len(catalog.models) * len(directions)
    worker_number = 0
    for direction, direction_id in directions:
        for model in catalog.models:
            worker_number += 1
            completed_keys = completed_worker_keys(
                args.output_dir.resolve(), max_queries=args.max_queries
            )
            if (direction_id, model.model_id) in completed_keys:
                print(
                    f"İZOLE MODEL {worker_number}/{total_workers} | checkpoint tamam, atlandı | "
                    f"yön={direction} | model={model.path.name}",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(ROOT / "geospatial_model_benchmark.py"),
                *benchmark_argv,
                "--worker-model-id",
                model.model_id,
                "--worker-direction",
                direction,
                "--worker-skip-final-export",
            ]
            print(
                f"İZOLE MODEL {worker_number}/{total_workers} | yön={direction} | "
                f"model={model.path.name}",
                flush=True,
            )
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                print(
                    f"İzole model süreci başarısız: exit={completed.returncode} | "
                    f"model={model.path.name}",
                    file=sys.stderr,
                )
                return completed.returncode

    final_command = [
        sys.executable,
        str(ROOT / "geospatial_model_benchmark.py"),
        *benchmark_argv,
        "--worker-finalize-only",
    ]
    print("İZOLE MODELLER TAMAMLANDI | son özet ve Excel hazırlanıyor", flush=True)
    return subprocess.run(final_command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
