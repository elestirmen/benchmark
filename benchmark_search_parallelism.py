#!/usr/bin/env python
"""Benchmark OpenCV/thread-pool settings on an existing prepared model map.

This is a read-only calibration tool: it loads a completed map representation
and query tiles, runs the production search functions, and verifies that every
parallelism configuration returns the same search outcomes.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from geospatial_model_benchmark import (
    QueryRecord,
    coarse_to_fine_search,
    model_query_template,
    prepare_roi_search,
    prepare_search_map,
    search_with_prepared_roi,
)


FACTORS = (16, 8, 4, 2, 1)
MODES = (
    ("roi_500m", 500.0),
    ("roi_1000m", 1000.0),
    ("roi_2000m", 2000.0),
    ("roi_4000m", 4000.0),
    ("roi_8000m", 8000.0),
    ("global", None),
)


def parse_config(value: str) -> tuple[int, int]:
    workers, opencv_threads = value.split(":", 1)
    return int(workers), int(opencv_threads)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--map", type=Path, required=True)
    result.add_argument("--query-manifest", type=Path, required=True)
    result.add_argument("--query-root", type=Path, required=True)
    result.add_argument("--samples", type=int, default=12)
    result.add_argument(
        "--config",
        action="append",
        type=parse_config,
        default=[],
        metavar="WORKERS:OPENCV_THREADS",
    )
    return result


def outcome_key(outcome: object) -> tuple[object, ...]:
    return tuple(getattr(outcome, name) for name in ("x", "y", "top1_score", "top2_score", "peak_margin", "psr"))


def main() -> int:
    args = parser().parse_args()
    configs = args.config or [(4, 1), (8, 1), (12, 1), (16, 1), (8, 2), (4, 4)]
    payload = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    records = [QueryRecord(**row) for row in payload["queries"]]
    if args.samples < 1 or args.samples > len(records):
        raise ValueError("samples must be between 1 and the manifest query count")
    indexes = [round(index * (len(records) - 1) / max(1, args.samples - 1)) for index in range(args.samples)]
    selected = [records[index] for index in indexes]
    templates = {
        (variant, record.query_id): model_query_template(
            args.query_root / variant / f"{record.query_id}.png", 16
        )
        for variant in ("clean", "hard_v1")
        for record in selected
    }
    prepared_map = prepare_search_map(
        args.map,
        model_id="PARALLELISM_BENCHMARK",
        factors=FACTORS,
        search_modes=MODES,
    )
    tasks = [(record, mode, radius) for record in selected for mode, radius in MODES]

    baseline: list[tuple[object, ...]] | None = None
    reports: list[dict[str, object]] = []
    for workers, opencv_threads in configs:
        cv2.setNumThreads(opencv_threads)

        def evaluate(task: tuple[QueryRecord, str, float | None]) -> tuple[object, ...]:
            record, mode, radius = task
            representative = templates[("clean", record.query_id)]
            prepared_roi = None
            if mode != "global":
                assert radius is not None
                prepared_roi = prepare_roi_search(
                    map_gray=prepared_map.gray,
                    map_transform=prepared_map.transform,
                    center_easting_m=record.center_easting_m,
                    center_northing_m=record.center_northing_m,
                    roi_radius_m=radius,
                    template_shape=representative.shape[:2],
                    factors=FACTORS,
                )
            values: list[object] = [record.query_id, mode]
            for variant in ("clean", "hard_v1"):
                if mode == "global":
                    assert prepared_map.pyramid is not None
                    outcome = coarse_to_fine_search(
                        prepared_map.pyramid,
                        templates[(variant, record.query_id)],
                        factors=FACTORS,
                        top_k=30,
                        refine_radius_full_px=160,
                        nms_radius_full_px=256,
                        template_pyramid=None,
                    )
                else:
                    assert prepared_roi is not None
                    outcome = search_with_prepared_roi(
                        prepared_roi,
                        templates[(variant, record.query_id)],
                        factors=FACTORS,
                        top_k=30,
                        refine_radius_px=160,
                        nms_radius_px=256,
                        template_pyramid=None,
                    )
                values.extend(outcome_key(outcome))
            return tuple(values)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(evaluate, tasks))
        elapsed = time.perf_counter() - started
        if baseline is None:
            baseline = outcomes
        reports.append(
            {
                "workers": workers,
                "opencv_threads": opencv_threads,
                "tasks": len(tasks),
                "results": len(tasks) * 2,
                "seconds": round(elapsed, 3),
                "results_per_second": round(len(tasks) * 2 / elapsed, 3),
                "exact_match": outcomes == baseline,
            }
        )
        print("PARALLELISM_RESULT", json.dumps(reports[-1], ensure_ascii=False), flush=True)
    print("PARALLELISM_SUMMARY", json.dumps(reports, ensure_ascii=False), flush=True)
    return 0 if all(report["exact_match"] for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
