from __future__ import annotations

import json
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import rasterio
from affine import Affine


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
import sys

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from geospatial_model_benchmark import (  # noqa: E402
    aggregate_results,
    augment_hard_v1,
    build_parser,
    build_pyramid,
    coarse_to_fine_search,
    generate_query_manifest,
    prepare_query_variants,
    refresh_excel_after_model,
    search_in_mode,
)


class ExcelCheckpointTests(unittest.TestCase):
    def test_model_is_the_default_excel_update_boundary(self) -> None:
        self.assertEqual(build_parser().parse_args([]).excel_update, "model")

    def test_model_checkpoint_is_non_strict_and_refreshes_summary_first(self) -> None:
        args = argparse.Namespace(excel_update="model", excel_engine="openpyxl")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            workbook = run_dir / "benchmark_results.xlsx"
            with (
                patch("geospatial_model_benchmark.write_summary_files") as write_summary,
                patch("geospatial_model_benchmark.invoke_excel_report") as invoke_excel,
            ):
                write_summary.return_value = (run_dir / "summary.json", run_dir / "summary.csv")
                invoke_excel.return_value = workbook
                result = refresh_excel_after_model(
                    run_dir,
                    args,
                    position=2,
                    total=7,
                    model_name="model.h5",
                    model_status="tamamlandı",
                )
        self.assertEqual(result, workbook)
        write_summary.assert_called_once_with(run_dir)
        invoke_excel.assert_called_once_with(run_dir, strict=False, engine="openpyxl")


class DefaultBenchmarkModeTests(unittest.TestCase):
    def test_clean_and_hard_v1_are_enabled_by_default(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).query_variants, ("clean", "hard_v1"))
        self.assertEqual(
            parser.parse_args(["--query-variants", "clean"]).query_variants,
            ("clean",),
        )

    def test_default_sampling_targets_300_queries_per_direction(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.samples_per_block, 5)
        self.assertEqual(args.max_queries, 300)

    def test_bidirectional_is_enabled_by_default_and_can_be_disabled(self) -> None:
        parser = build_parser()
        self.assertTrue(parser.parse_args([]).bidirectional)
        self.assertFalse(parser.parse_args(["--no-bidirectional"]).bidirectional)

    def test_default_modes_are_independent_expanding_rois_then_global(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(
            args.search_modes,
            (
                ("roi_500m", 500.0),
                ("roi_1000m", 1000.0),
                ("roi_2000m", 2000.0),
                ("roi_4000m", 4000.0),
                ("roi_8000m", 8000.0),
                ("global", None),
            ),
        )


class CoarseToFineSearchTests(unittest.TestCase):
    def test_exact_unique_patch_is_recovered(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(768, 896), dtype=np.uint8)
        x, y, size = 431, 287, 128
        template = image[y : y + size, x : x + size].copy()
        factors = (8, 4, 2, 1)
        outcome = coarse_to_fine_search(
            build_pyramid(image, factors),
            template,
            factors=factors,
            top_k=12,
            refine_radius_full_px=48,
            nms_radius_full_px=64,
        )
        self.assertEqual((outcome.x, outcome.y), (x, y))
        self.assertGreater(outcome.top1_score, 0.999)
        self.assertGreater(outcome.peak_margin, 0.1)

    def test_roi_search_returns_global_coordinates(self) -> None:
        rng = np.random.default_rng(19)
        image = rng.integers(0, 256, size=(640, 704), dtype=np.uint8)
        x, y, size = 318, 241, 96
        template = image[y : y + size, x : x + size].copy()
        transform = Affine(1, 0, 500000, 0, -1, 4200000)
        center_e, center_n = transform * (x + size / 2, y + size / 2)
        outcome = search_in_mode(
            map_gray=image,
            global_pyramid=None,
            map_transform=transform,
            template=template,
            center_easting_m=center_e,
            center_northing_m=center_n,
            mode_name="roi_100m",
            roi_radius_m=100.0,
            factors=(4, 2, 1),
            top_k=8,
            refine_radius_px=32,
            nms_radius_px=48,
        )
        self.assertEqual((outcome.x, outcome.y), (x, y))


class AggregateTests(unittest.TestCase):
    def test_clean_and_hard_results_are_aggregated_separately(self) -> None:
        rows = [
            {
                "direction": "A",
                "query_variant": variant,
                "search_mode": "global",
                "model_id": "M",
                "status": "ok",
                "error_m": error,
                "search_seconds": 1.0,
                "top1_score": 0.8,
            }
            for variant, error in (("clean", 2.0), ("hard_v1", 50.0))
        ]
        summary = aggregate_results(rows, bootstrap_iterations=0)
        self.assertEqual([row["query_variant"] for row in summary], ["clean", "hard_v1"])
        self.assertEqual(summary[0]["success_30m"], 1.0)
        self.assertEqual(summary[1]["success_30m"], 0.0)

    def test_summary_order_is_roi_small_to_global_within_each_model(self) -> None:
        rows = []
        for mode in ("global", "roi_8000m", "roi_500m", "roi_2000m", "roi_1000m"):
            rows.append(
                {
                    "direction": "A",
                    "search_mode": mode,
                    "model_id": "M",
                    "status": "ok",
                    "error_m": 5.0,
                    "search_seconds": 1.0,
                    "top1_score": 0.8,
                }
            )
        summary = aggregate_results(rows, bootstrap_iterations=0)
        self.assertEqual(
            [row["search_mode"] for row in summary],
            ["roi_500m", "roi_1000m", "roi_2000m", "roi_8000m", "global"],
        )

    def test_summary_uses_only_successful_errors(self) -> None:
        rows = [
            {"direction": "A", "model_id": "M", "status": "ok", "error_m": 4.0, "search_seconds": 1.0, "top1_score": 0.8},
            {"direction": "A", "model_id": "M", "status": "ok", "error_m": 16.0, "search_seconds": 3.0, "top1_score": 0.6},
            {"direction": "A", "model_id": "M", "status": "error", "error_m": None, "search_seconds": 2.0, "top1_score": None},
        ]
        summary = aggregate_results(rows)[0]
        self.assertEqual(summary["total_queries"], 3)
        self.assertEqual(summary["ok_queries"], 2)
        self.assertAlmostEqual(summary["coverage"], 2 / 3)
        self.assertAlmostEqual(summary["median_error_m"], 10.0)
        self.assertAlmostEqual(summary["success_5m"], 0.5)
        self.assertAlmostEqual(summary["success_25m"], 1.0)
        self.assertEqual(summary["success_30m_queries"], 2)
        self.assertAlmostEqual(summary["success_30m"], 2 / 3)
        self.assertAlmostEqual(summary["success_30m_failure_rate"], 1 / 3)
        self.assertAlmostEqual(summary["mean_error_under_30m"], 10.0)
        self.assertAlmostEqual(summary["median_error_under_30m"], 10.0)
        self.assertAlmostEqual(summary["auc_30m"], 4 / 9)

    def test_large_miss_does_not_dominate_primary_30m_score(self) -> None:
        rows = [
            {
                "direction": "A",
                "model_id": "M",
                "status": "ok",
                "error_m": 10.0,
                "search_seconds": 1.0,
                "top1_score": 0.8,
            },
            {
                "direction": "A",
                "model_id": "M",
                "status": "ok",
                "error_m": 5000.0,
                "search_seconds": 1.0,
                "top1_score": 0.2,
            },
        ]
        summary = aggregate_results(rows, bootstrap_iterations=25)[0]
        self.assertAlmostEqual(summary["success_30m"], 0.5)
        self.assertAlmostEqual(summary["mean_error_under_30m"], 10.0)
        self.assertAlmostEqual(summary["median_error_under_30m"], 10.0)
        self.assertGreater(summary["mean_error_m"], 2000.0)

    def test_no_valid_match_has_zero_primary_success_with_defined_ci(self) -> None:
        rows = [
            {
                "direction": "A",
                "model_id": "M",
                "status": "error",
                "error_m": None,
                "search_seconds": 1.0,
                "top1_score": None,
                "block_id": "B1",
            }
        ]
        summary = aggregate_results(rows, bootstrap_iterations=25)[0]
        self.assertEqual(summary["success_30m"], 0.0)
        self.assertEqual(summary["success_30m_ci95_low"], 0.0)
        self.assertEqual(summary["success_30m_ci95_high"], 0.0)


class ManifestTests(unittest.TestCase):
    def test_hard_v1_augmentation_is_deterministic_and_changes_pixels(self) -> None:
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        first, first_params = augment_hard_v1(image, seed=99)
        second, second_params = augment_hard_v1(image, seed=99)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_params, second_params)
        self.assertNotIn("rotation_deg", first_params)
        self.assertNotIn("scale", first_params)
        self.assertNotIn("perspective_x_per_px", first_params)
        self.assertEqual(first.shape, image.shape)
        self.assertGreater(float(np.mean(np.abs(first.astype(float) - image.astype(float)))), 10.0)
        self.assertGreaterEqual(first_params["blur_sigma"], 0.80)
        self.assertGreaterEqual(first_params["noise_std"], 3.0)
        self.assertLessEqual(first_params["jpeg_quality"], 75)

    def _write_raster(self, path: Path, seed: int) -> None:
        rng = np.random.default_rng(seed)
        data = rng.integers(20, 235, size=(3, 640, 768), dtype=np.uint8)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=640,
            width=768,
            count=3,
            dtype="uint8",
            crs="EPSG:32636",
            transform=Affine(0.3, 0, 600000, 0, -0.3, 4200000),
        ) as dataset:
            dataset.write(data)

    def test_seeded_manifest_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.tif"
            map_path = root / "map.tif"
            self._write_raster(query, 1)
            self._write_raster(map_path, 2)
            kwargs = dict(
                tile_size=64,
                samples_per_block=2,
                block_size_m=60.0,
                max_queries=6,
                seed=42,
                min_std=1.0,
                min_entropy=1.0,
                max_dark_fraction=1.0,
                edge_buffer_m=None,
                force=True,
            )
            first = generate_query_manifest(query, map_path, root / "first", **kwargs)
            second = generate_query_manifest(query, map_path, root / "second", **kwargs)
            first_centres = [(row.center_easting_m, row.center_northing_m) for row in first]
            second_centres = [(row.center_easting_m, row.center_northing_m) for row in second]
            self.assertEqual(first_centres, second_centres)
            payload = json.loads((root / "first" / "query_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 42)
            self.assertAlmostEqual(payload["effective_edge_buffer_m"], 64 * 0.3)
            self.assertEqual(len(payload["queries"]), 6)

    def test_explicit_edge_buffer_keeps_query_centres_away_from_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.tif"
            map_path = root / "map.tif"
            self._write_raster(query, 3)
            self._write_raster(map_path, 4)
            records = generate_query_manifest(
                query,
                map_path,
                root / "buffered",
                tile_size=64,
                samples_per_block=2,
                block_size_m=60.0,
                max_queries=8,
                seed=42,
                min_std=1.0,
                min_entropy=1.0,
                max_dark_fraction=1.0,
                edge_buffer_m=30.0,
                force=True,
            )
            with rasterio.open(query) as dataset:
                centre_margin = (64 / 2.0 + 2.0) * 0.3 + 30.0
                for record in records:
                    self.assertGreaterEqual(
                        record.center_easting_m, dataset.bounds.left + centre_margin
                    )
                    self.assertLessEqual(
                        record.center_easting_m, dataset.bounds.right - centre_margin
                    )
                    self.assertGreaterEqual(
                        record.center_northing_m, dataset.bounds.bottom + centre_margin
                    )
                    self.assertLessEqual(
                        record.center_northing_m, dataset.bounds.top - centre_margin
                    )


if __name__ == "__main__":
    unittest.main()
