from __future__ import annotations

import json
import argparse
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import rasterio
from affine import Affine


BENCHMARK_DIR = Path(__file__).resolve().parents[1]

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from geospatial_model_benchmark import (  # noqa: E402
    HARD_V1_PROFILE,
    JsonlBatchWriter,
    RESULT_COLUMNS,
    PreparedSearchMap,
    QueryRecord,
    aggregate_results,
    augment_hard_v1,
    build_model_catalog,
    build_model_map,
    build_parser,
    main as benchmark_main,
    build_pyramid,
    coarse_to_fine_search,
    compute_starts,
    generate_query_manifest,
    invoke_excel_report,
    model_prediction_to_legacy_gray,
    nms_top_candidates,
    parse_patterns,
    read_jsonl,
    refresh_excel_after_model,
    resolve_run_directory,
    resume_signature_payload,
    resume_payloads_compatible,
    run_direction,
    run_searches_for_representation,
    run_searches_for_variants,
    search_in_mode,
    select_models,
    streaming_map_metadata,
    validate_rasters,
    write_summary_files,
)


class ExcelCheckpointTests(unittest.TestCase):
    def test_existing_explicit_run_id_auto_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            run_dir = output_root / "same_parameters"
            run_dir.mkdir()
            (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
            args = build_parser().parse_args(
                ["--output-root", str(output_root), "--run-id", "same_parameters"]
            )
            args.output_root = args.output_root.resolve()
            resolved, automatic = resolve_run_directory(args)
            self.assertTrue(automatic)
            self.assertEqual(resolved, run_dir)
            self.assertEqual(args.resume_run, run_dir)

    def test_new_explicit_run_id_does_not_auto_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            args = build_parser().parse_args(
                ["--output-root", str(output_root), "--run-id", "new_run"]
            )
            args.output_root = args.output_root.resolve()
            resolved, automatic = resolve_run_directory(args)
            self.assertFalse(automatic)
            self.assertEqual(resolved, output_root / "new_run")
            self.assertIsNone(args.resume_run)

    def test_model_is_the_default_excel_update_boundary(self) -> None:
        self.assertEqual(build_parser().parse_args([]).excel_update, "model")

    def test_eight_search_workers_are_enabled_by_default(self) -> None:
        self.assertEqual(build_parser().parse_args([]).search_workers, 8)

    def test_worker_controls_are_operational_not_scientific(self) -> None:
        base = build_parser().parse_args([])
        worker = build_parser().parse_args(
            [
                "--worker-model-id",
                "MODEL_1",
                "--worker-direction",
                "forward",
                "--worker-skip-final-export",
            ]
        )
        with patch.object(Path, "stat") as stat:
            stat.return_value = SimpleNamespace(st_size=1, st_mtime_ns=2)
            self.assertEqual(resume_signature_payload(base), resume_signature_payload(worker))

    def test_batch_size_is_operational_for_resume_compatibility(self) -> None:
        previous = {"schema_version": 2, "batch_size": 64, "seed": 42}
        current = {"schema_version": 2, "seed": 42}
        self.assertTrue(resume_payloads_compatible(previous, current))
        self.assertFalse(
            resume_payloads_compatible(previous, {"schema_version": 2, "seed": 43})
        )

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
        write_summary.assert_called_once_with(run_dir, write_results_csv=False)
        invoke_excel.assert_called_once_with(
            run_dir,
            strict=False,
            engine="openpyxl",
            lightweight=True,
            validation_mode="checkpoint",
        )

    def test_lightweight_reporter_flag_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            workbook = run_dir / "benchmark_results.xlsx"
            workbook.write_bytes(b"xlsx")
            marker = json.dumps({"path": str(workbook), "report": {}})
            completed = SimpleNamespace(
                returncode=0,
                stdout=f"WORKBOOK_READY_JSON: {marker}\n",
                stderr="",
            )
            with patch(
                "geospatial_model_benchmark.subprocess.run", return_value=completed
            ) as run:
                result = invoke_excel_report(
                    run_dir,
                    strict=False,
                    engine="openpyxl",
                    lightweight=True,
                    validation_mode="checkpoint",
                )
            self.assertEqual(result, workbook.resolve())
            command = run.call_args.args[0]
            self.assertIn("--lightweight", command)
            self.assertNotIn("--incremental", command)

    def test_incremental_report_returns_actual_locked_copy_path_from_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            alternate = run_dir / "benchmark_results_20260809_120000_kilitli_kopya.xlsx"
            alternate.write_bytes(b"xlsx")
            marker = json.dumps({"path": str(alternate), "report": {}})
            completed = SimpleNamespace(
                returncode=0,
                stdout=f"WORKBOOK_READY_JSON: {marker}\n",
                stderr="",
            )
            with patch(
                "geospatial_model_benchmark.subprocess.run", return_value=completed
            ) as run:
                result = invoke_excel_report(
                    run_dir,
                    strict=False,
                    engine="openpyxl",
                    incremental=True,
                    validation_mode="checkpoint",
                )
            self.assertEqual(result, alternate.resolve())
            command = run.call_args.args[0]
            self.assertIn("--incremental", command)
            self.assertEqual(command[command.index("--validation-mode") + 1], "checkpoint")


class CheckpointWriterTests(unittest.TestCase):
    def test_jsonl_rows_are_flushed_in_batches_and_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            with JsonlBatchWriter(path, max_rows=2, max_seconds=60.0) as writer:
                writer.append({"row": 1})
                self.assertEqual(path.stat().st_size, 0)
                writer.append({"row": 2})
                self.assertGreater(path.stat().st_size, 0)
                writer.append({"row": 3})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"row": 1}, {"row": 2}, {"row": 3}])

    def test_model_summary_refresh_does_not_rewrite_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            row = {
                "direction": "A",
                "query_variant": "clean",
                "search_mode": "global",
                "model_id": "M",
                "status": "ok",
                "error_m": 4.0,
                "search_seconds": 1.0,
                "top1_score": 0.8,
            }
            (run_dir / "results.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            (run_dir / "run_config.json").write_text(
                json.dumps({"bootstrap_iterations": 0, "seed": 42}), encoding="utf-8"
            )
            results_csv = run_dir / "results.csv"
            results_csv.write_text("sentinel", encoding="utf-8")
            write_summary_files(run_dir, write_results_csv=False)
            self.assertEqual(results_csv.read_text(encoding="utf-8"), "sentinel")

    def test_partial_final_jsonl_line_is_quarantined_and_valid_prefix_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_bytes(b'{"row": 1}\n{"row": 2')
            self.assertEqual(read_jsonl(path), [{"row": 1}])
            self.assertEqual(path.read_bytes(), b'{"row": 1}\n')
            recovery = list(path.parent.glob("results.jsonl.recovery_*.bin"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(recovery[0].read_bytes(), b'{"row": 2')

    def test_jsonl_corruption_in_the_middle_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            original = b'{"row": 1}\nnot-json\n{"row": 3}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "middle"):
                read_jsonl(path)
            self.assertEqual(path.read_bytes(), original)

    def test_incremental_summary_reads_only_appended_rows_then_matches_full_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "bootstrap_iterations": 10,
                        "seed": 42,
                        "scientific_semantics_version": 2,
                    }
                ),
                encoding="utf-8",
            )
            results = run_dir / "results.jsonl"
            first = {
                "direction": "A",
                "query_variant": "clean",
                "search_mode": "global",
                "model_id": "M1",
                "query_id": "Q1",
                "block_id": "B1",
                "status": "ok",
                "error_m": 2.0,
                "search_seconds": 1.0,
                "top1_score": 0.9,
            }
            results.write_text(json.dumps(first) + "\n", encoding="utf-8")
            write_summary_files(run_dir, write_results_csv=False)
            previous_offset = json.loads(
                (run_dir / ".summary_state" / "summary_state.json").read_text(encoding="utf-8")
            )["jsonl_boundary"]["offset"]
            second = {**first, "query_id": "Q2", "error_m": 30.0}
            with results.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(second) + "\n")
            with patch(
                "geospatial_model_benchmark.read_jsonl",
                side_effect=AssertionError("checkpoint must not scan the full JSONL"),
            ):
                write_summary_files(run_dir, write_results_csv=False)
            state = json.loads(
                (run_dir / ".summary_state" / "summary_state.json").read_text(encoding="utf-8")
            )
            self.assertGreater(state["jsonl_boundary"]["offset"], previous_offset)
            write_summary_files(run_dir, write_results_csv=True)
            metadata = json.loads((run_dir / "summary_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["incremental_equivalence"], "exact_match")
            self.assertEqual(len((run_dir / "results.csv").read_text(encoding="utf-8-sig").splitlines()), 3)


class CorrectnessGuardTests(unittest.TestCase):
    def _write_raster(
        self,
        path: Path,
        *,
        transform: Affine = Affine(0.3, 0, 600000, 0, -0.3, 4200000),
        dtype: str = "uint8",
        seed: int = 1,
        width: int = 192,
        height: int = 160,
    ) -> None:
        rng = np.random.default_rng(seed)
        if dtype == "uint8":
            data = rng.integers(0, 256, size=(3, height, width), dtype=np.uint8)
        else:
            data = rng.integers(0, 4096, size=(3, height, width), dtype=np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=dtype,
            crs="EPSG:32636",
            transform=transform,
        ) as dataset:
            dataset.write(data)

    def test_hdf5_is_part_of_default_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            for name in ("a.h5", "b.keras", "c.hdf5", "ignore.txt"):
                (model_dir / name).write_bytes(b"x")
            self.assertEqual(parse_patterns(None), ("*.h5", "*.keras", "*.hdf5"))
            self.assertEqual(
                [path.name for path in select_models(model_dir, parse_patterns(None), None)],
                ["a.h5", "b.keras", "c.hdf5"],
            )

    def test_recursive_catalog_keeps_same_name_different_models_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "run_a").mkdir()
            (model_dir / "run_b").mkdir()
            (model_dir / "run_a" / "model.h5").write_bytes(b"weights-a")
            (model_dir / "run_b" / "model.h5").write_bytes(b"weights-b")

            catalog = build_model_catalog(model_dir, parse_patterns(None), None)

            self.assertEqual(catalog.discovered_files, 2)
            self.assertEqual(len(catalog.models), 2)
            self.assertEqual(catalog.duplicate_name_groups, 1)
            self.assertEqual(catalog.conflicting_name_groups, 1)
            self.assertEqual(len({model.model_id for model in catalog.models}), 2)
            self.assertTrue(all(model.sha256[:12] in model.model_id for model in catalog.models))
            self.assertEqual(
                {model.relative_path for model in catalog.models},
                {"run_a/model.h5", "run_b/model.h5"},
            )

    def test_recursive_catalog_skips_byte_identical_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "run_a").mkdir()
            (model_dir / "run_b").mkdir()
            (model_dir / "run_a" / "first.h5").write_bytes(b"same-weights")
            (model_dir / "run_b" / "copy.hdf5").write_bytes(b"same-weights")

            catalog = build_model_catalog(model_dir, parse_patterns(None), None)

            self.assertEqual(catalog.discovered_files, 2)
            self.assertEqual(len(catalog.models), 1)
            self.assertEqual(catalog.identical_files_skipped, 1)
            self.assertEqual(len(catalog.models[0].duplicate_paths), 1)

    def test_five_point_sampling_uses_checkpoint_quartiles_per_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            for epoch in range(1, 10):
                path = model_dir / f"GPU_model_f32_k3_epoch_{epoch:05d}_sigmoid.h5"
                path.write_bytes(f"weights-{epoch}".encode())

            catalog = build_model_catalog(
                model_dir,
                parse_patterns(None),
                None,
                "five-point",
            )

            self.assertEqual(catalog.sampling_mode, "five-point")
            self.assertEqual(catalog.series_count, 1)
            self.assertEqual(catalog.sampled_files, 5)
            self.assertEqual(
                [model.checkpoint_number for model in catalog.models],
                [1, 3, 5, 7, 9],
            )
            self.assertEqual(
                [model.selection_points for model in catalog.models],
                [("first",), ("q25",), ("middle",), ("q75",), ("last",)],
            )

    def test_five_point_sampling_separates_architectures_in_same_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            for architecture in ("f32_k3", "f48_k5"):
                for epoch in range(1, 10):
                    path = model_dir / f"GPU_model_{architecture}_epoch_{epoch:05d}.h5"
                    path.write_bytes(f"{architecture}-{epoch}".encode())

            catalog = build_model_catalog(
                model_dir,
                parse_patterns(None),
                None,
                "five-point",
            )

            self.assertEqual(catalog.series_count, 2)
            self.assertEqual(catalog.sampled_files, 10)
            self.assertEqual(len(catalog.models), 10)
            self.assertEqual(len({model.series_id for model in catalog.models}), 2)

    def test_five_point_sampling_requires_epoch_or_step_in_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "model.h5").write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "epoch/step"):
                build_model_catalog(
                    model_dir,
                    parse_patterns(None),
                    None,
                    "five-point",
                )

    def test_output_conversion_modes_are_non_overlapping_and_explicit(self) -> None:
        from goruntu_islemleri import prediction_to_uint8

        sigmoid = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
        tanh = np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32)
        raw = np.asarray([[0.0, 128.0, 255.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(prediction_to_uint8(sigmoid, "sigmoid"), [[0, 127, 255]]))
        self.assertTrue(np.array_equal(prediction_to_uint8(tanh, "tanh"), [[0, 127, 255]]))
        self.assertTrue(np.array_equal(prediction_to_uint8(sigmoid, "auto"), [[0, 127, 255]]))
        self.assertTrue(np.array_equal(prediction_to_uint8(tanh, "auto"), [[0, 127, 255]]))
        self.assertTrue(np.array_equal(prediction_to_uint8(raw, "raw"), [[0, 128, 255]]))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            prediction_to_uint8(np.asarray([[-4.0, 500.0]], dtype=np.float32), "auto")

    def test_runtime_direct_and_png_query_paths_are_pixel_equal(self) -> None:
        from goruntu_islemleri import LoadedModelRuntime

        class FakeRawModel:
            input_shape = (None, 8, 8, 3)
            output_shape = (None, 8, 8, 3)

            def predict(self, batch, verbose=0):
                return batch

        rng = np.random.default_rng(321)
        rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        runtime = LoadedModelRuntime(
            FakeRawModel(),
            model_path="fake.keras",
            normalization="raw",
            enhancement="none",
            output_value_mode="raw",
        )
        direct = runtime.predict_images([rgb], image_size=(8, 8), source_color="rgb")[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            cv2.imwrite(str(input_dir / "Q00001.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            [output_path] = runtime.process_directory(
                str(input_dir), str(output_dir), image_size=(8, 8), batch_size=1
            )
            cached = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(direct, cached))

    def test_raster_gsd_rotation_and_dtype_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.tif"
            valid = root / "valid.tif"
            gsd = root / "gsd.tif"
            rotated = root / "rotated.tif"
            uint16 = root / "uint16.tif"
            self._write_raster(query)
            self._write_raster(valid, seed=2)
            self._write_raster(gsd, transform=Affine(0.4, 0, 600000, 0, -0.4, 4200000))
            self._write_raster(rotated, transform=Affine(0.3, 0.01, 600000, 0, -0.3, 4200000))
            self._write_raster(uint16, dtype="uint16")
            validate_rasters(query, valid)
            with self.assertRaisesRegex(ValueError, "GSD mismatch"):
                validate_rasters(query, gsd)
            with self.assertRaisesRegex(ValueError, "rotation/skew"):
                validate_rasters(query, rotated)
            with self.assertRaisesRegex(ValueError, "dtype"):
                validate_rasters(query, uint16)

    def test_manifest_truth_and_perfect_match_use_the_same_even_tile_centre(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster_path = root / "same.tif"
            self._write_raster(raster_path, width=384, height=384, seed=99)
            records = generate_query_manifest(
                raster_path,
                raster_path,
                root / "queries",
                tile_size=64,
                samples_per_block=1,
                block_size_m=50.0,
                max_queries=1,
                seed=42,
                min_std=1.0,
                min_entropy=1.0,
                max_dark_fraction=1.0,
                edge_buffer_m=0.0,
                force=True,
            )
            record = records[0]
            with rasterio.open(raster_path) as dataset:
                expected_e, expected_n = dataset.transform * (record.source_col, record.source_row)
                map_gray = cv2.cvtColor(
                    np.moveaxis(dataset.read([1, 2, 3]), 0, -1),
                    cv2.COLOR_RGB2GRAY,
                )
                transform = dataset.transform
            self.assertAlmostEqual(record.center_easting_m, expected_e, places=9)
            self.assertAlmostEqual(record.center_northing_m, expected_n, places=9)
            prepared = PreparedSearchMap(map_gray, transform, build_pyramid(map_gray, (4, 2, 1)))
            results = root / "results.jsonl"
            run_searches_for_representation(
                run_id="perfect",
                direction="same",
                query_variant="clean",
                model_id="RAW_BASELINE",
                model_file="",
                model_sha256="",
                map_path=raster_path,
                prepared_map=prepared,
                map_build_seconds=0.0,
                queries=records,
                query_paths=None,
                query_inference_total_seconds=0.0,
                crop_border=0,
                search_modes=(("global", None),),
                factors=(4, 2, 1),
                top_k=8,
                refine_radius_px=32,
                nms_radius_px=32,
                normalization="RAW",
                source_query_raster=raster_path,
                source_map_raster=raster_path,
                results_jsonl=results,
                done=set(),
                search_workers=1,
            )
            row = json.loads(results.read_text(encoding="utf-8"))
            self.assertEqual(row["error_px"], 0.0)
            self.assertAlmostEqual(row["error_m"], 0.0, places=9)
            self.assertAlmostEqual(
                row["predicted_center_easting_m"], row["expected_center_easting_m"], places=9
            )
            self.assertAlmostEqual(
                row["predicted_center_northing_m"], row["expected_center_northing_m"], places=9
            )

    def test_old_scientific_semantics_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.tif"
            map_path = root / "map.tif"
            self._write_raster(query)
            self._write_raster(map_path, seed=2)
            run_dir = root / "old_run"
            run_dir.mkdir()
            (run_dir / "run_config.json").write_text(json.dumps({}), encoding="utf-8")
            (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
            try:
                with self.assertRaisesRegex(RuntimeError, "older scientific semantics"):
                    benchmark_main(
                        [
                            "--query-raster",
                            str(query),
                            "--map-raster",
                            str(map_path),
                            "--resume-run",
                            str(run_dir),
                            "--no-include-models",
                            "--max-queries",
                            "1",
                        ]
                    )
            finally:
                logger = logging.getLogger("geospatial_benchmark")
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)


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
    @staticmethod
    def reference_nms(response: np.ndarray, top_k: int, radius: int):
        work = response.astype(np.float32, copy=True)
        candidates = []
        for _ in range(top_k):
            _, max_value, _, max_location = cv2.minMaxLoc(work)
            if not np.isfinite(max_value):
                break
            x, y = int(max_location[0]), int(max_location[1])
            candidates.append((x, y, float(max_value)))
            x0, x1 = max(0, x - radius), min(work.shape[1], x + radius + 1)
            y0, y1 = max(0, y - radius), min(work.shape[0], y + radius + 1)
            work[y0:y1, x0:x1] = -np.inf
        return candidates

    def test_block_heap_nms_matches_full_rescan_exactly(self) -> None:
        rng = np.random.default_rng(20260818)
        for shape, radius, top_k in (
            ((31, 47), 3, 12),
            ((257, 513), 17, 30),
            ((620, 781), 41, 30),
        ):
            response = rng.normal(size=shape).astype(np.float32)
            expected = self.reference_nms(response, top_k, radius)
            actual = [
                (candidate.x, candidate.y, candidate.score)
                for candidate in nms_top_candidates(response, top_k, radius)
            ]
            self.assertEqual(actual, expected)

    def test_block_heap_nms_preserves_row_major_ties(self) -> None:
        response = np.zeros((530, 530), dtype=np.float32)
        response[10, 300] = 1.0
        response[10, 20] = 1.0
        response[300, 10] = 1.0
        expected = self.reference_nms(response, top_k=8, radius=64)
        actual = [
            (candidate.x, candidate.y, candidate.score)
            for candidate in nms_top_candidates(response, top_k=8, radius=64)
        ]
        self.assertEqual(actual, expected)

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
    def test_raw_result_schema_uses_only_the_25m_threshold(self) -> None:
        self.assertIn("success_25m", RESULT_COLUMNS)
        self.assertNotIn("success_30m", RESULT_COLUMNS)

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
            for variant, error in (("clean", 2.0), ("hard_v1", 27.0))
        ]
        summary = aggregate_results(rows, bootstrap_iterations=0)
        self.assertEqual([row["query_variant"] for row in summary], ["clean", "hard_v1"])
        self.assertEqual(summary[0]["success_25m"], 1.0)
        self.assertEqual(summary[1]["success_25m"], 0.0)

    def test_summary_records_latest_result_timestamp_for_each_group(self) -> None:
        rows = [
            {"direction": "A", "model_id": "M", "status": "ok", "error_m": 2.0,
             "created_at_utc": stamp}
            for stamp in ("2026-08-15T10:00:00+00:00", "2026-08-15T10:05:00+00:00")
        ]
        summary = aggregate_results(rows, bootstrap_iterations=0)
        self.assertEqual(summary[0]["result_completed_at_utc"], "2026-08-15T10:05:00+00:00")

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
        self.assertEqual(summary["success_25m_queries"], 2)
        self.assertAlmostEqual(summary["success_25m"], 2 / 3)
        self.assertAlmostEqual(summary["success_25m_failure_rate"], 1 / 3)
        self.assertAlmostEqual(summary["mean_error_under_25m"], 10.0)
        self.assertAlmostEqual(summary["median_error_under_25m"], 10.0)
        self.assertAlmostEqual(summary["auc_25m"], 2 / 5)

    def test_large_miss_does_not_dominate_primary_25m_score(self) -> None:
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
        self.assertAlmostEqual(summary["success_25m"], 0.5)
        self.assertAlmostEqual(summary["mean_error_under_25m"], 10.0)
        self.assertAlmostEqual(summary["median_error_under_25m"], 10.0)
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
        self.assertEqual(summary["success_25m"], 0.0)
        self.assertEqual(summary["success_25m_ci95_low"], 0.0)
        self.assertEqual(summary["success_25m_ci95_high"], 0.0)


class ParallelSearchTests(unittest.TestCase):
    def test_parallel_search_matches_serial_numeric_results(self) -> None:
        rng = np.random.default_rng(90210)
        map_gray = rng.integers(0, 256, size=(384, 448), dtype=np.uint8)
        transform = Affine(0.3, 0, 600000, 0, -0.3, 4200000)
        factors = (4, 2, 1)
        prepared = PreparedSearchMap(map_gray, transform, build_pyramid(map_gray, factors))
        locations = [(45, 55), (90, 210), (205, 120), (280, 340)]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries: list[QueryRecord] = []
            for index, (row, col) in enumerate(locations, start=1):
                tile = map_gray[row : row + 64, col : col + 64]
                tile_path = root / f"Q{index:05d}.png"
                self.assertTrue(cv2.imwrite(str(tile_path), tile))
                queries.append(
                    QueryRecord(
                        query_id=f"Q{index:05d}",
                        block_id="B00_00",
                        center_easting_m=600000 + (col + 32) * 0.3,
                        center_northing_m=4200000 - (row + 32) * 0.3,
                        source_row=row + 32,
                        source_col=col + 32,
                        query_std=float(np.std(tile)),
                        query_entropy=7.0,
                        dark_fraction=0.0,
                        raw_tile_file=str(tile_path),
                    )
                )

            outputs: dict[int, list[dict[str, object]]] = {}
            for workers in (1, 2):
                result_dir = root / f"workers_{workers}"
                jsonl = result_dir / "results.jsonl"
                run_searches_for_representation(
                    run_id="parallel_equality",
                    direction="synthetic",
                    query_variant="clean",
                    model_id="RAW_BASELINE",
                    model_file="",
                    model_sha256="",
                    map_path=root / "map.tif",
                    prepared_map=prepared,
                    map_build_seconds=0.0,
                    queries=queries,
                    query_paths=None,
                    query_inference_total_seconds=0.0,
                    crop_border=0,
                    search_modes=(("global", None),),
                    factors=factors,
                    top_k=5,
                    refine_radius_px=32,
                    nms_radius_px=16,
                    normalization="RAW",
                    source_query_raster=root / "query.tif",
                    source_map_raster=root / "map.tif",
                    results_jsonl=jsonl,
                    done=set(),
                    search_workers=workers,
                )
                outputs[workers] = [json.loads(line) for line in jsonl.read_text().splitlines()]

            ignored = {"created_at_utc", "search_seconds"}
            serial = [{k: v for k, v in row.items() if k not in ignored} for row in outputs[1]]
            parallel = [{k: v for k, v in row.items() if k not in ignored} for row in outputs[2]]
            self.assertEqual(serial, parallel)


class PerformanceEquivalenceTests(unittest.TestCase):
    class IdentityGrayRuntime:
        requested_output_value_mode = "raw"

        def predict_images(self, images, *, image_size, source_color="rgb"):
            self.assert_source_color = source_color
            return [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]

    def _write_rgb_raster(self, path: Path, *, width: int, height: int) -> np.ndarray:
        rng = np.random.default_rng(2026)
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs="EPSG:32636",
            transform=Affine(0.3, 0, 600000, 0, -0.3, 4200000),
        ) as dataset:
            dataset.write(np.moveaxis(rgb, -1, 0))
        return rgb

    def test_streaming_map_is_pixel_equal_to_corrected_legacy_placement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tif"
            rgb = self._write_rgb_raster(source, width=170, height=150)
            model = root / "identity.keras"
            model.write_bytes(b"fake")
            tile_size = 64
            overlap = 16
            metadata = streaming_map_metadata(source, tile_size=tile_size, overlap=overlap)
            output, _ = build_model_map(
                model,
                root / "model",
                root / "unused_tiles",
                metadata,
                source,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=3,
                normalization="minus1_1",
                enhancement="none",
                force=True,
                keep_intermediate=False,
                model_runtime=self.IdentityGrayRuntime(),
                model_sha256="fake-sha",
            )
            expected = np.zeros(rgb.shape[:2], dtype=np.uint8)
            crop = overlap // 2
            y_starts = compute_starts(rgb.shape[0], tile_size, tile_size - overlap)
            x_starts = compute_starts(rgb.shape[1], tile_size, tile_size - overlap)
            for i, start_y in enumerate(y_starts):
                for j, start_x in enumerate(x_starts):
                    prediction = cv2.cvtColor(
                        rgb[start_y : start_y + tile_size, start_x : start_x + tile_size],
                        cv2.COLOR_RGB2GRAY,
                    )
                    top = crop if i > 0 else 0
                    bottom = crop if i < len(y_starts) - 1 else 0
                    left = crop if j > 0 else 0
                    right = crop if j < len(x_starts) - 1 else 0
                    y2 = prediction.shape[0] - bottom if bottom else prediction.shape[0]
                    x2 = prediction.shape[1] - right if right else prediction.shape[1]
                    patch_data = prediction[top:y2, left:x2]
                    dest_y, dest_x = start_y + top, start_x + left
                    expected[
                        dest_y : dest_y + patch_data.shape[0],
                        dest_x : dest_x + patch_data.shape[1],
                    ] = patch_data
            with rasterio.open(output) as dataset:
                actual = dataset.read(1)
                self.assertEqual(dataset.transform, Affine(0.3, 0, 600000, 0, -0.3, 4200000))
                self.assertEqual(str(dataset.crs), "EPSG:32636")
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(list((root / "model").rglob("*.png")), [])

    def test_three_channel_streaming_gray_matches_legacy_bgr_conversion(self) -> None:
        prediction = np.asarray(
            [[[10, 20, 200], [200, 20, 10]], [[0, 100, 255], [255, 100, 0]]],
            dtype=np.uint8,
        )
        expected = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
        self.assertTrue(
            np.array_equal(model_prediction_to_legacy_gray(prediction), expected)
        )

    def test_clean_and_hard_share_one_roi_pyramid_with_numeric_equality(self) -> None:
        rng = np.random.default_rng(77)
        map_gray = rng.integers(0, 256, size=(320, 352), dtype=np.uint8)
        transform = Affine(1, 0, 500000, 0, -1, 4200000)
        row, col, size = 120, 140, 64
        tile = map_gray[row : row + size, col : col + size]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for variant in ("clean", "hard_v1"):
                path = root / f"{variant}.png"
                self.assertTrue(cv2.imwrite(str(path), tile))
                records.append(
                    (
                        variant,
                        [
                            QueryRecord(
                                query_id="Q00001",
                                block_id="B1",
                                center_easting_m=500000 + col + size / 2,
                                center_northing_m=4200000 - row - size / 2,
                                source_row=row + size // 2,
                                source_col=col + size // 2,
                                query_std=float(np.std(tile)),
                                query_entropy=7.0,
                                dark_fraction=0.0,
                                raw_tile_file=str(path),
                            )
                        ],
                        None,
                        0.0,
                    )
                )
            prepared = PreparedSearchMap(map_gray, transform, None)
            shared_jsonl = root / "shared.jsonl"
            original_build = build_pyramid
            with patch(
                "geospatial_model_benchmark.build_pyramid",
                wraps=original_build,
            ) as pyramid_builder:
                run_searches_for_variants(
                    run_id="shared",
                    direction="synthetic",
                    model_id="RAW_BASELINE",
                    model_file="",
                    model_sha256="",
                    map_path=root / "map.tif",
                    prepared_map=prepared,
                    map_build_seconds=0.0,
                    variant_inputs=records,
                    crop_border=0,
                    search_modes=(("roi_100m", 100.0),),
                    factors=(4, 2, 1),
                    top_k=5,
                    refine_radius_px=32,
                    nms_radius_px=16,
                    normalization="RAW",
                    source_query_raster=root / "query.tif",
                    source_map_raster=root / "map.tif",
                    results_jsonl=shared_jsonl,
                    done=set(),
                    search_workers=1,
                )
            self.assertEqual(pyramid_builder.call_count, 1)
            shared_rows = read_jsonl(shared_jsonl)
            self.assertEqual(len(shared_rows), 2)
            ignored = {"query_variant", "created_at_utc", "search_seconds"}
            first = {key: value for key, value in shared_rows[0].items() if key not in ignored}
            second = {key: value for key, value in shared_rows[1].items() if key not in ignored}
            self.assertEqual(first, second)

            legacy_jsonl = root / "legacy.jsonl"
            for variant, variant_records, _, _ in records:
                run_searches_for_representation(
                    run_id="shared",
                    direction="synthetic",
                    query_variant=variant,
                    model_id="RAW_BASELINE",
                    model_file="",
                    model_sha256="",
                    map_path=root / "map.tif",
                    prepared_map=prepared,
                    map_build_seconds=0.0,
                    queries=variant_records,
                    query_paths=None,
                    query_inference_total_seconds=0.0,
                    crop_border=0,
                    search_modes=(("roi_100m", 100.0),),
                    factors=(4, 2, 1),
                    top_k=5,
                    refine_radius_px=32,
                    nms_radius_px=16,
                    normalization="RAW",
                    source_query_raster=root / "query.tif",
                    source_map_raster=root / "map.tif",
                    results_jsonl=legacy_jsonl,
                    done=set(),
                    search_workers=1,
                )
            legacy_rows = read_jsonl(legacy_jsonl)
            allowed_differences = {"created_at_utc", "search_seconds"}
            self.assertEqual(
                [
                    {key: value for key, value in row.items() if key not in allowed_differences}
                    for row in shared_rows
                ],
                [
                    {key: value for key, value in row.items() if key not in allowed_differences}
                    for row in legacy_rows
                ],
            )

    def test_one_model_runtime_is_loaded_once_for_map_clean_and_hard(self) -> None:
        class FakeRuntime:
            load_calls = 0
            close_calls = 0

            @classmethod
            def load(cls, *args, **kwargs):
                cls.load_calls += 1
                return cls()

            def close(self):
                type(self).close_calls += 1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.tif"
            map_path = root / "map.tif"
            self._write_rgb_raster(query, width=128, height=128)
            self._write_rgb_raster(map_path, width=128, height=128)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "model.keras").write_bytes(b"fake")
            tile_path = root / "Q00001.png"
            cv2.imwrite(str(tile_path), np.full((64, 64), 128, dtype=np.uint8))
            record = QueryRecord(
                query_id="Q00001",
                block_id="B1",
                center_easting_m=600010.0,
                center_northing_m=4199990.0,
                source_row=32,
                source_col=32,
                query_std=20.0,
                query_entropy=5.0,
                dark_fraction=0.0,
                raw_tile_file=str(tile_path),
            )
            args = SimpleNamespace(
                tile_size=64,
                overlap=16,
                samples_per_block=1,
                block_size_m=50.0,
                max_queries=1,
                seed=42,
                min_query_std=1.0,
                min_query_entropy=1.0,
                max_dark_fraction=1.0,
                query_edge_buffer_m=0.0,
                force_queries=False,
                query_variants=("clean", "hard_v1"),
                include_models=True,
                include_raw=False,
                models=None,
                model_dir=model_dir,
                max_models=None,
                run_id="runtime_once",
                search_modes=(("global", None),),
                pyramid_factors=(4, 2, 1),
                top_k=5,
                refine_radius_px=16,
                nms_radius_px=16,
                batch_size=16,
                normalization="minus1_1",
                enhancement="none",
                output_value_mode="auto",
                force_maps=False,
                keep_intermediate=False,
                crop_border=8,
                search_workers=1,
                fail_fast=True,
                excel_update="model",
                excel_engine="openpyxl",
            )
            prepared = PreparedSearchMap(np.zeros((128, 128), dtype=np.uint8), Affine.identity(), {})
            with (
                patch("geospatial_model_benchmark.generate_query_manifest", return_value=[record]),
                patch(
                    "geospatial_model_benchmark.prepare_query_variants",
                    return_value={"clean": [record], "hard_v1": [record]},
                ),
                patch("geospatial_model_benchmark.import_loaded_model_runtime", return_value=FakeRuntime),
                patch("geospatial_model_benchmark.build_model_map", return_value=(map_path, 1.0)),
                patch("geospatial_model_benchmark.prepare_search_map", return_value=prepared),
                patch(
                    "geospatial_model_benchmark.build_model_queries",
                    return_value=({record.query_id: tile_path}, 1.0),
                ) as build_queries,
                patch("geospatial_model_benchmark.run_searches_for_variants") as shared_search,
                patch("geospatial_model_benchmark.refresh_excel_after_model"),
            ):
                run_direction(args, root / "run", query, map_path)
            self.assertEqual(FakeRuntime.load_calls, 1)
            self.assertEqual(FakeRuntime.close_calls, 1)
            self.assertEqual(build_queries.call_count, 2)
            self.assertEqual(shared_search.call_count, 1)

    def test_progress_info_is_throttled_below_result_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tile = np.arange(64, dtype=np.uint8).reshape(8, 8)
            variant_inputs = []
            for variant in ("clean", "hard_v1"):
                records = []
                for index in range(30):
                    path = root / f"{variant}_{index}.png"
                    cv2.imwrite(str(path), tile)
                    records.append(
                        QueryRecord(
                            query_id=f"Q{index:05d}",
                            block_id="B1",
                            center_easting_m=32.0,
                            center_northing_m=-32.0,
                            source_row=32,
                            source_col=32,
                            query_std=float(np.std(tile)),
                            query_entropy=5.0,
                            dark_fraction=0.0,
                            raw_tile_file=str(path),
                        )
                    )
                variant_inputs.append((variant, records, None, 0.0))
            prepared = PreparedSearchMap(
                np.zeros((64, 64), dtype=np.uint8),
                Affine.identity(),
                {1: np.zeros((64, 64), dtype=np.uint8)},
            )
            fake_outcome = SimpleNamespace(
                x=28,
                y=28,
                top1_score=1.0,
                top2_score=0.5,
                peak_margin=0.5,
                psr=10.0,
            )
            with (
                patch("geospatial_model_benchmark.coarse_to_fine_search", return_value=fake_outcome),
                patch("geospatial_model_benchmark.LOG.info") as info,
            ):
                run_searches_for_variants(
                    run_id="progress",
                    direction="synthetic",
                    model_id="RAW_BASELINE",
                    model_file="",
                    model_sha256="",
                    map_path=root / "map.tif",
                    prepared_map=prepared,
                    map_build_seconds=0.0,
                    variant_inputs=variant_inputs,
                    crop_border=0,
                    search_modes=(("global", None),),
                    factors=(1,),
                    top_k=2,
                    refine_radius_px=8,
                    nms_radius_px=8,
                    normalization="RAW",
                    source_query_raster=root / "query.tif",
                    source_map_raster=root / "map.tif",
                    results_jsonl=root / "progress.jsonl",
                    done=set(),
                    search_workers=1,
                )
            progress_calls = [
                call for call in info.call_args_list if call.args and call.args[0].startswith("SHARED PROGRESS")
            ]
            self.assertLess(len(progress_calls), 60)
            self.assertLessEqual(len(progress_calls), 5)


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
        self.assertGreater(float(np.mean(np.abs(first.astype(float) - image.astype(float)))), 1.0)
        self.assertIn(first_params["scenario"], HARD_V1_PROFILE["scenarios"])
        self.assertEqual(first_params["profile_revision"], "uav_camera_v1")

    def test_hard_v1_uav_scenarios_are_represented_without_geometry(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        outputs = [augment_hard_v1(image, seed=seed)[1] for seed in range(100)]
        scenarios = {row["scenario"] for row in outputs}
        self.assertEqual(scenarios, set(HARD_V1_PROFILE["scenarios"]))
        self.assertAlmostEqual(sum(HARD_V1_PROFILE["scenarios"].values()), 1.0)
        for row in outputs:
            self.assertNotIn("rotation_deg", row)
            self.assertNotIn("scale", row)
            self.assertNotIn("perspective_x_per_px", row)

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

    def test_manifest_regenerates_when_max_queries_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.tif"
            map_path = root / "map.tif"
            self._write_raster(query, 5)
            self._write_raster(map_path, 6)
            common = dict(
                tile_size=64,
                samples_per_block=2,
                block_size_m=60.0,
                seed=42,
                min_std=1.0,
                min_entropy=1.0,
                max_dark_fraction=1.0,
                edge_buffer_m=None,
                force=False,
            )
            first = generate_query_manifest(
                query, map_path, root / "queries", max_queries=2, **common
            )
            second = generate_query_manifest(
                query, map_path, root / "queries", max_queries=4, **common
            )
            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 4)

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
