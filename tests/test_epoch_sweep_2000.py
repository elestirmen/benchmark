from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import rasterio
import cv2
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import epoch_sweep_2000 as sweep  # noqa: E402
from geospatial_model_benchmark import (  # noqa: E402
    QueryRecord,
    generate_query_manifest,
    prepare_query_variants,
    resume_payloads_compatible,
)


def ranked_model(
    rank: int,
    model_id: str,
    sha: str,
    *,
    score: float = 0.9,
    relative_path: str = "model_epoch_00001.h5",
    duplicate_paths: tuple[str, ...] = (),
) -> sweep.RankedModel:
    return sweep.RankedModel(
        rank=rank,
        model_id=model_id,
        model_file="",
        model_sha256=sha,
        source_series_id="series",
        source_series_key="series/model_epoch_{n}",
        source_relative_path=relative_path,
        source_checkpoint_number=1,
        completed_directions=("A", "B"),
        clean_success_25m=score,
        hard_success_25m=score,
        mean_success_25m=score,
        clean_auc_25m=score,
        hard_auc_25m=score,
        mean_auc_25m=score,
        clean_success_5m=score,
        hard_success_5m=score,
        mean_success_5m=score,
        median_error_under_25m=2.0,
        source_duplicate_paths=duplicate_paths,
    )


def archive_checkpoint(
    path: Path,
    root: Path,
    epoch: int,
    sha: str,
    key: str,
) -> sweep.ArchiveCheckpoint:
    return sweep.ArchiveCheckpoint(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        epoch=epoch,
        sha256=sha,
        normalized_stem="model_epoch_{epoch}",
        lineage_key=key,
    )


class RankingTests(TestCase):
    def test_top10_uses_complete_global_cells_and_requested_order(self) -> None:
        config = {
            "max_queries": 300,
            "query_raster": "query.tif",
            "map_raster": "map.tif",
            "bidirectional": True,
        }
        directions = sweep.configured_directions(config)
        summary = []
        identities = {}
        for index in range(11):
            model_id = f"M{index:02d}"
            identities[model_id] = {
                "model_id": model_id,
                "path": "",
                "relative_path": f"{model_id}_epoch_1.h5",
                "sha256": f"{index + 1:064x}",
            }
            score = 0.99 - index * 0.01
            for direction in directions:
                for variant in ("clean", "hard_v1"):
                    summary.append(
                        {
                            "direction": direction,
                            "query_variant": variant,
                            "search_mode": "global",
                            "model_id": model_id,
                            "total_queries": 300,
                            "success_25m": score,
                            "auc_25m": score - 0.01,
                            "success_5m": score - 0.02,
                            "success_10m": score - 0.01,
                            "median_error_under_25m": 2.0 + index,
                        }
                    )
            summary.append(
                {
                    **summary[-1],
                    "search_mode": "roi_500m",
                    "success_25m": 1.0,
                }
            )
        ranked = sweep.rank_top_models(config, summary, identities)
        self.assertEqual([item.model_id for item in ranked], [f"M{i:02d}" for i in range(10)])
        self.assertEqual([item.rank for item in ranked], list(range(1, 11)))

    def test_model_with_partial_second_direction_is_excluded(self) -> None:
        config = {
            "max_queries": 300,
            "query_raster": "query.tif",
            "map_raster": "map.tif",
            "bidirectional": True,
        }
        directions = sweep.configured_directions(config)
        summary = []
        identities = {}
        for index in range(10):
            model_id = f"M{index}"
            identities[model_id] = {"sha256": f"{index + 1:064x}", "relative_path": f"m{index}_epoch_1.h5"}
            for variant in ("clean", "hard_v1"):
                summary.append(
                    {
                        "direction": directions[0], "query_variant": variant, "search_mode": "global",
                        "model_id": model_id, "total_queries": 300, "success_25m": 0.8,
                        "auc_25m": 0.7, "success_5m": 0.6, "success_10m": 0.7,
                        "median_error_under_25m": 3.0,
                    }
                )
                summary.append({**summary[-1], "direction": directions[1], "total_queries": 299, "success_25m": 0.0})
        with self.assertRaisesRegex(ValueError, "En az 10 tamamlanmış model"):
            sweep.rank_top_models(config, summary, identities)


class ArchiveLineageTests(TestCase):
    def test_missing_sha_never_falls_back_to_same_named_other_run(self) -> None:
        ranked = ranked_model(1, "selected", "f" * 64)
        path = Path("archive/run_b/model_epoch_00001.h5")
        checkpoint = sweep.ArchiveCheckpoint(
            path=path,
            relative_path="run_b/model_epoch_00001.h5",
            epoch=1,
            sha256="e" * 64,
            normalized_stem="model_epoch_{epoch}",
            lineage_key="run_b/model_epoch_{epoch}",
        )
        with self.assertRaisesRegex(ValueError, "SHA256 ankrajı"):
            sweep.resolve_archive_lineage(
                ranked,
                {checkpoint.lineage_key: [checkpoint]},
                {checkpoint.sha256: {checkpoint.lineage_key}},
            )

    def test_epoch_parser_supports_legacy_separators_and_excludes_step_batch(self) -> None:
        self.assertEqual(sweep.normalized_epoch_stem(Path("x_epoch_00007.h5"))[0], 7)
        self.assertEqual(sweep.normalized_epoch_stem(Path("x_EPOCH-12.h5"))[0], 12)
        self.assertEqual(sweep.normalized_epoch_stem(Path("x_epoch 3.keras"))[0], 3)
        self.assertIsNone(sweep.normalized_epoch_stem(Path("x_step_500.h5")))
        self.assertIsNone(sweep.normalized_epoch_stem(Path("x_batch_10.h5")))
        self.assertIsNone(sweep.normalized_epoch_stem(Path("epoch_1_epoch_2.h5")))

    def test_archive_scan_uses_filename_not_epoch_text_in_parent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "run_epoch_999.h5"
            parent.mkdir()
            (parent / "model_epoch_00002.h5").write_bytes(b"epoch")
            (parent / "model_step_500.h5").write_bytes(b"step")
            groups, _ = sweep.scan_epoch_archive(root)
            checkpoints = next(iter(groups.values()))
            self.assertEqual([item.epoch for item in checkpoints], [2])

    def test_sha_anchor_prefers_richer_mirror_and_top10_is_not_backfilled(self) -> None:
        sha_a = "a" * 64
        sha_b = "b" * 64
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rich_paths = []
            for epoch, sha in ((1, sha_a), (2, sha_b)):
                path = root / "rich" / f"model_epoch_{epoch:05d}.h5"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(bytes.fromhex(sha[:2]))
                rich_paths.append(archive_checkpoint(path, root, epoch, sha, "rich/model_epoch_{epoch}"))
            mirror_path = root / "mirror" / "model_epoch_00001.h5"
            mirror_path.parent.mkdir()
            mirror_path.write_bytes(b"x")
            mirror = archive_checkpoint(mirror_path, root, 1, sha_a, "mirror/model_epoch_{epoch}")
            groups = {"rich/model_epoch_{epoch}": rich_paths, "mirror/model_epoch_{epoch}": [mirror]}
            sha_index = {sha_a: set(groups), sha_b: {"rich/model_epoch_{epoch}"}}
            ranked = [
                ranked_model(
                    1,
                    "M1",
                    sha_a,
                    relative_path="mirror/model_epoch_00001.h5",
                    duplicate_paths=("rich/model_epoch_00001.h5",),
                ),
                ranked_model(
                    2,
                    "M2",
                    sha_b,
                    relative_path="rich/model_epoch_00002.h5",
                ),
            ]
            lineages = sweep.resolve_selected_lineages(ranked, groups, sha_index)
            self.assertEqual(len(lineages), 1)
            self.assertEqual(lineages[0].lineage_key, "rich/model_epoch_{epoch}")
            self.assertEqual([item.rank for item in lineages[0].selected_models], [1, 2])

    def test_shared_early_sha_does_not_merge_independent_runs(self) -> None:
        sha_a = "a" * 64
        sha_b = "b" * 64
        run_a = [
            sweep.ArchiveCheckpoint(
                path=Path("archive/run_a/model_epoch_1.h5"),
                relative_path="run_a/model_epoch_1.h5",
                epoch=1,
                sha256=sha_a,
                normalized_stem="model_epoch_{epoch}",
                lineage_key="run_a/model_epoch_{epoch}",
            )
        ]
        run_b = [
            sweep.ArchiveCheckpoint(
                path=Path(f"archive/run_b/model_epoch_{epoch}.h5"),
                relative_path=f"run_b/model_epoch_{epoch}.h5",
                epoch=epoch,
                sha256=sha,
                normalized_stem="model_epoch_{epoch}",
                lineage_key="run_b/model_epoch_{epoch}",
            )
            for epoch, sha in ((1, sha_a), (2, sha_b))
        ]
        ranked = ranked_model(1, "selected", sha_a)
        with self.assertRaisesRegex(ValueError, "bağımsız/eşit lineage"):
            sweep.resolve_archive_lineage(
                ranked,
                {
                    "run_a/model_epoch_{epoch}": run_a,
                    "run_b/model_epoch_{epoch}": run_b,
                },
                {
                    sha_a: {
                        "run_a/model_epoch_{epoch}",
                        "run_b/model_epoch_{epoch}",
                    }
                },
            )

    def test_same_lineage_epoch_different_sha_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "run" / "model_epoch_1.h5"
            second = root / "run" / "model_epoch_1.keras"
            first.parent.mkdir()
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            key = "run/model_epoch_{epoch}"
            checkpoints = [
                archive_checkpoint(first, root, 1, "1" * 64, key),
                archive_checkpoint(second, root, 1, "2" * 64, key),
            ]
            with self.assertRaisesRegex(ValueError, "farklı SHA256"):
                sweep.lineage_checkpoint_plan(
                    sweep.ResolvedLineage(key, "lineage", checkpoints, [ranked_model(1, "M", "1" * 64)])
                )


class SamplingTests(TestCase):
    def test_legacy_hard_cache_is_upgraded_without_changing_pixels_on_resume(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_path = root / "clean.png"
            self.assertTrue(
                cv2.imwrite(str(clean_path), np.full((32, 32, 3), 90, dtype=np.uint8))
            )
            record = QueryRecord(
                "Q00001", "B00_00", 100.0, 200.0, 16, 16,
                0.0, 0.0, 0.0, str(clean_path),
            )
            hard = prepare_query_variants(
                [record], root, variants=("hard_v1",), seed=42, force=False
            )["hard_v1"][0]
            hard_path = Path(hard.raw_tile_file)
            before = hard_path.read_bytes()
            manifest_path = root / "variants" / "hard_v1" / "query_variant_manifest.json"
            legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
            legacy["schema_version"] = 1
            legacy.pop("base_queries_sha256", None)
            legacy["queries"][0].pop("raw_tile_sha256", None)
            manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

            prepare_query_variants(
                [record], root, variants=("hard_v1",), seed=42, force=False,
                resume_has_results=True,
            )
            upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertIn("base_queries_sha256", upgraded)
            self.assertEqual(hard_path.read_bytes(), before)

            invalid = dict(upgraded)
            invalid["base_queries_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "kriptografik"):
                prepare_query_variants(
                    [record], root, variants=("hard_v1",), seed=42, force=False,
                    resume_has_results=True,
                )
            self.assertEqual(hard_path.read_bytes(), before)

    def test_legacy_full_catalog_resume_is_compatible_only_for_same_inventory(self) -> None:
        previous = {
            "schema_version": 2,
            "model_catalog_schema_version": 1,
            "model_inventory_count": 2,
            "model_inventory_sha256": "a" * 64,
        }
        current = {
            **previous,
            "model_catalog_schema_version": 2,
            "model_sampling": "full",
            "query_sampling": "block_sequential",
        }
        self.assertTrue(resume_payloads_compatible(previous, current))
        self.assertFalse(
            resume_payloads_compatible(
                previous, {**current, "model_inventory_sha256": "b" * 64}
            )
        )

    def test_hard_cache_is_invalidated_when_base_tile_changes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_path = root / "clean.png"
            first_pixels = np.full((32, 32, 3), 60, dtype=np.uint8)
            second_pixels = np.full((32, 32, 3), 190, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(clean_path), first_pixels))

            def record() -> QueryRecord:
                return QueryRecord(
                    query_id="Q00001",
                    block_id="B00_00",
                    center_easting_m=100.0,
                    center_northing_m=200.0,
                    source_row=16,
                    source_col=16,
                    query_std=0.0,
                    query_entropy=0.0,
                    dark_fraction=0.0,
                    raw_tile_file=str(clean_path),
                )

            first = prepare_query_variants(
                [record()], root, variants=("hard_v1",), seed=42, force=False
            )["hard_v1"][0]
            first_hard_sha = sweep.core.sha256_file(Path(first.raw_tile_file))
            self.assertTrue(cv2.imwrite(str(clean_path), second_pixels))
            second = prepare_query_variants(
                [record()], root, variants=("hard_v1",), seed=42, force=False
            )["hard_v1"][0]
            second_hard_sha = sweep.core.sha256_file(Path(second.raw_tile_file))
            self.assertNotEqual(first_hard_sha, second_hard_sha)

    def test_balanced_exact_manifest_has_exact_count_and_repeatable_centres(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster_path = root / "raster.tif"
            data = np.random.default_rng(7).integers(0, 256, size=(3, 256, 256), dtype=np.uint8)
            with rasterio.open(
                raster_path, "w", driver="GTiff", width=256, height=256, count=3,
                dtype="uint8", crs="EPSG:32636", transform=from_origin(0, 256, 1, 1),
            ) as dataset:
                dataset.write(data)
            kwargs = dict(
                tile_size=32, samples_per_block=1, block_size_m=80.0, max_queries=17,
                seed=42, min_std=0.0, min_entropy=0.0, max_dark_fraction=1.0,
                edge_buffer_m=0.0, force=False, sampling_strategy="balanced_exact",
            )
            first = generate_query_manifest(raster_path, raster_path, root / "first", **kwargs)
            second = generate_query_manifest(raster_path, raster_path, root / "second", **kwargs)
            self.assertEqual(len(first), 17)
            self.assertEqual([sweep._record_centre(item) for item in first], [sweep._record_centre(item) for item in second])
            counts = {}
            for item in first:
                counts[item.block_id] = counts.get(item.block_id, 0) + 1
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            payload = json.loads((root / "first" / "query_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["sampling_strategy"], "balanced_exact")


class StagingAndResumeTests(TestCase):
    def test_interim_exact_summary_is_not_completion_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "summary_metadata.json").write_text(
                json.dumps({"incremental_equivalence": "exact_match"}),
                encoding="utf-8",
            )
            self.assertFalse(sweep._source_has_completion_evidence(run_dir))
            (run_dir / "benchmark.log").write_text(
                "ara ozet tamamlandi\nBENCHMARK TAMAMLANDI\n", encoding="utf-8"
            )
            self.assertTrue(sweep._source_has_completion_evidence(run_dir))

    def test_source_output_archive_and_staging_paths_must_be_disjoint(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            child = parent / "child"
            sibling = root / "sibling"
            self.assertTrue(sweep._paths_overlap(parent, child))
            self.assertTrue(sweep._paths_overlap(child, parent))
            self.assertFalse(sweep._paths_overlap(parent, sibling))

    def test_scientific_contract_rejects_source_semantics_or_hard_profile_drift(self) -> None:
        config = {
            "scientific_semantics_version": sweep.core.SCIENTIFIC_SEMANTICS_VERSION,
            "hard_v1_profile": sweep.core.HARD_V1_PROFILE,
        }
        with self.assertRaisesRegex(ValueError, "semantics"):
            sweep.scientific_contract(
                {**config, "scientific_semantics_version": -1}, {}
            )
        with self.assertRaisesRegex(ValueError, "hard_v1"):
            sweep.scientific_contract({**config, "hard_v1_profile": {}}, {})

    def test_duplicate_sha_epochs_are_copied_once_and_all_aliases_are_manifested(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            staging = root / "staging"
            archive.mkdir()
            paths = []
            for epoch in (1, 2):
                path = archive / f"model_epoch_{epoch:05d}.h5"
                path.write_bytes(b"identical checkpoint")
                paths.append(path)
            sha = sweep.core.sha256_file(paths[0])
            key = "./model_epoch_{epoch}"
            checkpoints = [
                archive_checkpoint(path, archive, epoch, sha, key)
                for epoch, path in zip((1, 2), paths)
            ]
            lineage = sweep.ResolvedLineage(
                key, "lineage", checkpoints, [ranked_model(1, "M", sha)]
            )
            payload = sweep.prepare_staging([lineage], staging)
            model_files = [
                path for path in staging.rglob("*") if path.is_file() and path.suffix == ".h5"
            ]
            self.assertEqual(len(model_files), 1)
            manifest = payload["lineages"][0]
            self.assertEqual([item["epoch"] for item in manifest["checkpoints"]], [1, 2])
            self.assertFalse(manifest["checkpoints"][0]["duplicate_sha_alias"])
            self.assertTrue(manifest["checkpoints"][1]["duplicate_sha_alias"])
            self.assertEqual(
                manifest["checkpoints"][0]["copied_file"],
                manifest["checkpoints"][1]["copied_file"],
            )
            sweep.validate_staging(payload)
            second = sweep.prepare_staging([lineage], staging)
            self.assertEqual(payload, second)

    def test_manifest_fingerprint_detects_changed_centre(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            tile = root / "a.png"
            tile.write_bytes(b"query pixels")
            payload = {
                "created_at_utc": "old",
                "queries": [
                    {
                        "query_id": "Q1",
                        "center_easting_m": 1.0,
                        "raw_tile_file": str(tile),
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            before = sweep.manifest_fingerprint(path)
            payload["created_at_utc"] = "new"
            path.write_text(json.dumps(payload), encoding="utf-8")
            timestamp_only = sweep.manifest_fingerprint(path)
            self.assertEqual(before["scientific_sha256"], timestamp_only["scientific_sha256"])
            payload["queries"][0]["center_easting_m"] = 2.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            changed = sweep.manifest_fingerprint(path)
            self.assertNotEqual(before["scientific_sha256"], changed["scientific_sha256"])

    def test_dry_run_does_not_create_output_or_staging(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            archive = root / "archive"
            output = root / "output"
            staging = root / "staging"
            benchmark.mkdir()
            archive.mkdir()
            ranked = [ranked_model(index, f"M{index}", f"{index:064x}") for index in range(1, 11)]
            checkpoint_path = archive / "model_epoch_1.h5"
            checkpoint_path.write_bytes(b"model")
            checkpoint = archive_checkpoint(
                checkpoint_path, archive, 1, ranked[0].model_sha256, "model_epoch_{epoch}"
            )
            lineage = sweep.ResolvedLineage(
                "model_epoch_{epoch}", "lineage", [checkpoint], ranked
            )
            args = argparse.Namespace(
                benchmark_run=benchmark,
                model_archive_dir=archive,
                output_dir=output,
                staging_model_dir=staging,
                batch_size=None,
                search_workers=None,
                keep_maps=False,
                fail_fast=False,
                dry_run=True,
                verbose=False,
            )
            with (
                patch.object(sweep, "load_source_run", return_value=({}, [], {})),
                patch.object(sweep, "rank_top_models", return_value=ranked),
                patch.object(sweep, "scan_epoch_archive", return_value=({}, {})),
                patch.object(sweep, "resolve_selected_lineages", return_value=[lineage]),
                patch.object(sweep, "scientific_contract", return_value={}) as contract,
                patch.object(sweep, "prepare_staging") as prepare,
                patch.object(sweep, "prepare_query_manifest_index") as manifest,
                patch.object(sweep.core, "main") as benchmark_main,
            ):
                self.assertEqual(sweep.run(args), 0)
            self.assertFalse(output.exists())
            self.assertFalse(staging.exists())
            contract.assert_called_once()
            prepare.assert_not_called()
            manifest.assert_not_called()
            benchmark_main.assert_not_called()


class ContractAndReportTests(TestCase):
    def test_core_argv_locks_requested_scientific_settings(self) -> None:
        config = {
            "query_raster": "q.tif",
            "map_raster": "m.tif",
            "bidirectional": True,
            "block_size_m": 1000,
            "samples_per_block": 5,
            "seed": 42,
            "min_query_std": 12,
            "min_query_entropy": 4,
            "max_dark_fraction": 0.2,
            "pyramid_factors": [16, 8, 4, 2, 1],
            "bootstrap_iterations": 1000,
        }
        argv = sweep.build_core_argv(
            output_dir=Path("out"), staging_root=Path("models"), source_config=config,
            batch_size=16, search_workers=4, keep_maps=False, fail_fast=True,
        )
        joined = " ".join(argv)
        for expected in (
            "--max-queries 2000", "--query-sampling balanced_exact",
            "--query-variants clean,hard_v1", "--search-modes global",
            "--tile-size 544", "--crop-border 16", "--normalization minus1_1",
            "--pyramid-factors 16,8,4,2,1", "--top-k 30", "--include-raw",
            "--excel-update model", "--excel-report", "--no-results-csv",
            "--fail-fast",
        ):
            self.assertIn(expected, joined)
        self.assertNotIn("--no-excel-report", joined)

    def test_best_epoch_uses_full_tie_break_chain(self) -> None:
        base = {
            "training_lineage": "L",
            "lineage_rank": 1,
            "status": "complete",
            "mean_success_25m": 0.9,
            "mean_auc_25m": 0.8,
            "mean_success_5m": 0.7,
            "clean_hard_gap": 0.1,
            "mean_median_error_under_25m": 3.0,
        }
        rows = [
            {**base, "epoch": 1, "mean_success_25m": 0.89},
            {**base, "epoch": 2, "mean_auc_25m": 0.79},
            {**base, "epoch": 3, "mean_success_5m": 0.69},
            {**base, "epoch": 4, "clean_hard_gap": 0.09},
            {**base, "epoch": 5, "clean_hard_gap": 0.09, "mean_median_error_under_25m": 2.0},
        ]
        best = sweep.best_epoch_rows(rows)
        self.assertEqual(best[0]["epoch"], 5)

    def test_epoch_result_formulas_and_direction_averaging(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {
                "query_raster": "q.tif", "map_raster": "m.tif", "bidirectional": True,
            }
            directions = sweep.configured_directions(config)
            sha = "a" * 64
            catalog = {"models": [{"sha256": sha, "model_id": "EPOCH_MODEL"}]}
            (output / "model_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            summary = []
            result_rows = []
            for direction_index, direction in enumerate(directions):
                for variant, score in (("clean", 0.9), ("hard_v1", 0.7)):
                    summary.append(
                        {
                            "model_id": "EPOCH_MODEL", "direction": direction,
                            "query_variant": variant, "search_mode": "global", "total_queries": 2000,
                            "success_25m": score - direction_index * 0.1, "auc_25m": score - 0.1,
                            "success_5m": score - 0.2, "success_10m": score - 0.1,
                            "median_error_under_25m": 2.0 + direction_index,
                        }
                    )
                    result_rows.append(
                        {
                            "model_id": "EPOCH_MODEL",
                            "direction": direction,
                            "query_variant": variant,
                            "search_mode": "global",
                            "status": "ok",
                            "error_m": (
                                (1.0, 9.0)[direction_index]
                                if variant == "clean"
                                else (4.0, 8.0)[direction_index]
                            ),
                        }
                    )
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (output / "results.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in result_rows),
                encoding="utf-8",
            )
            staging = {
                "lineages": [
                    {
                        "training_lineage": "L", "lineage_rank": 1,
                        "checkpoints": [
                            {
                                "epoch": 1, "sha256": sha, "copied_file": "epoch_1.h5",
                                "canonical_epoch": 1, "duplicate_sha_alias": False,
                            }
                        ],
                    }
                ]
            }
            rows, _ = sweep.build_epoch_result_rows(output, staging, config)
            row = rows[0]
            self.assertAlmostEqual(row["clean_success_25m"], 0.85)
            self.assertAlmostEqual(row["hard_success_25m"], 0.65)
            self.assertAlmostEqual(row["mean_success_25m"], 0.75)
            self.assertAlmostEqual(row["clean_hard_gap"], 0.20)
            self.assertAlmostEqual(row["clean_median_error_under_25m"], 5.0)
            self.assertAlmostEqual(row["hard_median_error_under_25m"], 6.0)

    def test_plot_is_created_for_each_lineage(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [
                {
                    "training_lineage": "L", "epoch": epoch,
                    "clean_success_25m": 0.8, "hard_success_25m": 0.7,
                    "mean_success_25m": 0.75,
                }
                for epoch in (1, 2)
            ]
            sweep.write_epoch_plots(output, rows)
            plot = next((output / "plots").glob("*.png"))
            self.assertGreater(plot.stat().st_size, 100)
            self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


class FocusedHardeningTests(TestCase):
    def test_stale_staging_model_is_rejected_before_any_copy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            staging = root / "staging"
            archive.mkdir()
            staging.mkdir()
            source = archive / "model_epoch_00001.h5"
            source.write_bytes(b"expected checkpoint")
            sha = sweep.core.sha256_file(source)
            checkpoint = archive_checkpoint(
                source, archive, 1, sha, "model_epoch_{epoch}"
            )
            lineage = sweep.ResolvedLineage(
                "model_epoch_{epoch}",
                "lineage",
                [checkpoint],
                [ranked_model(1, "M", sha)],
            )
            stale = staging / "stale_epoch_99999.h5"
            stale.write_bytes(b"stale checkpoint")

            with patch.object(sweep, "_atomic_copy_verified") as copy_checkpoint:
                with self.assertRaisesRegex(RuntimeError, "Staging"):
                    sweep.prepare_staging([lineage], staging)

            copy_checkpoint.assert_not_called()
            self.assertEqual(stale.read_bytes(), b"stale checkpoint")
            self.assertEqual(
                [path for path in staging.rglob("*") if path.is_file()], [stale]
            )

    def test_manifest_fingerprint_detects_png_byte_tamper(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tile = root / "Q00001.png"
            tile.write_bytes(b"\x89PNG\r\n\x1a\noriginal-pixels")
            manifest = root / "query_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-12T00:00:00+00:00",
                        "queries": [
                            {
                                "query_id": "Q00001",
                                "block_id": "B00_00",
                                "center_easting_m": 1.0,
                                "center_northing_m": 2.0,
                                "raw_tile_file": str(tile),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            before = sweep.manifest_fingerprint(manifest)
            tile.write_bytes(b"\x89PNG\r\n\x1a\ntampered-pixels")
            after = sweep.manifest_fingerprint(manifest)

            self.assertEqual(before["file_sha256"], after["file_sha256"])
            self.assertEqual(before["scientific_sha256"], after["scientific_sha256"])
            self.assertNotEqual(before["tiles_sha256"], after["tiles_sha256"])

    def test_classify_sweep_outcome_full_partial_and_zero(self) -> None:
        cases = (
            ("full", 4, 4, ("complete", 0)),
            ("partial", 4, 3, ("complete_with_errors", 0)),
            ("zero", 4, 0, ("failed", 1)),
        )
        for label, expected_rows, complete_rows, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    sweep.classify_sweep_outcome(
                        core_return_code=0,
                        execution_error=None,
                        expected_epoch_rows=expected_rows,
                        complete_epoch_rows=complete_rows,
                    ),
                    expected,
                )

    def test_prepared_config_without_results_uses_read_only_validation_branch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            archive = root / "archive"
            output = root / "output"
            staging = root / "staging"
            for directory in (benchmark, archive, output, staging):
                directory.mkdir()
            prepared_config = output / "epoch_sweep_config.json"
            prepared_bytes = b'{"status":"prepared"}\n'
            prepared_config.write_bytes(prepared_bytes)

            sha = "a" * 64
            ranked = [ranked_model(1, "M", sha)]
            source = archive / "model_epoch_00001.h5"
            source.write_bytes(b"model")
            checkpoint = archive_checkpoint(
                source, archive, 1, sha, "model_epoch_{epoch}"
            )
            lineage = sweep.ResolvedLineage(
                "model_epoch_{epoch}", "lineage", [checkpoint], ranked
            )
            staging_payload = {
                "staging_root": str(staging),
                "lineages": [{"training_lineage": "lineage", "checkpoints": []}],
            }
            args = argparse.Namespace(
                benchmark_run=benchmark,
                model_archive_dir=archive,
                output_dir=output,
                staging_model_dir=staging,
                batch_size=16,
                search_workers=1,
                keep_maps=False,
                fail_fast=False,
                dry_run=False,
                verbose=False,
            )
            source_config = {
                "query_raster": "query.tif",
                "map_raster": "map.tif",
                "bidirectional": False,
            }

            with (
                patch.object(sweep, "load_source_run", return_value=(source_config, [], {})),
                patch.object(sweep, "rank_top_models", return_value=ranked),
                patch.object(sweep, "scan_epoch_archive", return_value=({}, {})),
                patch.object(sweep, "resolve_selected_lineages", return_value=[lineage]),
                patch.object(sweep, "plan_payload", return_value={}),
                patch.object(sweep, "scientific_contract", return_value={}),
                patch.object(sweep, "load_existing_staging", return_value=staging_payload) as load_staging,
                patch.object(sweep, "validate_query_manifest_index", return_value={}) as validate_queries,
                patch.object(sweep, "validate_staging", side_effect=RuntimeError("validation sentinel")) as validate_staging,
                patch.object(sweep, "prepare_staging") as prepare_staging,
                patch.object(sweep, "prepare_query_manifest_index") as prepare_queries,
            ):
                with self.assertRaisesRegex(RuntimeError, "validation sentinel"):
                    sweep.run(args)

            load_staging.assert_called_once_with([lineage], staging.resolve())
            validate_queries.assert_called_once_with(output.resolve())
            validate_staging.assert_called_once_with(staging_payload)
            prepare_staging.assert_not_called()
            prepare_queries.assert_not_called()
            self.assertEqual(prepared_config.read_bytes(), prepared_bytes)
            self.assertEqual(
                [path.relative_to(output) for path in output.rglob("*") if path.is_file()],
                [Path("epoch_sweep_config.json")],
            )
