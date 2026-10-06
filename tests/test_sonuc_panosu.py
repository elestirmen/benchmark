import json
import tempfile
import unittest
from pathlib import Path

import sonuc_panosu as pano

QUERY = r"C:\x\urgup_cevresi_gmap_utm_30cm.tif"
MAP = r"C:\x\urgup_cevresi_bingmap_utm_30cm.tif"
FORWARD = "urgup_cevresi_gmap_utm_30cm__TO__urgup_cevresi_bingmap_utm_30cm"
REVERSE = "urgup_cevresi_bingmap_utm_30cm__TO__urgup_cevresi_gmap_utm_30cm"


def summary_rows(model_id, values, max_queries=10, mode="global"):
    """values: (g2b_clean, g2b_hard, b2g_clean, b2g_hard) success_25m."""
    cells = [(FORWARD, "clean"), (FORWARD, "hard_v1"), (REVERSE, "clean"), (REVERSE, "hard_v1")]
    return [
        {
            "direction": direction, "query_variant": variant, "search_mode": mode,
            "model_id": model_id, "total_queries": max_queries, "ok_queries": max_queries,
            "rejected_queries": 0, "error_queries": 0, "success_25m": value,
            "result_completed_at_utc": "2026-09-25T00:00:00+00:00",
        }
        for (direction, variant), value in zip(cells, values)
    ]


def write_run(root, name, rows, models, max_queries=10, seed=42):
    run = root / name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({
        "query_raster": QUERY, "map_raster": MAP, "max_queries": max_queries, "seed": seed,
        "query_sampling": "balanced_exact", "normalization": "minus1_1",
    }), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(rows), encoding="utf-8")
    (run / "model_catalog.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    return run


class SonucPanosuTests(unittest.TestCase):
    def test_global_is_the_mean_of_four_groups_and_delta_uses_same_protocol_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = write_run(
                root, "a",
                summary_rows("RAW_BASELINE", (0.6, 0.6, 0.4, 0.4)) + summary_rows("m__abc", (0.8, 0.7, 0.7, 0.6)),
                [{"model_id": "m__abc", "sha256": "abc", "relative_path": "m.h5", "path": "C:/m.h5"}],
            )
            records = pano.dedupe_records(pano.read_benchmark_run(run))
            pano.attach_raw_deltas(records)
            model = next(r for r in records if not r["is_raw"])
            self.assertAlmostEqual(0.70, model["global"])
            self.assertAlmostEqual(0.20, model["delta_raw"])
            self.assertAlmostEqual(0.8, model["g2b_clean"])
            self.assertAlmostEqual(0.6, model["b2g_hard_v1"])

    def test_incomplete_groups_do_not_produce_a_global_score(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = summary_rows("m__abc", (0.8, 0.7, 0.7, 0.6))
            rows[-1]["total_queries"] = 5  # yarim kalmis grup
            run = write_run(Path(temp), "a", rows, [{"model_id": "m__abc", "sha256": "abc"}])
            record = pano.read_benchmark_run(run)[0]
            self.assertNotIn("global", record)
            self.assertEqual(3, record["complete_groups"])

    def test_same_model_in_two_copies_is_one_row_and_protocols_stay_apart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = [{"model_id": "m__abc", "sha256": "abc"}]
            first = write_run(root, "kopya1", summary_rows("m__abc", (0.5,) * 4), model)
            second = write_run(root, "kopya2", summary_rows("m__abc", (0.5,) * 4), model)
            other = write_run(root, "baska", summary_rows("m__abc", (0.9,) * 4, max_queries=20), model, max_queries=20)
            records = pano.dedupe_records(
                pano.read_benchmark_run(first) + pano.read_benchmark_run(second) + pano.read_benchmark_run(other)
            )
            self.assertEqual(2, len(records))
            self.assertEqual({10, 20}, {r["max_queries"] for r in records})

    def test_markdown_lists_the_protocol_with_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            run = write_run(
                Path(temp), "a",
                summary_rows("RAW_BASELINE", (0.6, 0.6, 0.4, 0.4)) + summary_rows("m__abc", (0.8, 0.7, 0.7, 0.6)),
                [{"model_id": "m__abc", "sha256": "abc", "relative_path": "m.h5"}],
            )
            records = pano.dedupe_records(pano.read_benchmark_run(run))
            pano.attach_raw_deltas(records)
            index = {"by_sha": {}, "by_name": {}, "cards": {"abc": {"kisa_ad": "yeni_model"}}}
            pano.decorate(records, index)
            text = pano.render_markdown({"records": records, "gates": [], "flights": [], "runs": [run]})
            self.assertIn("urgup_cevresi 10 sorgu", text)
            self.assertIn("yeni_model", text)
            self.assertIn("+0.200", text)
            self.assertIn("RAW (model yok)", text)


if __name__ == "__main__":
    unittest.main()
