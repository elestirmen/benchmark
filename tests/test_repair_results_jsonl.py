import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repair_results_jsonl import repair_results  # noqa: E402


class RepairResultsTests(unittest.TestCase):
    def _row(self, *, query_id: str, error_m: float = 3.0, created: str = "t1"):
        return {
            "direction": "A__TO__B",
            "query_variant": "clean",
            "search_mode": "global",
            "model_id": "M",
            "query_id": query_id,
            "status": "ok",
            "error_m": error_m,
            "predicted_center_easting_m": 1.0,
            "predicted_center_northing_m": 2.0,
            "created_at_utc": created,
            "search_seconds": 1.0,
        }

    def test_copy_first_dedup_keeps_one_scientific_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            rows = [
                self._row(query_id="Q1"),
                {**self._row(query_id="Q1", created="t2"), "search_seconds": 2.0},
                self._row(query_id="Q2"),
            ]
            source = run_dir / "results.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = repair_results(run_dir)
            repaired = [json.loads(line) for line in source.read_text().splitlines()]
            self.assertEqual([row["query_id"] for row in repaired], ["Q1", "Q2"])
            self.assertEqual(report["duplicate_rows_removed"], 1)
            self.assertEqual(report["critical_conflicts"], 0)
            self.assertTrue(Path(report["backup"]).is_file())

    def test_critical_conflict_aborts_without_replacing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            rows = [
                self._row(query_id="Q1", error_m=3.0),
                self._row(query_id="Q1", error_m=30.0),
            ]
            source = run_dir / "results.jsonl"
            original = "".join(json.dumps(row) + "\n" for row in rows)
            source.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "kritik içerik"):
                repair_results(run_dir)
            self.assertEqual(source.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
