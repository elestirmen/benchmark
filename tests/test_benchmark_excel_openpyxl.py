import json
import sys
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from build_benchmark_excel_openpyxl import build_workbook  # noqa: E402


class OpenpyxlReportTests(unittest.TestCase):
    def test_minimal_report_has_required_sheets_formulas_and_excel_safe_empty_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "run_config.json").write_text(
                json.dumps({"run_id": "unit_excel", "seed": 42}), encoding="utf-8"
            )
            summary = [
                {
                    "direction": "A__TO__B",
                    "search_mode": "global",
                    "model_id": "RAW_BASELINE",
                    "total_queries": 1,
                    "ok_queries": 1,
                    "rejected_queries": 0,
                    "error_queries": 0,
                    "coverage": 1.0,
                    "mean_error_m": 2.5,
                    "median_error_m": 2.5,
                    "success_30m_queries": 1,
                    "success_30m": 1.0,
                    "success_30m_failure_rate": 0.0,
                    "auc_30m": 0.9,
                    "mean_error_under_30m": 2.5,
                    "median_error_under_30m": 2.5,
                    "success_10m": 1.0,
                }
            ]
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            result = {
                "run_id": "unit_excel",
                "status": "ok",
                "direction": "A__TO__B",
                "search_mode": "global",
                "model_id": "RAW_BASELINE",
                "query_id": "Q00001",
                "error_m": 2.5,
            }
            (run_dir / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

            output = run_dir / "benchmark_results.xlsx"
            report = build_workbook(run_dir, output)
            self.assertEqual(report["zip_integrity"], "ok")
            self.assertEqual(report["formula_error_literals"], [])

            workbook = load_workbook(output, data_only=False)
            self.assertEqual(
                workbook.sheetnames,
                ["Özet", "Model Özeti", "Ham Sonuçlar", "Sorgu Manifesti", "Yapılandırma", "Hatalar"],
            )
            self.assertEqual(workbook["Özet"]["B5"].value, "=COUNTA('Ham Sonuçlar'!A2:A2)")
            self.assertEqual(len(workbook["Özet"]._charts), 1)
            self.assertEqual(workbook["Hatalar"].max_row, 2)
            self.assertEqual(len(workbook["Hatalar"].tables), 1)


if __name__ == "__main__":
    unittest.main()
