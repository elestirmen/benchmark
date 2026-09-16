"""lr/batch cokme teshisi testleri.

Teshisin tek isi cokmeyi ayirt etmek; yanlis pozitif bir "saglam" sonucu
249 saatlik sweepi bilinen bir cokme uzerine kurdurur.
"""

import sys
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import end_to_end_pipeline as pipeline  # noqa: E402
import lr_batch_probe as probe  # noqa: E402


def gate_report(entries, raw_ratio=0.665):
    results = {"RAW": {"truth_ncc_mean": 0.3056, "false_peak_ratio": raw_ratio}}
    results.update(entries)
    return {"raw_false_peak_ratio": raw_ratio, "results": results}


class ProbeJobTests(unittest.TestCase):
    def test_probe_varies_only_learning_rate_and_batch(self):
        """Baska bir eksen degisirse sonuc yorumlanamaz hale gelir."""
        jobs = probe.build_probe_jobs(6)
        fixed_keys = [
            key
            for key in pipeline.CONFIG_KEYS
            if key not in {"LEARNING_RATE", "BATCH_SIZE", "EPOCHS"}
        ]
        for job in jobs:
            with self.subTest(job=job["name"]):
                for key in fixed_keys:
                    self.assertEqual(
                        pipeline.BASELINE_JOB[key],
                        job[key],
                        f"{key} repro konfigurasyonundan sapmis",
                    )

    def test_probe_inherits_the_fixed_early_stopping_contract(self):
        """Erken durdurma acik kalirsa teshis de kisa kesilir."""
        for job in probe.build_probe_jobs(6):
            with self.subTest(job=job["name"]):
                self.assertEqual(0, job["EARLY_STOPPING_PATIENCE"])
                self.assertEqual(0, job["REDUCE_LR_PATIENCE"])

    def test_probe_includes_the_current_setting_as_a_control(self):
        names = {job["name"] for job in probe.build_probe_jobs(6)}
        self.assertIn("lr1e-3_b2", names)
        control = next(
            job for job in probe.build_probe_jobs(6) if job["name"] == "lr1e-3_b2"
        )
        self.assertEqual(pipeline.BASELINE_JOB["LEARNING_RATE"], control["LEARNING_RATE"])
        self.assertEqual(pipeline.BASELINE_JOB["BATCH_SIZE"], control["BATCH_SIZE"])

    def test_probe_does_not_touch_the_sweep(self):
        before = [dict(job) for job in pipeline.TRAINING_JOBS]
        probe.build_probe_jobs(3)
        self.assertEqual(before, pipeline.TRAINING_JOBS)


class ProbeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.job = probe.build_probe_jobs(6)[0]

    def test_constant_output_is_reported_as_collapsed(self):
        report = gate_report(
            {
                "_ts_model_f48_k4_epoch_00001_helu_.h5": {
                    "truth_ncc_mean": 0.0,
                    "false_peak_ratio": float("inf"),
                },
                "son_model.h5": {
                    "truth_ncc_mean": -0.0,
                    "false_peak_ratio": float("inf"),
                },
            }
        )
        summary = probe.summarize(self.job, report)
        self.assertTrue(summary["collapsed"])
        self.assertEqual(0, summary["healthy_checkpoints"])
        self.assertIsNone(summary["best"])

    def test_a_single_healthy_epoch_prevents_a_collapse_verdict(self):
        report = gate_report(
            {
                "_ts_model_f48_k4_epoch_00001_helu_.h5": {
                    "truth_ncc_mean": 0.21,
                    "false_peak_ratio": 0.48,
                },
                "_ts_model_f48_k4_epoch_00005_helu_.h5": {
                    "truth_ncc_mean": 0.0,
                    "false_peak_ratio": float("inf"),
                },
            }
        )
        summary = probe.summarize(self.job, report)
        self.assertFalse(summary["collapsed"])
        self.assertEqual(1, summary["healthy_checkpoints"])
        self.assertEqual(1, summary["best"]["epoch"])
        self.assertAlmostEqual(0.48, summary["best"]["false_peak_ratio"])

    def test_best_ignores_collapsed_checkpoints_even_with_a_finite_ratio(self):
        """Cokmus bir checkpoint sayisal olarak iyi bir oran uretebilir."""
        report = gate_report(
            {
                "_ts_model_f48_k4_epoch_00002_helu_.h5": {
                    "truth_ncc_mean": 0.005,
                    "false_peak_ratio": 0.02,
                },
                "_ts_model_f48_k4_epoch_00004_helu_.h5": {
                    "truth_ncc_mean": 0.18,
                    "false_peak_ratio": 0.55,
                },
            }
        )
        summary = probe.summarize(self.job, report)
        self.assertEqual(4, summary["best"]["epoch"])
        self.assertAlmostEqual(0.55, summary["best"]["false_peak_ratio"])

    def test_epochs_are_reported_in_order(self):
        report = gate_report(
            {
                f"_ts_model_f48_k4_epoch_{index:05d}_helu_.h5": {
                    "truth_ncc_mean": 0.1,
                    "false_peak_ratio": 0.9,
                }
                for index in (5, 1, 3)
            }
        )
        summary = probe.summarize(self.job, report)
        self.assertEqual([1, 3, 5], [row["epoch"] for row in summary["checkpoints"]])


if __name__ == "__main__":
    unittest.main()
