"""Kayip fonksiyonu teshisi testleri.

Teshisin tek isi kaybi tek eksen olarak izole etmek. Baska bir eksen kayarsa
"BCE cokmedi" sonucu kaybin degil o eksenin eseri olabilir ve 249 saatlik sweep
yine yanlis tabana kurulur.
"""

import sys
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import end_to_end_pipeline as pipeline  # noqa: E402
import loss_probe as probe  # noqa: E402


class ProbeJobTests(unittest.TestCase):
    def test_probe_varies_only_the_loss_function(self):
        jobs = probe.build_probe_jobs(6)
        fixed_keys = [
            key
            for key in pipeline.CONFIG_KEYS
            if key not in {"LOSS_FUNCTION", "EPOCHS"}
        ]
        for job in jobs:
            with self.subTest(job=job["name"]):
                for key in fixed_keys:
                    self.assertEqual(
                        pipeline.BASELINE_JOB[key],
                        job[key],
                        f"{key} repro konfigurasyonundan sapmis",
                    )

    def test_the_winner_loss_is_probed_first(self):
        """Kosu yarida kesilirse elde kalan satir belirleyici olan olmali."""
        jobs = probe.build_probe_jobs(6)
        self.assertEqual(probe.WINNER_LOSS, jobs[0]["LOSS_FUNCTION"])

    def test_the_current_sweep_setting_is_included_as_a_control(self):
        losses = {job["LOSS_FUNCTION"] for job in probe.build_probe_jobs(6)}
        self.assertIn(pipeline.BASELINE_JOB["LOSS_FUNCTION"], losses)

    def test_the_unweighted_variant_is_probed_next_to_the_weighted_one(self):
        """weighted_hybrid ile hybrid arasindaki tek fark koyu agirliklandirma;
        ikisi birlikte olculmezse sucun agirlikta oldugu gosterilemez."""
        losses = {job["LOSS_FUNCTION"] for job in probe.build_probe_jobs(6)}
        self.assertIn("weighted_hybrid", losses)
        self.assertIn("hybrid", losses)

    def test_probe_inherits_the_fixed_early_stopping_contract(self):
        for job in probe.build_probe_jobs(6):
            with self.subTest(job=job["name"]):
                self.assertEqual(0, job["EARLY_STOPPING_PATIENCE"])
                self.assertEqual(0, job["REDUCE_LR_PATIENCE"])

    def test_losses_are_distinct(self):
        jobs = probe.build_probe_jobs(6)
        losses = [job["LOSS_FUNCTION"] for job in jobs]
        self.assertEqual(len(losses), len(set(losses)))

    def test_job_names_are_distinct(self):
        names = [job["name"] for job in probe.build_probe_jobs(6)]
        self.assertEqual(len(names), len(set(names)))

    def test_epoch_budget_is_applied_to_every_job(self):
        for job in probe.build_probe_jobs(4):
            with self.subTest(job=job["name"]):
                self.assertEqual(4, job["EPOCHS"])

    def test_probe_does_not_touch_the_sweep(self):
        before = [dict(job) for job in pipeline.TRAINING_JOBS]
        probe.build_probe_jobs(3)
        self.assertEqual(before, pipeline.TRAINING_JOBS)

    def test_probe_writes_outside_the_lr_batch_probe_root(self):
        """Iki teshisin ciktisi karisirsa hicbiri yorumlanamaz."""
        import lr_batch_probe

        self.assertNotEqual(lr_batch_probe.PROBE_ROOT, probe.PROBE_ROOT)


class ProbeSummaryTests(unittest.TestCase):
    """summarize() lr_batch_probe ile paylasilir; kayip alani da tasinmali."""

    def test_summary_reports_the_loss_function(self):
        job = probe.build_probe_jobs(6)[0]
        report = {
            "raw_false_peak_ratio": 0.337,
            "results": {
                "RAW": {"truth_ncc_mean": 0.359, "false_peak_ratio": 0.337},
                "_ts_model_f48_k4_epoch_00003_helu_.h5": {
                    "truth_ncc_mean": 0.22,
                    "false_peak_ratio": 0.41,
                },
            },
        }
        summary = probe.harness.summarize(job, report)
        self.assertEqual(probe.WINNER_LOSS, summary["loss_function"])
        self.assertFalse(summary["collapsed"])
        self.assertEqual(3, summary["best"]["epoch"])


if __name__ == "__main__":
    unittest.main()
