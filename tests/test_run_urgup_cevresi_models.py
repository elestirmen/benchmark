import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_urgup_cevresi_models as launcher  # noqa: E402


class WideUrgupLauncherTests(unittest.TestCase):
    def test_default_contract_matches_epoch_sweep_style(self):
        args = launcher.parser().parse_args([])
        argv = launcher.core_arguments(args, resume=False)
        self.assertIn("balanced_exact", argv)
        self.assertIn("clean,hard_v1", argv)
        self.assertIn("roi500,roi1000,roi2000,roi4000,roi8000,global", argv)
        self.assertIn("--bidirectional", argv)
        self.assertIn("--cleanup-maps", argv)
        self.assertEqual(argv[argv.index("--max-queries") + 1], "3000")
        self.assertEqual(argv[argv.index("--batch-size") + 1], "16")
        self.assertEqual(argv[argv.index("--model-dir") + 1], str(launcher.MODEL_DIR))

    def test_resume_uses_the_same_output_directory(self):
        args = launcher.parser().parse_args([])
        argv = launcher.core_arguments(args, resume=True)
        self.assertEqual(
            argv[argv.index("--resume-run") + 1], str(args.output_dir.resolve())
        )
        self.assertNotIn("--run-id", argv)

    def test_model_isolation_is_enabled_by_default(self):
        self.assertTrue(launcher.parser().parse_args([]).isolate_models)

    def test_completed_worker_keys_requires_all_expected_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            rows = [
                {
                    "direction": "A__TO__B",
                    "model_id": "MODEL_1",
                    "query_variant": "clean",
                    "search_mode": f"mode_{index}",
                    "total_queries": 3,
                    "ok_queries": 3,
                    "rejected_queries": 0,
                    "error_queries": 0,
                }
                for index in range(2)
            ]
            (output_dir / "summary.json").write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(
                launcher.completed_worker_keys(
                    output_dir, max_queries=3, expected_groups=2
                ),
                {("A__TO__B", "MODEL_1")},
            )

    def test_isolated_launcher_spawns_one_process_per_model_and_direction_then_finalizes(self):
        model = SimpleNamespace(model_id="MODEL_1", path=Path("one.h5"))
        catalog = SimpleNamespace(models=(model,))
        with (
            patch.object(launcher, "validate_inputs", return_value={"resume": True}),
            patch.object(launcher, "core_arguments", return_value=["--resume-run", "out"]),
            patch("geospatial_model_benchmark.build_model_catalog", return_value=catalog),
            patch("run_urgup_cevresi_models.subprocess.run") as run,
        ):
            run.return_value = SimpleNamespace(returncode=0)
            result = launcher.main([])

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 3)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("forward", commands[0])
        self.assertIn("reverse", commands[1])
        self.assertIn("--worker-skip-final-export", commands[0])
        self.assertIn("--worker-finalize-only", commands[2])


if __name__ == "__main__":
    unittest.main()
