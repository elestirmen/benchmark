import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_DIR.parent
PIPELINE_PATH = BENCHMARK_DIR / "end_to_end_pipeline.py"
TRAINING_PATH = (
    REPO_ROOT / "autoencoder_sat_to_map" / "egitim" / "autoencoder_unified.py"
)


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("end_to_end_pipeline_tested", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EndToEndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline_module()
        cls.training_source = TRAINING_PATH.read_text(encoding="utf-8")

    def test_all_jobs_use_a_supported_output_contract(self):
        """Cikis aktivasyonu tanh veya sigmoid olabilir.

        sigmoid bilerek desteklenir: benchmark'ta en yuksek success_25m alan
        top_modeller/ modellerinin hepsi sigmoid ciktiliydi ve eski tanh-only
        kisiti bilinen-iyi konfigurasyonun uretilmesini imkansiz kiliyordu.
        """
        self.pipeline.validate_pipeline_configuration()
        self.assertEqual(1000, self.pipeline.BENCHMARK_MAX_QUERIES)
        self.assertTrue(self.pipeline.TRAINING_JOBS)
        self.assertTrue(
            all(
                job["OUTPUT_ACTIVATION"] in self.pipeline.SUPPORTED_OUTPUT_ACTIVATIONS
                for job in self.pipeline.TRAINING_JOBS
            )
        )
        # BATCH_SIZE artik sweepin birincil ekseni; sabit olmasi beklenmez.
        # Korunan sey kontrolun varligi ve gecerli bir deger araligi.
        self.assertTrue(
            any(
                job["BATCH_SIZE"] == self.pipeline.DEFAULT_BATCH_SIZE
                for job in self.pipeline.TRAINING_JOBS
            ),
            "batch ekseninin kontrol isi yok",
        )
        self.assertTrue(
            all(job["BATCH_SIZE"] >= 1 for job in self.pipeline.TRAINING_JOBS)
        )

    def test_unsupported_output_activation_is_rejected(self):
        original = self.pipeline.TRAINING_JOBS
        try:
            self.pipeline.TRAINING_JOBS = [
                dict(original[0], name="bad_activation", OUTPUT_ACTIVATION="relu")
            ]
            with self.assertRaises(ValueError):
                self.pipeline.validate_pipeline_configuration()
        finally:
            self.pipeline.TRAINING_JOBS = original

    def test_binary_crossentropy_requires_a_sigmoid_output(self):
        """BCE [0, 1] etiket ister; tanh ciktisiyla kayip NaN'a gider.

        Taban kayip 2026-09-08'de weighted_hybrid'den binary_crossentropy'ye
        cekildi (loss probe'da BCE cokmeyen tek kayipti). O anda tanh ciktili
        iki is (flat_f48_tanh, classic_f32_k3_control) tabani devraldi ve
        etiketleri [-1, 1] oldugu icin sessizce NaN uretecekti -- 33 saat GPU.
        Bu kombinasyon artik konfigurasyon duzeyinde reddedilir.
        """
        original = self.pipeline.TRAINING_JOBS
        try:
            self.pipeline.TRAINING_JOBS = [
                dict(
                    original[0],
                    name="bce_with_tanh",
                    OUTPUT_ACTIVATION="tanh",
                    LOSS_FUNCTION="binary_crossentropy",
                )
            ]
            with self.assertRaises(ValueError):
                self.pipeline.validate_pipeline_configuration()
        finally:
            self.pipeline.TRAINING_JOBS = original

    def test_no_shipped_job_pairs_binary_crossentropy_with_tanh(self):
        """Gonderilen 15 isin hicbiri gecersiz kayip/aktivasyon ciftine sahip olmamali."""
        for job in self.pipeline.TRAINING_JOBS:
            if str(job["LOSS_FUNCTION"]).lower() == "binary_crossentropy":
                self.assertEqual(
                    "sigmoid", str(job["OUTPUT_ACTIVATION"]).lower(), job["name"]
                )

    def test_flat_bottleneck_jobs_must_keep_full_resolution(self):
        """stride 2 uzamsal indirme ekler ve mimarinin anlamini yok eder."""
        original = self.pipeline.TRAINING_JOBS
        try:
            self.pipeline.TRAINING_JOBS = [
                dict(
                    original[0],
                    name="bad_stride",
                    MODEL_TYPE="flat_bottleneck",
                    STRIDES=(2, 2),
                )
            ]
            with self.assertRaises(ValueError):
                self.pipeline.validate_pipeline_configuration()
        finally:
            self.pipeline.TRAINING_JOBS = original

    def test_sweep_starts_from_the_known_winning_configuration(self):
        """Ilk is, benchmark'ta 0.703 alan mimariyi birebir yeniden uretmeli."""
        first = self.pipeline.TRAINING_JOBS[0]
        self.assertEqual("flat_bottleneck", first["MODEL_TYPE"])
        self.assertEqual(48, first["FILTER_COUNT"])
        self.assertEqual((4, 4), first["KERNEL_SIZE"])
        self.assertEqual((1, 1), first["STRIDES"])
        self.assertEqual(3, first["FLAT_DEPTH"])
        self.assertEqual("elu", first["HIDDEN_ACTIVATION"])
        self.assertEqual("sigmoid", first["OUTPUT_ACTIVATION"])

    def test_each_job_differs_from_the_baseline_on_one_axis_only(self):
        """Tek eksende ayrilma, sonucun hangi degiskene bagli oldugunu korur."""
        baseline = self.pipeline.BASELINE_JOB
        keys = [k for k in self.pipeline.CONFIG_KEYS]
        for job in self.pipeline.TRAINING_JOBS[1:]:
            with self.subTest(job=job["name"]):
                changed = [k for k in keys if job[k] != baseline[k]]
                # classic kontrol grubu mimari + aktivasyonu birlikte degistirir
                # cunku eski sweep'in sinifini oldugu gibi temsil etmesi gerekir.
                limit = 4 if job["MODEL_TYPE"] != baseline["MODEL_TYPE"] else 1
                # Taban kayip artik binary_crossentropy ve BCE matematiksel
                # olarak sigmoid'e baglidir (etiket araligi [0, 1]). Bu yuzden
                # aktivasyon eksenini test eden bir is kaybi da degistirmek
                # ZORUNDADIR; bu, gevsetilmis bir kisit degil BCE'nin tanimindan
                # gelen bir konfound. Yalniz bu tek ek eksene izin verilir ve
                # kaybin gercekten aktivasyon yuzunden degistigi dogrulanir.
                if job["OUTPUT_ACTIVATION"] != baseline["OUTPUT_ACTIVATION"]:
                    self.assertNotEqual(
                        "binary_crossentropy",
                        str(job["LOSS_FUNCTION"]).lower(),
                        f"{job['name']} sigmoid disi cikisla BCE kullanamaz",
                    )
                    limit += 1
                self.assertLessEqual(
                    len(changed),
                    limit,
                    f"{job['name']} degisen eksenler: {changed}",
                )

    def test_hidden_activation_experiment_keeps_the_baseline_output(self):
        by_name = {job["name"]: job for job in self.pipeline.TRAINING_JOBS}
        self.assertEqual("relu", by_name["flat_f48_relu"]["HIDDEN_ACTIVATION"])
        self.assertEqual("sigmoid", by_name["flat_f48_relu"]["OUTPUT_ACTIVATION"])
        self.assertEqual("tanh", by_name["flat_f48_tanh"]["OUTPUT_ACTIVATION"])
        self.assertEqual("elu", by_name["flat_f48_tanh"]["HIDDEN_ACTIVATION"])

    def test_every_generated_training_script_is_valid_and_explicit(self):
        for job in self.pipeline.TRAINING_JOBS:
            with self.subTest(job=job["name"]):
                code = self.pipeline.build_training_code(
                    self.training_source,
                    job,
                    Path(r"C:\pipeline_test") / job["name"],
                )
                ast.parse(code)
                self.assertNotIn("ACTIVATION_FUNC", code)
                self.assertIn(
                    f"    HIDDEN_ACTIVATION = {job['HIDDEN_ACTIVATION']!r}",
                    code,
                )
                self.assertIn(
                    f"    OUTPUT_ACTIVATION = {job['OUTPUT_ACTIVATION']!r}",
                    code,
                )
                self.assertIn(
                    f"    MODEL_TYPE = {job['MODEL_TYPE']!r}",
                    code,
                )
                self.assertIn(f"    BATCH_SIZE = {job['BATCH_SIZE']}", code)

    def test_training_length_is_controlled_by_the_pipeline_not_the_script(self):
        """Erken durdurma pipeline'da acikca kapatilmali.

        Egitim scriptinin varsayilani patience=8 / monitor=val_loss /
        restore_best_weights=True idi ve bu anahtar CONFIG_KEYS disinda
        oldugu icin pipeline ona hic dokunmuyordu. Sonucu, "100 epoch =
        522k ornek" butcesinin sessizce gerceklesmemesiydi.
        """
        self.assertIn("EARLY_STOPPING_PATIENCE", self.pipeline.CONFIG_KEYS)
        self.assertIn("REDUCE_LR_PATIENCE", self.pipeline.CONFIG_KEYS)
        for job in self.pipeline.TRAINING_JOBS:
            with self.subTest(job=job["name"]):
                self.assertEqual(0, job["EARLY_STOPPING_PATIENCE"])
                code = self.pipeline.build_training_code(
                    self.training_source, job, Path(r"C:\pipeline_test") / job["name"]
                )
                self.assertIn("    EARLY_STOPPING_PATIENCE = 0", code)
                self.assertIn("    REDUCE_LR_PATIENCE = 0", code)

    def test_every_epoch_is_written_to_disk(self):
        """Ara epochlar kaydedilmezse en ayirt edici nokta geri alinamaz.

        Diske yalniz best_val_loss / best_mae / best_ssim / son_model
        dusuyordu; dordu de piksel rekonstruksiyonuyla secilmis noktalar ve
        bu sweepin kendi olcumune gore iyi rekonstruksiyon eslesmeyi bozuyor.
        """
        code = self.pipeline.build_training_code(
            self.training_source,
            self.pipeline.TRAINING_JOBS[0],
            Path(r"C:\pipeline_test") / "job",
        )
        self.assertIn("    SAVE_CHECKPOINTS = True", code)

    def test_epoch_checkpoints_are_kept_out_of_the_benchmark_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            epoch_names = [
                f"_ts_model_f48_k4_epoch_{index:05d}_helu_.h5" for index in range(1, 8)
            ]
            keep_names = [
                "_ts_best_val_loss_model_f48_k4.h5",
                "_ts_best_mae_model_f48_k4.h5",
                "_ts_best_ssim_model_f48_k4.h5",
                "son_model.h5",
            ]
            for name in epoch_names + keep_names:
                (model_dir / name).write_bytes(b"x")

            moved = self.pipeline.partition_epoch_checkpoints(model_dir)
            self.assertEqual(len(epoch_names), len(moved))

            remaining = {path.name for path in self.pipeline.model_files_in(model_dir)}
            self.assertEqual(set(keep_names), remaining)
            stored = {
                path.name
                for path in self.pipeline.model_files_in(
                    self.pipeline.epoch_checkpoint_dir(model_dir)
                )
            }
            self.assertEqual(set(epoch_names), stored)

    def test_margin_gate_samples_epochs_evenly_and_keeps_the_last(self):
        """Her epochu olcmek is basina ~100 dk CPU eklerdi; ornekleme sinirli."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            epoch_dir = self.pipeline.epoch_checkpoint_dir(model_dir)
            epoch_dir.mkdir(parents=True)
            for index in range(1, 101):
                (epoch_dir / f"_ts_model_f48_k4_epoch_{index:05d}_helu_.h5").write_bytes(b"x")
            (model_dir / "son_model.h5").write_bytes(b"x")

            sampled = self.pipeline.sample_epoch_checkpoints(model_dir, limit=20)
            self.assertLessEqual(len(sampled), 20)
            epochs = [self.pipeline.checkpoint_epoch_number(path) for path in sampled]
            self.assertEqual(sorted(epochs), epochs)
            self.assertEqual(1, epochs[0])
            self.assertEqual(100, epochs[-1], "son epoch her zaman olculmeli")

            candidates = self.pipeline.margin_gate_candidates(model_dir)
            self.assertIn("son_model.h5", {path.name for path in candidates})

    def test_margin_gate_promotes_the_most_discriminative_epoch(self):
        """Kapi artik yalniz raporlamaz; benchmark'a girecek adayi secer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            epoch_dir = self.pipeline.epoch_checkpoint_dir(model_dir)
            epoch_dir.mkdir(parents=True)
            champion = "_ts_model_f48_k4_epoch_00012_helu_.h5"
            (epoch_dir / champion).write_bytes(b"x")
            (epoch_dir / "_ts_model_f48_k4_epoch_00090_helu_.h5").write_bytes(b"x")
            (model_dir / "_ts_best_val_loss_model_f48_k4.h5").write_bytes(b"x")

            report = {
                "results": {
                    "RAW": {"false_peak_ratio": 0.62},
                    champion: {"false_peak_ratio": 0.31},
                    "_ts_model_f48_k4_epoch_00090_helu_.h5": {"false_peak_ratio": 0.88},
                    "_ts_best_val_loss_model_f48_k4.h5": {"false_peak_ratio": 0.74},
                }
            }
            promoted = self.pipeline.promote_margin_gate_champion(model_dir, report)
            self.assertIsNotNone(promoted)
            self.assertEqual(champion, promoted.name)
            self.assertIn(
                champion, {path.name for path in self.pipeline.model_files_in(model_dir)}
            )

    def test_margin_gate_does_not_copy_a_champion_already_benchmarked(self):
        """Sampiyon zaten ust klasordeyse kopyalanmaz ama KIMLIGI dondurulur.

        Eski sozlesme bu durumda None donduruyordu; artik cagiran taraf
        (keep_only_benchmark_champion) sampiyonun hangi dosya oldugunu bilmek
        zorunda, cunku benchmark listesini o tek dosyaya indirir.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            best = "_ts_best_val_loss_model_f48_k4.h5"
            (model_dir / best).write_bytes(b"x")
            report = {"results": {"RAW": {"false_peak_ratio": 0.62}, best: {"false_peak_ratio": 0.31}}}
            promoted = self.pipeline.promote_margin_gate_champion(model_dir, report)
            self.assertEqual(best, promoted.name)
            self.assertEqual(
                [best],
                [path.name for path in self.pipeline.model_files_in(model_dir)],
            )

    def test_gate_summary_best_ratio_excludes_raw(self):
        """"En iyi oran" model oranlarindan gelmeli, RAW barajindan degil.

        2026-09-10'da job 1'in log satiri "en iyi oran=0.45 | RAW baraji=0.45"
        dedi; gercek en iyi model 0.571'di. Sebep min() hesabina RAW'in dahil
        edilmesiydi ve butun modeller RAW'dan kotu oldugunda bu her zaman
        RAW'in kendi degerini yazar -- yani isin kotu oldugunu tam da o anda
        gizler.
        """
        report = {
            "raw_false_peak_ratio": 0.4519,
            "passed": [],
            "results": {
                "RAW": {"false_peak_ratio": 0.4519, "truth_ncc_mean": 0.3501},
                "ep3.h5": {"false_peak_ratio": 0.5713, "truth_ncc_mean": 0.2501},
                "ep8.h5": {"false_peak_ratio": 1.3200, "truth_ncc_mean": 0.0772},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            gate_root = model_dir / "gate"
            with (
                mock.patch.object(self.pipeline, "PIPELINE_BENCHMARK_ROOT", gate_root),
                mock.patch.object(self.pipeline.subprocess, "run"),
                mock.patch.object(
                    self.pipeline, "margin_gate_candidates", return_value=[model_dir / "x.h5"]
                ),
                mock.patch.object(self.pipeline, "read_json", return_value=report),
                mock.patch.object(self.pipeline, "promote_margin_gate_champion", return_value=None),
            ):
                summary = self.pipeline.run_margin_gate(
                    self.pipeline.TRAINING_JOBS[0], model_dir
                )
        self.assertAlmostEqual(0.5713, summary["best_false_peak_ratio"], places=4)
        self.assertAlmostEqual(0.4519, summary["raw_false_peak_ratio"], places=4)
        self.assertFalse(summary["passed"])

    def test_benchmark_runs_only_the_gate_champion(self):
        """Benchmark --model-dir'i yalniz sampiyonu iceren klasor olmali.

        Alt klasore tasima yaklasimi ise YARAMAZ: benchmark katalogu
        --model-dir'i ozyinelemeli tarar (geospatial_model_benchmark.py:563).
        2026-09-10'da job 1 icin katalog "dosya=25 | calistirilacak=20" dedi --
        _epochs/ ve _benchmark_disi/ altindaki her sey de sayilmisti, yani
        20 x ~50 dk = ~16 saat. Bu test o senaryoyu kilitler.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            champion = "_ts_model_f48_k4_epoch_00003_helu_.h5"
            (model_dir / champion).write_bytes(b"x")
            for name in ("_ts_best_val_loss_model_f48_k4.h5", "_ts_son_model.h5"):
                (model_dir / name).write_bytes(b"x")
            epoch_dir = self.pipeline.epoch_checkpoint_dir(model_dir)
            epoch_dir.mkdir(parents=True)
            (epoch_dir / "_ts_model_f48_k4_epoch_00019_helu_.h5").write_bytes(b"x")

            staged = self.pipeline.stage_benchmark_champion(
                model_dir, model_dir / champion
            )

            self.assertEqual(
                self.pipeline.BENCHMARK_CANDIDATE_DIR, staged.name
            )
            # Klasorun OZYINELEMELI icerigi tek dosya olmali.
            self.assertEqual(
                [champion],
                sorted(path.name for path in staged.rglob("*") if path.is_file()),
            )
            self.assertEqual(staged, self.pipeline.benchmark_model_dir(model_dir))
            # Kaynak dosyalar yerinde kalir; kapi bir sonraki kosuda ayni
            # checkpoint kumesini olcsun.
            self.assertIn(
                champion, {path.name for path in self.pipeline.model_files_in(model_dir)}
            )
            self.assertTrue((epoch_dir / "_ts_model_f48_k4_epoch_00019_helu_.h5").is_file())

    def test_benchmark_falls_back_to_the_job_dir_without_a_champion(self):
        """Kapi secim yapamadiysa benchmark eski davranisina dusmeli.

        Aksi halde kapi hatasi gorevi sessizce ADAYSIZ benchmark'a cevirir.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "_ts_son_model.h5").write_bytes(b"x")
            self.assertIsNone(
                self.pipeline.stage_benchmark_champion(model_dir, None)
            )
            self.assertEqual(model_dir, self.pipeline.benchmark_model_dir(model_dir))

    def test_benchmark_command_targets_the_champion_dir(self):
        """Kurulan komutta --model-dir sampiyon klasorunu gostermeli."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            model_dir.mkdir()
            champion = "_ts_model_f48_k4_epoch_00003_helu_.h5"
            (model_dir / champion).write_bytes(b"x")
            self.pipeline.stage_benchmark_champion(model_dir, model_dir / champion)
            benchmark_root = Path(temp_dir) / "outputs"
            with (
                mock.patch.object(self.pipeline, "PIPELINE_BENCHMARK_ROOT", benchmark_root),
                mock.patch.object(self.pipeline.subprocess, "run") as subprocess_run,
                mock.patch.object(
                    self.pipeline,
                    "benchmark_completion_status",
                    return_value=(True, "tamamlandi"),
                ),
            ):
                self.pipeline.run_benchmark(
                    self.pipeline.TRAINING_JOBS[0], model_dir, include_raw=True
                )
            command = subprocess_run.call_args.args[0]
            given = command[command.index("--model-dir") + 1]
            self.assertEqual(
                str(model_dir / self.pipeline.BENCHMARK_CANDIDATE_DIR), given
            )

    def test_sweep_starts_from_the_known_winning_configuration(self):
        """Ilk is, benchmark'ta 0.703 alan mimariyi birebir yeniden uretmeli."""
        first = self.pipeline.TRAINING_JOBS[0]
        self.assertEqual("flat_bottleneck", first["MODEL_TYPE"])
        self.assertEqual(48, first["FILTER_COUNT"])
        self.assertEqual((4, 4), first["KERNEL_SIZE"])
        self.assertEqual((1, 1), first["STRIDES"])
        self.assertEqual(3, first["FLAT_DEPTH"])
        self.assertEqual("elu", first["HIDDEN_ACTIVATION"])
        self.assertEqual("sigmoid", first["OUTPUT_ACTIVATION"])

    def test_each_job_differs_from_the_baseline_on_one_axis_only(self):
        """Tek eksende ayrilma, sonucun hangi degiskene bagli oldugunu korur."""
        baseline = self.pipeline.BASELINE_JOB
        keys = [k for k in self.pipeline.CONFIG_KEYS]
        for job in self.pipeline.TRAINING_JOBS[1:]:
            with self.subTest(job=job["name"]):
                changed = [k for k in keys if job[k] != baseline[k]]
                # classic kontrol grubu mimari + aktivasyonu birlikte degistirir
                # cunku eski sweep'in sinifini oldugu gibi temsil etmesi gerekir.
                limit = 4 if job["MODEL_TYPE"] != baseline["MODEL_TYPE"] else 1
                # Taban kayip artik binary_crossentropy ve BCE matematiksel
                # olarak sigmoid'e baglidir (etiket araligi [0, 1]). Bu yuzden
                # aktivasyon eksenini test eden bir is kaybi da degistirmek
                # ZORUNDADIR; bu, gevsetilmis bir kisit degil BCE'nin tanimindan
                # gelen bir konfound. Yalniz bu tek ek eksene izin verilir ve
                # kaybin gercekten aktivasyon yuzunden degistigi dogrulanir.
                if job["OUTPUT_ACTIVATION"] != baseline["OUTPUT_ACTIVATION"]:
                    self.assertNotEqual(
                        "binary_crossentropy",
                        str(job["LOSS_FUNCTION"]).lower(),
                        f"{job['name']} sigmoid disi cikisla BCE kullanamaz",
                    )
                    limit += 1
                self.assertLessEqual(
                    len(changed),
                    limit,
                    f"{job['name']} degisen eksenler: {changed}",
                )

    def test_hidden_activation_experiment_keeps_the_baseline_output(self):
        by_name = {job["name"]: job for job in self.pipeline.TRAINING_JOBS}
        self.assertEqual("relu", by_name["flat_f48_relu"]["HIDDEN_ACTIVATION"])
        self.assertEqual("sigmoid", by_name["flat_f48_relu"]["OUTPUT_ACTIVATION"])
        self.assertEqual("tanh", by_name["flat_f48_tanh"]["OUTPUT_ACTIVATION"])
        self.assertEqual("elu", by_name["flat_f48_tanh"]["HIDDEN_ACTIVATION"])

    def test_every_generated_training_script_is_valid_and_explicit(self):
        for job in self.pipeline.TRAINING_JOBS:
            with self.subTest(job=job["name"]):
                code = self.pipeline.build_training_code(
                    self.training_source,
                    job,
                    Path(r"C:\pipeline_test") / job["name"],
                )
                ast.parse(code)
                self.assertNotIn("ACTIVATION_FUNC", code)
                self.assertIn(
                    f"    HIDDEN_ACTIVATION = {job['HIDDEN_ACTIVATION']!r}",
                    code,
                )
                self.assertIn(
                    f"    OUTPUT_ACTIVATION = {job['OUTPUT_ACTIVATION']!r}",
                    code,
                )
                self.assertIn(
                    f"    MODEL_TYPE = {job['MODEL_TYPE']!r}",
                    code,
                )
                self.assertIn(f"    BATCH_SIZE = {job['BATCH_SIZE']}", code)

    def test_training_length_is_controlled_by_the_pipeline_not_the_script(self):
        """Erken durdurma pipeline'da acikca kapatilmali.

        Egitim scriptinin varsayilani patience=8 / monitor=val_loss /
        restore_best_weights=True idi ve bu anahtar CONFIG_KEYS disinda
        oldugu icin pipeline ona hic dokunmuyordu. Sonucu, "100 epoch =
        522k ornek" butcesinin sessizce gerceklesmemesiydi.
        """
        self.assertIn("EARLY_STOPPING_PATIENCE", self.pipeline.CONFIG_KEYS)
        self.assertIn("REDUCE_LR_PATIENCE", self.pipeline.CONFIG_KEYS)
        for job in self.pipeline.TRAINING_JOBS:
            with self.subTest(job=job["name"]):
                self.assertEqual(0, job["EARLY_STOPPING_PATIENCE"])
                code = self.pipeline.build_training_code(
                    self.training_source, job, Path(r"C:\pipeline_test") / job["name"]
                )
                self.assertIn("    EARLY_STOPPING_PATIENCE = 0", code)
                self.assertIn("    REDUCE_LR_PATIENCE = 0", code)

    def test_every_epoch_is_written_to_disk(self):
        """Ara epochlar kaydedilmezse en ayirt edici nokta geri alinamaz.

        Diske yalniz best_val_loss / best_mae / best_ssim / son_model
        dusuyordu; dordu de piksel rekonstruksiyonuyla secilmis noktalar ve
        bu sweepin kendi olcumune gore iyi rekonstruksiyon eslesmeyi bozuyor.
        """
        code = self.pipeline.build_training_code(
            self.training_source,
            self.pipeline.TRAINING_JOBS[0],
            Path(r"C:\pipeline_test") / "job",
        )
        self.assertIn("    SAVE_CHECKPOINTS = True", code)

    def test_epoch_checkpoints_are_kept_out_of_the_benchmark_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            epoch_names = [
                f"_ts_model_f48_k4_epoch_{index:05d}_helu_.h5" for index in range(1, 8)
            ]
            keep_names = [
                "_ts_best_val_loss_model_f48_k4.h5",
                "_ts_best_mae_model_f48_k4.h5",
                "_ts_best_ssim_model_f48_k4.h5",
                "son_model.h5",
            ]
            for name in epoch_names + keep_names:
                (model_dir / name).write_bytes(b"x")

            moved = self.pipeline.partition_epoch_checkpoints(model_dir)
            self.assertEqual(len(epoch_names), len(moved))

            remaining = {path.name for path in self.pipeline.model_files_in(model_dir)}
            self.assertEqual(set(keep_names), remaining)
            stored = {
                path.name
                for path in self.pipeline.model_files_in(
                    self.pipeline.epoch_checkpoint_dir(model_dir)
                )
            }
            self.assertEqual(set(epoch_names), stored)

    def test_margin_gate_samples_epochs_evenly_and_keeps_the_last(self):
        """Her epochu olcmek is basina ~100 dk CPU eklerdi; ornekleme sinirli."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            epoch_dir = self.pipeline.epoch_checkpoint_dir(model_dir)
            epoch_dir.mkdir(parents=True)
            for index in range(1, 101):
                (epoch_dir / f"_ts_model_f48_k4_epoch_{index:05d}_helu_.h5").write_bytes(b"x")
            (model_dir / "son_model.h5").write_bytes(b"x")

            sampled = self.pipeline.sample_epoch_checkpoints(model_dir, limit=20)
            self.assertLessEqual(len(sampled), 20)
            epochs = [self.pipeline.checkpoint_epoch_number(path) for path in sampled]
            self.assertEqual(sorted(epochs), epochs)
            self.assertEqual(1, epochs[0])
            self.assertEqual(100, epochs[-1], "son epoch her zaman olculmeli")

            candidates = self.pipeline.margin_gate_candidates(model_dir)
            self.assertIn("son_model.h5", {path.name for path in candidates})

    def test_margin_gate_promotes_the_most_discriminative_epoch(self):
        """Kapi artik yalniz raporlamaz; benchmark'a girecek adayi secer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            epoch_dir = self.pipeline.epoch_checkpoint_dir(model_dir)
            epoch_dir.mkdir(parents=True)
            champion = "_ts_model_f48_k4_epoch_00012_helu_.h5"
            (epoch_dir / champion).write_bytes(b"x")
            (epoch_dir / "_ts_model_f48_k4_epoch_00090_helu_.h5").write_bytes(b"x")
            (model_dir / "_ts_best_val_loss_model_f48_k4.h5").write_bytes(b"x")

            report = {
                "results": {
                    "RAW": {"false_peak_ratio": 0.62},
                    champion: {"false_peak_ratio": 0.31},
                    "_ts_model_f48_k4_epoch_00090_helu_.h5": {"false_peak_ratio": 0.88},
                    "_ts_best_val_loss_model_f48_k4.h5": {"false_peak_ratio": 0.74},
                }
            }
            promoted = self.pipeline.promote_margin_gate_champion(model_dir, report)
            self.assertIsNotNone(promoted)
            self.assertEqual(champion, promoted.name)
            self.assertIn(
                champion, {path.name for path in self.pipeline.model_files_in(model_dir)}
            )

    def test_margin_gate_does_not_copy_a_champion_already_benchmarked(self):
        """Sampiyon zaten ust klasordeyse kopyalanmaz ama KIMLIGI dondurulur.

        Eski sozlesme bu durumda None donduruyordu; artik cagiran taraf
        (keep_only_benchmark_champion) sampiyonun hangi dosya oldugunu bilmek
        zorunda, cunku benchmark listesini o tek dosyaya indirir.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            best = "_ts_best_val_loss_model_f48_k4.h5"
            (model_dir / best).write_bytes(b"x")
            report = {"results": {"RAW": {"false_peak_ratio": 0.62}, best: {"false_peak_ratio": 0.31}}}
            promoted = self.pipeline.promote_margin_gate_champion(model_dir, report)
            self.assertEqual(best, promoted.name)
            self.assertEqual(
                [best],
                [path.name for path in self.pipeline.model_files_in(model_dir)],
            )

    def test_gate_summary_best_ratio_excludes_raw(self):
        """"En iyi oran" model oranlarindan gelmeli, RAW barajindan degil.

        2026-09-10'da job 1'in log satiri "en iyi oran=0.45 | RAW baraji=0.45"
        dedi; gercek en iyi model 0.571'di. Sebep min() hesabina RAW'in dahil
        edilmesiydi ve butun modeller RAW'dan kotu oldugunda bu her zaman
        RAW'in kendi degerini yazar -- yani isin kotu oldugunu tam da o anda
        gizler.
        """
        report = {
            "raw_false_peak_ratio": 0.4519,
            "passed": [],
            "results": {
                "RAW": {"false_peak_ratio": 0.4519, "truth_ncc_mean": 0.3501},
                "ep3.h5": {"false_peak_ratio": 0.5713, "truth_ncc_mean": 0.2501},
                "ep8.h5": {"false_peak_ratio": 1.3200, "truth_ncc_mean": 0.0772},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            gate_root = model_dir / "gate"
            with (
                mock.patch.object(self.pipeline, "PIPELINE_BENCHMARK_ROOT", gate_root),
                mock.patch.object(self.pipeline.subprocess, "run"),
                mock.patch.object(
                    self.pipeline, "margin_gate_candidates", return_value=[model_dir / "x.h5"]
                ),
                mock.patch.object(self.pipeline, "read_json", return_value=report),
                mock.patch.object(self.pipeline, "promote_margin_gate_champion", return_value=None),
            ):
                summary = self.pipeline.run_margin_gate(
                    self.pipeline.TRAINING_JOBS[0], model_dir
                )
        self.assertAlmostEqual(0.5713, summary["best_false_peak_ratio"], places=4)
        self.assertAlmostEqual(0.4519, summary["raw_false_peak_ratio"], places=4)
        self.assertFalse(summary["passed"])

    def test_cost_estimate_scales_with_architecture(self):
        """Tek olculen deger (f48/k4) her ise uygulanirsa f96 ve k7 gizlenir."""
        by_name = {job["name"]: job for job in self.pipeline.TRAINING_JOBS}
        baseline = self.pipeline.estimate_minutes_per_epoch(
            by_name["flat_f48_k4_sigmoid_repro"]
        )
        self.assertAlmostEqual(
            4.0,
            self.pipeline.estimate_minutes_per_epoch(by_name["flat_f96_k4"]) / baseline,
            places=6,
        )
        self.assertAlmostEqual(
            (7 / 4) ** 2,
            self.pipeline.estimate_minutes_per_epoch(by_name["flat_f48_k7"]) / baseline,
            places=6,
        )
        self.assertGreater(
            self.pipeline.estimate_job_hours(by_name["flat_f96_k4"]),
            self.pipeline.estimate_job_hours(by_name["flat_f24_k4"]),
        )

    def test_margin_gate_measures_the_training_region(self):
        """Kapi secim yaptigi icin test seridine bakarak secmemeli."""
        self.assertEqual("train", self.pipeline.MARGIN_GATE_AOI)

    def test_collapsed_output_is_detected_before_the_benchmark(self):
        """2026-09-04 kosusunda flat_f48/f64/f96 sabit cikti uretip coktu.

        Gercek NCC 0.0 iken 1000 sorguluk benchmark calistirmak checkpoint
        basina ~50 dk GPU'yu bilinen bir sifir icin harcar.
        """
        collapsed = {
            "results": {
                "RAW": {"truth_ncc_mean": 0.3056},
                "a.h5": {"truth_ncc_mean": 0.0, "false_peak_ratio": float("inf")},
                "b.h5": {"truth_ncc_mean": -0.0, "false_peak_ratio": float("inf")},
            }
        }
        self.assertTrue(self.pipeline.collapsed_model_output(collapsed))

        healthy = {
            "results": {
                "RAW": {"truth_ncc_mean": 0.3056},
                "a.h5": {"truth_ncc_mean": 0.0916, "false_peak_ratio": 0.98},
            }
        }
        self.assertFalse(self.pipeline.collapsed_model_output(healthy))

        # Olcum yapilamadiysa cokme varsayilmamali; kosu normal surmelidir.
        self.assertFalse(self.pipeline.collapsed_model_output({}))
        self.assertFalse(self.pipeline.collapsed_model_output(None))
        self.assertFalse(
            self.pipeline.collapsed_model_output(
                {"results": {"RAW": {"truth_ncc_mean": 0.3}}}
            )
        )

    def test_partially_collapsed_job_still_runs(self):
        """Bir checkpoint saglamsa is elenmemeli; cokme TUM adaylar icin gecerli."""
        mixed = {
            "results": {
                "a.h5": {"truth_ncc_mean": 0.0},
                "b.h5": {"truth_ncc_mean": 0.12},
            }
        }
        self.assertFalse(self.pipeline.collapsed_model_output(mixed))

    def test_combined_summary_collects_jobs_by_directory_not_job_list(self):
        """Is listesi degisse de tamamlanmis eski gorevler rapora girmeli."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.pipeline.PIPELINE_RUN_ID = "20260905_120000"
            self.pipeline.PIPELINE_MODELS_ROOT = root / "models"
            self.pipeline.PIPELINE_BENCHMARK_ROOT = root / "outputs"
            self.pipeline.PIPELINE_MODELS_ROOT.mkdir(parents=True)
            self.pipeline.PIPELINE_BENCHMARK_ROOT.mkdir(parents=True)

            legacy = "classic_f32_k3_baseline"
            self.assertNotIn(
                legacy, {job["name"] for job in self.pipeline.TRAINING_JOBS}
            )
            output_dir = self.pipeline.PIPELINE_BENCHMARK_ROOT / legacy
            output_dir.mkdir()
            rows = [
                {
                    "model_id": "legacy_model",
                    "direction": "G__TO__B",
                    "query_variant": "clean",
                    "search_mode": "global",
                    "total_queries": 1000,
                    "success_25m": 0.011,
                }
            ]
            (output_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
            (output_dir / "summary.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            (output_dir / "summary_metadata.json").write_text(
                json.dumps(
                    {
                        "expected_queries_per_group": 1000,
                        "incomplete_group_count": 0,
                        "group_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "summary_incomplete.json").write_text("[]", encoding="utf-8")
            (output_dir / "benchmark_results.xlsx").write_bytes(b"xlsx")
            (output_dir / "excel_validation.json").write_text(
                json.dumps({"zip_integrity": "ok", "formula_error_literals": []}),
                encoding="utf-8",
            )

            json_path = self.pipeline.refresh_combined_summary(None, None)
            self.assertIsNotNone(json_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            jobs = {row["pipeline_job"] for row in payload["rows"]}
            self.assertIn(legacy, jobs)

            status = self.pipeline.build_job_status_rows(None)
            legacy_rows = [row for row in status if row["name"] == legacy]
            self.assertEqual(1, len(legacy_rows))
            self.assertIn("guncel is listesinde yok", legacy_rows[0]["state"])

    def test_completed_benchmark_requires_full_query_and_excel_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
            (output_dir / "summary.json").write_text("{}", encoding="utf-8")
            (output_dir / "summary_metadata.json").write_text(
                json.dumps(
                    {
                        "expected_queries_per_group": 1000,
                        "incomplete_group_count": 0,
                        "group_count": 96,
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "summary_incomplete.json").write_text("[]", encoding="utf-8")
            (output_dir / "benchmark_results.xlsx").write_bytes(b"xlsx")
            (output_dir / "excel_validation.json").write_text(
                json.dumps(
                    {
                        "zip_integrity": "ok",
                        "formula_error_literals": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                (True, "tamamlandi"),
                self.pipeline.benchmark_completion_status(output_dir),
            )

            (output_dir / "summary_incomplete.json").write_text(
                '[{"missing": 1}]',
                encoding="utf-8",
            )
            self.assertFalse(self.pipeline.benchmark_completion_status(output_dir)[0])

    def test_default_run_selection_resumes_latest_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            (models_root / "20260829_220232").mkdir()
            (models_root / "20260830_021548").mkdir()
            args = self.pipeline.parse_args([])
            with mock.patch.object(self.pipeline, "MODELS_PIPELINE_DIR", models_root):
                self.assertEqual(
                    ("20260830_021548", "otomatik_resume"),
                    self.pipeline.choose_run_id(args),
                )

    def test_oom_skips_the_job_without_changing_the_batch(self):
        """OOM'da batch DEGISTIRILMEZ; is atlanir.

        BATCH_SIZE bu sweepin eksen degiskeni. Iki onceki surum dogrudan 1'e,
        sonraki yariya iniyordu. 2026-09-10'da bce_b32 batch 32'de OOM aldi ve
        batch 16'ya dustu -- yani bce_b16 ile ayni deneyi "bce_b32" adiyla
        ~4,5 saat yeniden kosacakti: kopya ve YANLIS ETIKETLI bir veri noktasi.
        "Batch 32 bu GPU'ya sigmiyor" bilgisi bir olcumdur; oyle kaydedilir.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            training_dir = temp_root / "training"
            training_dir.mkdir()
            (training_dir / self.pipeline.TRAINING_SCRIPT_NAME).write_text(
                self.training_source,
                encoding="utf-8",
            )
            job = dict(self.pipeline.TRAINING_JOBS[1])
            with (
                mock.patch.object(self.pipeline, "TRAINING_SCRIPT_DIR", str(training_dir)),
                mock.patch.object(self.pipeline, "PIPELINE_MODELS_ROOT", temp_root / "models"),
                mock.patch.object(
                    self.pipeline,
                    "run_training_process",
                    side_effect=[(1, True)],
                ) as runner,
            ):
                with self.assertRaises(self.pipeline.TrainingOutOfMemory) as caught:
                    self.pipeline.modify_and_run_training(job)
            # Tek deneme: geri cekilme yok.
            self.assertEqual(1, runner.call_count)
            self.assertIn(str(job["BATCH_SIZE"]), str(caught.exception))

    def test_non_oom_training_failure_still_raises_plain_runtime_error(self):
        """Sigmama disi egitim hatasi OOM ile karistirilmamali.

        Ayrimi korumak gerekiyor: OOM'da sweep sonraki ise gecer, gercek bir
        egitim hatasinda durup sebebi gosterir.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            training_dir = temp_root / "training"
            training_dir.mkdir()
            (training_dir / self.pipeline.TRAINING_SCRIPT_NAME).write_text(
                self.training_source,
                encoding="utf-8",
            )
            job = dict(self.pipeline.TRAINING_JOBS[1])
            with (
                mock.patch.object(self.pipeline, "TRAINING_SCRIPT_DIR", str(training_dir)),
                mock.patch.object(self.pipeline, "PIPELINE_MODELS_ROOT", temp_root / "models"),
                mock.patch.object(
                    self.pipeline,
                    "run_training_process",
                    side_effect=[(3, False)],
                ),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    self.pipeline.modify_and_run_training(job)
            self.assertNotIsInstance(
                caught.exception, self.pipeline.TrainingOutOfMemory
            )
    def test_shared_raw_baseline_requires_all_24_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            rows = []
            for direction in ("A__TO__B", "B__TO__A"):
                for variant in self.pipeline.BENCHMARK_VARIANTS:
                    for mode in self.pipeline.BENCHMARK_SEARCH_MODES:
                        rows.append(
                            {
                                "direction": direction,
                                "query_variant": variant,
                                "search_mode": mode,
                                "model_id": self.pipeline.RAW_MODEL_ID,
                                "total_queries": 1000,
                            }
                        )
            (output_dir / "summary.json").write_text(
                json.dumps(rows),
                encoding="utf-8",
            )
            self.assertEqual(
                (True, "tamamlandi"),
                self.pipeline.raw_baseline_completion_status(output_dir),
            )
            rows.pop()
            (output_dir / "summary.json").write_text(
                json.dumps(rows),
                encoding="utf-8",
            )
            self.assertFalse(self.pipeline.raw_baseline_completion_status(output_dir)[0])

    def _run_benchmark_capturing_command(self, job_index, *, include_raw):
        """run_benchmark'i mock'lu calistir ve kurulan komutu dondur.

        run_benchmark artik cikis kodundan bagimsiz olarak tamamlanma kanitina
        bakar, bu yuzden testte tamamlanma da mock'lanmalidir.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_root = Path(temp_dir) / "outputs"
            with (
                mock.patch.object(self.pipeline, "PIPELINE_BENCHMARK_ROOT", benchmark_root),
                mock.patch.object(self.pipeline.subprocess, "run") as subprocess_run,
                mock.patch.object(
                    self.pipeline,
                    "benchmark_completion_status",
                    return_value=(True, "tamamlandi"),
                ),
            ):
                self.pipeline.run_benchmark(
                    self.pipeline.TRAINING_JOBS[job_index],
                    Path(temp_dir) / "models",
                    include_raw=include_raw,
                )
            return subprocess_run.call_args.args[0]

    def test_later_model_benchmark_explicitly_disables_raw(self):
        command = self._run_benchmark_capturing_command(1, include_raw=False)
        self.assertIn("--no-include-raw", command)

    def test_first_model_benchmark_keeps_raw_enabled(self):
        command = self._run_benchmark_capturing_command(0, include_raw=True)
        self.assertNotIn("--no-include-raw", command)

    def test_benchmark_command_pins_the_first_inference_batch(self):
        command = self._run_benchmark_capturing_command(0, include_raw=True)
        self.assertIn("--batch-size", command)
        index = command.index("--batch-size")
        self.assertEqual(
            str(self.pipeline.BENCHMARK_BATCH_ATTEMPTS[0]),
            command[index + 1],
        )

    def test_benchmark_retries_with_a_smaller_batch_after_gpu_oom(self):
        """classic_f96_k3'u olduren senaryo: cikarimda OOM, geri cekilme yok."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_root = Path(temp_dir) / "outputs"
            job = self.pipeline.TRAINING_JOBS[0]
            output_dir = benchmark_root / job["name"]
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model_errors.jsonl").write_text(
                json.dumps({"error_type": "ResourceExhaustedError"}),
                encoding="utf-8",
            )
            completions = [(False, "eksik"), (True, "tamamlandi")]
            with (
                mock.patch.object(self.pipeline, "PIPELINE_BENCHMARK_ROOT", benchmark_root),
                mock.patch.object(self.pipeline.subprocess, "run") as subprocess_run,
                mock.patch.object(
                    self.pipeline,
                    "benchmark_completion_status",
                    side_effect=completions,
                ),
            ):
                self.pipeline.run_benchmark(
                    job,
                    Path(temp_dir) / "models",
                    include_raw=True,
                )
            self.assertEqual(2, subprocess_run.call_count)
            batches = []
            for call in subprocess_run.call_args_list:
                command = call.args[0]
                batches.append(command[command.index("--batch-size") + 1])
            self.assertEqual(
                [
                    str(self.pipeline.BENCHMARK_BATCH_ATTEMPTS[0]),
                    str(self.pipeline.BENCHMARK_BATCH_ATTEMPTS[1]),
                ],
                batches,
            )

    def test_partial_legacy_benchmark_keeps_its_resume_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "run_config.json").write_text(
                json.dumps({"include_raw": True}),
                encoding="utf-8",
            )
            self.assertTrue(
                self.pipeline.benchmark_include_raw_setting(
                    output_dir,
                    Path(temp_dir) / "existing_shared_raw",
                )
            )
            (output_dir / "run_config.json").unlink()
            self.assertFalse(
                self.pipeline.benchmark_include_raw_setting(
                    output_dir,
                    Path(temp_dir) / "existing_shared_raw",
                )
            )


if __name__ == "__main__":
    unittest.main()
