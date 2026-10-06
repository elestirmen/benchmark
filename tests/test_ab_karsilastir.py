import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deney_araclari"))
import ab_karsilastir as ab  # noqa: E402


def _write(root, route, model, errs):
    d = Path(root) / f"rota_{route}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"model": f"{model}.h5", "max_frames": 0,
               "kareler": [{"frame": i, "err_m": e} for i, e in enumerate(errs)]}
    (d / f"{model}.json").write_text(json.dumps(payload), encoding="utf-8")


class AbKarsilastirTests(unittest.TestCase):
    def _two_roots(self, a_errs, b_errs, route="6_tezde_ucus_4", model="topluluk_x1_2022"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        a, b = Path(tmp.name) / "a", Path(tmp.name) / "b"
        _write(a, route, model, a_errs)
        _write(b, route, model, b_errs)
        return ab.load(a), ab.load(b)

    def test_clear_improvement_is_accepted(self):
        a, b = self._two_roots([100.0] * 40 + [10.0] * 60, [10.0] * 100)
        _, res = ab.compare(a, b, "topluluk_x1_2022")
        self.assertEqual(ab.decide(res)[0], "KABUL")

    def test_significant_drop_is_rejected(self):
        a, b = self._two_roots([10.0] * 100, [100.0] * 40 + [10.0] * 60)
        _, res = ab.compare(a, b, "topluluk_x1_2022")
        self.assertEqual(ab.decide(res)[0], "RED")

    def test_small_change_is_inconclusive(self):
        a, b = self._two_roots([10.0] * 98 + [100.0] * 2, [10.0] * 97 + [100.0] * 3)
        _, res = ab.compare(a, b, "topluluk_x1_2022")
        self.assertEqual(ab.decide(res)[0], "BELIRSIZ")

    def test_gps_shifted_route_excluded_only_at_25m(self):
        a, b = self._two_roots([100.0] * 50 + [30.0] * 50, [30.0] * 100, route="guzergah_1_donus")
        _, res = ab.compare(a, b, "topluluk_x1_2022")
        self.assertNotIn(("Tum rotalar", 25), res)
        self.assertGreater(res[("Tum rotalar", 70)]["fark"], 0)

    def test_birebir_detects_identity_and_difference(self):
        a, b = self._two_roots([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
        self.assertEqual(ab.birebir(a, b)[0], 0)
        a, b = self._two_roots([10.0, 20.0, 30.0], [10.0, 21.0, 30.0])
        self.assertEqual(ab.birebir(a, b)[0], 1)


if __name__ == "__main__":
    unittest.main()
