import unittest

import numpy as np

import margin_gate


class MarginGateTests(unittest.TestCase):
    def test_ncc_matrix_matches_pairwise_ncc(self):
        rng = np.random.default_rng(3)
        queries = [rng.normal(size=(12, 12)).astype(np.float32) for _ in range(5)]
        maps = [q + rng.normal(scale=0.5, size=q.shape).astype(np.float32) for q in queries]
        maps[2] = np.full((12, 12), 7.0, dtype=np.float32)  # sifir varyans -> 0
        matrix = margin_gate.ncc_matrix(queries, maps)
        for i, query in enumerate(queries):
            for j, search in enumerate(maps):
                self.assertAlmostEqual(margin_gate.ncc(query, search), matrix[i, j], places=5)

    def test_score_pairs_uses_the_best_wrong_location(self):
        rng = np.random.default_rng(4)
        queries = [rng.normal(size=(16, 16)) for _ in range(4)]
        maps = [q.copy() for q in queries]
        maps[1] = queries[0] * 0.9 + rng.normal(scale=0.1, size=(16, 16))  # 0 icin guclu yanlis tepe
        score = margin_gate.score_pairs(queries, maps)
        truth = [margin_gate.ncc(queries[i], maps[i]) for i in range(4)]
        wrong = [max(margin_gate.ncc(queries[i], maps[j]) for j in range(4) if j != i) for i in range(4)]
        self.assertAlmostEqual(float(np.mean(truth)), score["truth_ncc_mean"], places=6)
        self.assertAlmostEqual(float(np.mean(wrong)), score["wrong_ncc_mean"], places=6)
        self.assertAlmostEqual(np.mean(wrong) / np.mean(truth), score["false_peak_ratio"], places=6)

    def test_default_gate_is_the_legacy_single_group(self):
        self.assertEqual([("g2b_clean", "g2b", "clean")], margin_gate.gate_groups(("clean",), False))
        self.assertEqual(
            ["g2b_clean", "g2b_hard_v1", "b2g_clean", "b2g_hard_v1"],
            [g[0] for g in margin_gate.gate_groups(("clean", "hard_v1"), True)],
        )

    def test_degradation_is_applied_to_the_query_only(self):
        self.assertEqual(("G_hard_v1", "B"), margin_gate.group_chip_keys("g2b", "hard_v1"))
        self.assertEqual(("B_hard_v1", "G"), margin_gate.group_chip_keys("b2g", "hard_v1"))
        self.assertEqual(("B", "G"), margin_gate.group_chip_keys("b2g", "clean"))

    def test_multi_group_score_is_the_mean_and_keeps_the_groups(self):
        a = {"samples": 3, "false_peak_ratio": 0.4, "wins": 2}
        b = {"samples": 3, "false_peak_ratio": 0.6, "wins": 1}
        self.assertEqual(a, margin_gate.combine_groups({"g2b_clean": a}))
        combined = margin_gate.combine_groups({"g2b_clean": a, "b2g_clean": b})
        self.assertAlmostEqual(0.5, combined["false_peak_ratio"])
        self.assertEqual(3, combined["samples"])
        self.assertEqual({"g2b_clean", "b2g_clean"}, set(combined["groups"]))

    def test_raw_groups_score_every_direction(self):
        rng = np.random.default_rng(5)
        google = [rng.integers(0, 255, size=(544, 544, 3), dtype=np.uint8) for _ in range(3)]
        bing = [np.clip(g.astype(int) + rng.integers(-20, 20, size=g.shape), 0, 255).astype(np.uint8) for g in google]
        groups = margin_gate.gate_groups(("clean",), True)
        score = margin_gate.raw_group_scores({"G": google, "B": bing}, groups)
        self.assertEqual({"g2b_clean", "b2g_clean"}, set(score["groups"]))
        self.assertLess(score["false_peak_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
