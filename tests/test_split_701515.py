import unittest
import numpy as np

from src.experiments.splitting import NAMES, grouped_split, quotas, stratified_split


class SplitExperimentTests(unittest.TestCase):
    def test_largest_remainder_quotas_preserve_total(self):
        for size in (20, 2959, 5639, 110744):
            q = quotas(size)
            self.assertEqual(int(q.sum()), size)
            self.assertTrue(np.all(np.abs(q - size * np.array([0.7, 0.15, 0.15])) < 1))
        self.assertEqual(quotas(2959).tolist(), [2071, 444, 444])

    def test_writer_groups_stay_together_with_exact_counts(self):
        groups = np.repeat([f"writer_{i}" for i in range(20)], 3)
        result = grouped_split(groups, time_limit=5)
        self.assertEqual([(result == name).sum() for name in NAMES], [42, 9, 9])
        for writer in set(groups):
            self.assertEqual(len(set(result[groups == writer])), 1)

    def test_each_source_appears_in_each_holdout(self):
        groups = np.repeat([f"style_{i}" for i in range(20)], 3)
        categories = np.repeat(["synthetic"] * 10 + ["human"] * 10, 3)
        result = grouped_split(groups, categories=categories, time_limit=5)
        for name in NAMES:
            self.assertEqual(set(categories[result == name]), {"synthetic", "human"})

    def test_single_writer_stratification_keeps_duplicate_glyphs_in_train(self):
        labels = np.repeat(["a", "d", "ê", "t"], 20)
        hashes = np.array([str(i) for i in range(len(labels))])
        hashes[1] = hashes[0]
        result = stratified_split(labels, hashes)
        self.assertEqual([(result == name).sum() for name in NAMES], [56, 12, 12])
        self.assertEqual(result[0], "train")
        self.assertEqual(result[1], "train")
        for name in NAMES:
            self.assertEqual(set(labels[result == name]), set(labels))
        np.testing.assert_array_equal(result, stratified_split(labels, hashes))


if __name__ == "__main__":
    unittest.main()
