"""Hand-calculated counterexamples for interpretation and selection behavior."""
import unittest
from recompute_preference_regret import compute_regrets


class PreferenceRegretExamples(unittest.TestCase):
    def test_oracle_set_regret_is_not_actual_top_one_and_ties_do_not_use_truth(self):
        # In test units A=0, B=1, C=.5. In training units they are 0,.5,.25.
        # Predicted B/C scores tie; lower original index chooses B although C
        # has a better realized outcome. A is in the returned set, giving zero
        # oracle set regret even though the prediction chooses the worst item.
        result = compute_regrets(
            [[0, 0, 0], [10, 10, 10], [5, 5, 5]], [2, 0, 1],
            [.9, .1, .1], {"minimum": [0]*3, "scale": [10]*3},
            {"minimum": [0]*3, "scale": [20]*3}, [1/3]*3,
        )
        self.assertEqual(result["top1_index"], 1)
        self.assertEqual(result["top1_exact_tie_count"], 2)
        self.assertAlmostEqual(result["best_in_returned_set_regret_test_normalized"], 0)
        self.assertAlmostEqual(result["actual_top1_regret_test_normalized"], 1)
        self.assertAlmostEqual(result["actual_top1_regret_training_normalized"], .5)

    def test_top_one_cannot_choose_an_unreturned_lower_predicted_score(self):
        result = compute_regrets(
            [[0]*3, [10]*3, [5]*3], [1, 2], [0, .3, .2],
            {"minimum": [0]*3, "scale": [10]*3},
            {"minimum": [0]*3, "scale": [10]*3}, [1/3]*3,
        )
        self.assertEqual(result["top1_index"], 2)
        self.assertAlmostEqual(result["best_in_returned_set_regret_test_normalized"], .5)
        self.assertAlmostEqual(result["actual_top1_regret_test_normalized"], .5)


if __name__ == "__main__":
    unittest.main()
