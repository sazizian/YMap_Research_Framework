import unittest
import numpy as np
from scipy.stats import wilcoxon
from ymap_research.statistics import paired_wilcoxon, holm_adjust, compare_system_metrics


class StatisticsTests(unittest.TestCase):
    def test_nine_system_exact_lower_bound_and_nine_holm_family(self):
        result = paired_wilcoxon(np.arange(1, 10), np.zeros(9))
        self.assertEqual(result['p_value'], 2 / 512)
        self.assertEqual(result['rank_biserial'], 1.)
        self.assertEqual(holm_adjust([result['p_value']] * 9), [18 / 512] * 9)

    def test_matches_scipy_exact_no_ties(self):
        x = np.array([1, -2, 3, 4, -5, 6, 7, -8, 9])
        expected = wilcoxon(x, method='exact', alternative='two-sided').pvalue
        self.assertAlmostEqual(paired_wilcoxon(x, np.zeros(9))['p_value'], expected)

    def test_ties_zeros_and_all_zero(self):
        result = paired_wilcoxon([1, 1, 0], [0, 0, 0])
        self.assertEqual(result['n_nonzero'], 2)
        self.assertEqual(result['n_zero'], 1)
        self.assertTrue(result['has_ties'])
        self.assertEqual(result['p_value'], .5)
        equal = paired_wilcoxon([1, 1], [1, 1])
        self.assertEqual(equal['p_value'], 1.)
        self.assertEqual(equal['rank_biserial'], 0.)

    def test_tied_signed_ranks_match_independent_brute_force_sign_distribution(self):
        from itertools import product
        differences = np.array([1., 1., -2., 0., -3.])
        result = paired_wilcoxon(differences, np.zeros(len(differences)))
        ranks = np.array([1.5, 1.5, 3., 4.])
        observed = min(3., 7.)
        extrema = [min(float(ranks[np.array(signs, dtype=bool)].sum()), float(ranks[~np.array(signs, dtype=bool)].sum())) for signs in product((0, 1), repeat=4)]
        self.assertEqual(result['p_value'], np.mean(np.asarray(extrema) <= observed))
        self.assertAlmostEqual(result['rank_biserial'], -.4)
        with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
            paired_wilcoxon([1, 2], [0, 0], zero_tolerance=float('nan'))

    def test_statistics_reject_mixed_scenarios_and_accept_missing_metrics(self):
        rows = [{'model': name, 'object_model': str(i), 'gd': None, 'workload_id': 'same'} for name in ['ref', 'other'] for i in range(3)]
        self.assertEqual(compare_system_metrics(rows, 'ref', ['gd']), [])
        rows[0]['workload_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'one workload/environment'):
            compare_system_metrics(rows, 'ref', ['gd'])
        with self.assertRaisesRegex(ValueError, 'must be unique'):
            compare_system_metrics(rows, 'ref', ['gd', 'gd'])

    def test_family_direction_and_pseudoreplication_rejection(self):
        rows = [{'model': name, 'object_model': str(i), 'gd': value} for name, value in [('ref', .1), ('other', .2)] for i in range(9)]
        result = compare_system_metrics(rows, 'ref', ['gd'])[0]
        self.assertEqual(result['rank_biserial'], -1.)
        self.assertEqual(result['rank_biserial_favoring_reference'], 1.)
        self.assertEqual(result['n_pairs'], 9)
        with self.assertRaises(ValueError):
            compare_system_metrics(rows + [rows[0]], 'ref', ['gd'])


if __name__ == '__main__':
    unittest.main()
