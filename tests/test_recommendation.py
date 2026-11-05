import unittest
import numpy as np
from ymap_research.recommendation import pareto_mask, fit_normalization, recommend, recommendation_metrics, hypervolume_3d


class RecommendationTests(unittest.TestCase):
    def test_pareto_keeps_tied_ids_and_drops_dominated(self):
        points = np.array([[1, 2, 1], [2, 1, 1], [1, 2, 1], [2, 2, 2]])
        self.assertEqual(pareto_mask(points).tolist(), [True, True, True, False])

    def test_exact_hypervolume_overlap_and_outside_box(self):
        points = [[.2, .8, .5], [.8, .2, .5], [.9, .9, .9], [2, 0, 0]]
        self.assertAlmostEqual(hypervolume_3d(points, [1, 1, 1]), .14)
        self.assertEqual(hypervolume_3d(np.empty((0, 3))), 0.0)

    def test_selection_uses_means_and_normalized_standard_deviations(self):
        mu = np.array([[1., 100., 10.], [1.1, 110., 11.]])
        std = np.array([[1., 100., 10.], [0., 0., 0.]])
        norm = {'minimum': [0, 0, 0], 'scale': [1, 100, 10]}
        plain = recommend(mu, std, norm, mode='preference')
        risk = recommend(mu, std, norm, risk_lambda=.2, mode='preference')
        self.assertEqual(plain['indices'], [0])
        self.assertEqual(risk['indices'], [1])
        self.assertAlmostEqual(risk['risk_objectives'][0][0], risk['risk_objectives'][0][1])
        self.assertEqual(plain['preference_scores'][0], 1.0)

    def test_measured_coordinates_and_fixed_pool_normalization(self):
        truth = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
        wrong = recommendation_metrics(truth, [2])
        right = recommendation_metrics(truth, [0])
        self.assertEqual(wrong['f1'], 0.0)
        self.assertAlmostEqual(wrong['gd'], np.sqrt(3))
        self.assertAlmostEqual(wrong['preference_regret'], 1.)
        self.assertEqual(right['igd_plus'], 0.)
        self.assertEqual(right['f1'], 1.)
        self.assertEqual(wrong['metric_normalization'], right['metric_normalization'])

    def test_constant_normalization_and_beam_budget_seed(self):
        norm = fit_normalization([[0, 1, 5], [9, 1, 5]])
        self.assertEqual(norm['scale'], [9., 1., 1.])
        mu = np.arange(60.).reshape(20, 3)
        first = recommend(mu, np.zeros_like(mu), norm, mode='beam', budget=7, seed=4)
        second = recommend(mu, np.zeros_like(mu), norm, mode='beam', budget=7, seed=4)
        self.assertEqual(first['visited_indices'], second['visited_indices'])
        self.assertLessEqual(len(first['visited_indices']), 7)
        self.assertEqual(len(set(first['visited_indices'])), len(first['visited_indices']))
        self.assertTrue(set(first['indices']).issubset(first['visited_indices']))
        self.assertTrue(first['trace'])

    def test_latent_neighborhood_and_dimensionless_exploration(self):
        mu = np.ones((8, 3))
        std = np.arange(24.).reshape(8, 3) / 10
        norm = {'minimum': [0, 0, 0], 'scale': [1, 1, 1]}
        result = recommend(mu, std, norm, mode='beam', budget=4, seed=1, embeddings=np.arange(16.).reshape(8, 2), exploration_eta=.5)
        self.assertEqual(result['neighborhood'], 'provided_embeddings')
        self.assertLessEqual(len(result['visited_indices']), 4)
        np.testing.assert_allclose(result['acquisition_scores'], -np.ones(8) + .5 * std.mean(axis=1))

    def test_nonintegral_selections_and_budgets_cannot_silently_change_evidence(self):
        with self.assertRaisesRegex(ValueError, 'integer candidate positions'):
            recommendation_metrics([[1, 2, 3], [2, 3, 4]], [.9])
        with self.assertRaisesRegex(ValueError, 'integer candidate positions'):
            recommendation_metrics([[1, 2, 3]], [True])
        norm = {'minimum': [0, 0, 0], 'scale': [1, 1, 1]}
        with self.assertRaisesRegex(ValueError, 'budget must be an integer'):
            recommend([[1, 2, 3]], normalization=norm, mode='beam', budget=.5)
        self.assertEqual(hypervolume_3d([]), 0.)
        self.assertEqual(recommend([], normalization=norm, mode='beam', budget=0)['indices'], [])

    def test_reject_invalid_indices_or_risk(self):
        with self.assertRaises(ValueError):
            recommendation_metrics([[1, 2, 3]], [0, 0])
        with self.assertRaises(ValueError):
            recommend([[1, 2, 3]], [[-1, 0, 0]], {'minimum': [0]*3, 'scale': [1]*3})


if __name__ == '__main__':
    unittest.main()
