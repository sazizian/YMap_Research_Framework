import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from ymap_research.evaluation import evaluate, load_dataset, make_regressor


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidates = self.root / 'candidates.jsonl'
        self.measurements = self.root / 'measurements.csv'
        records, rows = [], []
        for group in range(4):
            for item in range(5):
                sid = f'g{group}-{item}'
                records.append({'schema_id': sid, 'object_model': f'g{group}', 'features': {'tables': item + 1, 'joins': group + item}, 'ddl': f'CREATE TABLE table_{group}_{item}(id INT);', 'graph': {}})
                for seed in [1, 2]:
                    rows.append({'schema_id': sid, 'object_model': f'g{group}', 'insert_ms': 10 + group + item + seed / 10, 'query_ms': 20 + 2 * group + 3 * item + seed / 10, 'storage_mb': 30 + group + 2 * item, 'label_source': 'synthetic_demo', 'workload_id': 'test-workload', 'environment_id': 'test-environment', 'seed': seed})
        self.candidates.write_text('\n'.join(json.dumps(row) for row in records))
        pd.DataFrame(rows).to_csv(self.measurements, index=False)
        self.config = {'mode': 'demo', 'models': ['linear'], 'ensemble_base': 'extra_trees', 'model_parameters': {'extra_trees': {'n_estimators': 3}}, 'seed': 5}

    def tearDown(self):
        self.temp.cleanup()

    def test_demo_cannot_enter_research_and_scenarios_not_pooled(self):
        with self.assertRaisesRegex(ValueError, 'measured labels'):
            load_dataset(self.candidates, self.measurements, {'mode': 'research'})
        frame = pd.read_csv(self.measurements)
        frame.loc[0, 'workload_id'] = 'different'
        frame.to_csv(self.measurements, index=False)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            load_dataset(self.candidates, self.measurements, {'mode': 'demo'})

    def test_duplicate_seed_rejected_and_repeats_averaged(self):
        records, frame, features = load_dataset(self.candidates, self.measurements, self.config)
        self.assertEqual(len(records), 20)
        self.assertTrue((frame.measurement_seed_count == 2).all())
        self.assertEqual(features, ['joins', 'tables'])
        self.assertAlmostEqual(frame.iloc[0].insert_ms, 10.15)
        raw = pd.read_csv(self.measurements)
        pd.concat([raw, raw.iloc[[0]]]).to_csv(self.measurements, index=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_dataset(self.candidates, self.measurements, self.config)

    def test_group_separation_saved_predictions_and_shared_ensemble_control(self):
        out = self.root / 'out'
        summary = evaluate(self.candidates, self.measurements, out, self.config)
        self.assertEqual(summary['system_count'], 4)
        audits = json.loads((out / 'fold_audit.json').read_text())
        for audit in audits:
            train, cal, test = set(audit['training_groups']), set(audit['calibration_groups']), {audit['test_group']}
            self.assertFalse(train & cal or train & test or cal & test)
            self.assertEqual(train | cal | test, {'g0', 'g1', 'g2', 'g3'})
        predictions = pd.read_csv(out / 'oof_predictions.csv')
        self.assertEqual(predictions.groupby(['model', 'schema_id']).size().max(), 1)
        self.assertEqual(set(predictions.model), {'linear', 'ensemble5'})
        metrics = pd.read_csv(out / 'per_system_metrics.csv')
        control = metrics[metrics.model == 'ensemble_mean'].sort_values('object_model')
        risk = metrics[metrics.model == 'ensemble_risk'].sort_values('object_model')
        np.testing.assert_array_equal(control.insert_ms_rmse, risk.insert_ms_rmse)
        self.assertEqual(len(metrics), 12)
        runtime = pd.read_csv(out / 'runtime_by_fold.csv')
        self.assertTrue(np.isfinite(runtime.seconds).all())
        self.assertTrue((runtime.seconds >= 0).all())
        ensemble = runtime[runtime.prediction_model == 'ensemble5']
        self.assertEqual(len(ensemble[ensemble.stage == 'fit']), 4)
        self.assertEqual(len(ensemble[ensemble.stage == 'selection']), 8)
        self.assertEqual(set(ensemble[ensemble.stage == 'selection'].variant), {'ensemble_mean', 'ensemble_risk'})
        timing = json.loads((out / 'evaluation_runtime.json').read_text())
        self.assertGreater(timing['total_evaluation_workflow_seconds'], runtime.seconds.sum())
        self.assertIn('schema synthesis/enumeration', timing['excluded_before_evaluation'])

    def test_saved_regret_uses_prediction_choice_and_training_normalization(self):
        out = self.root / 'regret'
        config = {**self.config, 'weights': [.8, .2, 0]}
        summary = evaluate(self.candidates, self.measurements, out, config)
        predictions = pd.read_csv(out / 'oof_predictions.csv')
        measurements = pd.read_csv(out / 'aggregated_measurements.csv').set_index('schema_id')
        audits = {row['test_group']: row for row in json.loads((out / 'fold_audit.json').read_text())}
        objective_names = ['insert_ms', 'query_ms', 'storage_mb']
        for row in json.loads((out / 'recommendations.json').read_text()):
            group = row['object_model']
            selected = row['selection']
            index = selected['preference_index']
            prediction_model = 'ensemble5' if row['model'].startswith('ensemble_') else row['model']
            fold = predictions[(predictions.object_model == group) & (predictions.model == prediction_model)].sort_values('schema_id')
            norm = audits[group]['selection_normalization']
            mean = fold[[f'mean_{name}' for name in objective_names]].to_numpy()
            std = fold[[f'std_{name}' for name in objective_names]].to_numpy()
            weights = np.asarray(selected['weights'])
            score = ((mean - norm['minimum']) / norm['scale'] + selected['risk_lambda'] * std / norm['scale']) @ weights
            expected_index = selected['indices'][np.argmin(score[selected['indices']])]
            self.assertEqual(index, expected_index)
            self.assertEqual(row['preference_schema_id'], fold.iloc[index].schema_id)
            truth = measurements.loc[fold.schema_id, objective_names].to_numpy()
            realized = ((truth - norm['minimum']) / norm['scale']) @ weights
            self.assertAlmostEqual(row['measured_metrics']['preference_regret'], realized[index] - realized.min())
        self.assertIn('historical_results', summary['preference_regret_definition'])

    def test_empty_beam_exports_undefined_decision_regret(self):
        out = self.root / 'empty_beam'
        summary = evaluate(self.candidates, self.measurements, out, {**self.config, 'selection_mode': 'beam', 'beam_budget': 0})
        for row in json.loads((out / 'recommendations.json').read_text()):
            self.assertIsNone(row['preference_schema_id'])
            self.assertIsNone(row['measured_metrics']['preference_regret'])
        for values in summary['macro_system_means'].values():
            self.assertIsNone(values['preference_regret'])
        columns = pd.read_csv(out / 'per_system_metrics.csv').columns
        self.assertNotIn('preference_decision', columns)

    def test_heldout_label_changes_cannot_change_same_fold_predictions(self):
        first_out = self.root / 'first'
        second_out = self.root / 'second'
        evaluate(self.candidates, self.measurements, first_out, self.config)
        frame = pd.read_csv(self.measurements)
        frame.loc[frame.object_model == 'g0', ['insert_ms', 'query_ms', 'storage_mb']] *= 1000
        frame.to_csv(self.measurements, index=False)
        evaluate(self.candidates, self.measurements, second_out, self.config)
        first = pd.read_csv(first_out / 'oof_predictions.csv')
        second = pd.read_csv(second_out / 'oof_predictions.csv')
        columns = ['model', 'schema_id', 'mean_insert_ms', 'mean_query_ms', 'mean_storage_mb']
        pd.testing.assert_frame_equal(first[first.object_model == 'g0'][columns].reset_index(drop=True), second[second.object_model == 'g0'][columns].reset_index(drop=True))
        first_choices = {row['model']: row['preference_schema_id'] for row in json.loads((first_out / 'recommendations.json').read_text()) if row['object_model'] == 'g0'}
        second_choices = {row['model']: row['preference_schema_id'] for row in json.loads((second_out / 'recommendations.json').read_text()) if row['object_model'] == 'g0'}
        self.assertEqual(first_choices, second_choices)

    def test_imputer_and_scaler_fit_only_training_values(self):
        model = make_regressor('ridge')
        model.fit([[1, None], [3, 4]], [[1, 2, 3], [3, 4, 5]])
        np.testing.assert_array_equal(model.named_steps['imputer'].statistics_, [2, 4])
        model.predict([[1000, None]])
        np.testing.assert_array_equal(model.named_steps['scaler'].mean_, [2, 4])

    def test_constant_targets_are_explicit_and_xgboost_never_substituted(self):
        from unittest.mock import patch
        from ymap_research.evaluation import _regression_metrics
        metrics = _regression_metrics(np.ones((3, 3)), np.ones((3, 3)), np.ones((5, 3)))
        self.assertIsNone(metrics['nrmse'])
        self.assertIsNone(metrics['insert_ms_r2'])
        self.assertTrue(metrics['insert_ms_training_constant'])
        self.assertEqual(metrics['insert_ms_rmse'], 0.)
        with patch('ymap_research.evaluation.importlib.util.find_spec', return_value=None):
            with self.assertRaisesRegex(ImportError, 'not substituted'):
                make_regressor('xgboost')
        with self.assertRaisesRegex(ValueError, 'internal validation'):
            make_regressor('mlp', model_config={'early_stopping': True})

    def test_disabled_calibration_and_constant_targets_export_valid_json_and_csv(self):
        frame = pd.read_csv(self.measurements)
        frame[['insert_ms', 'query_ms', 'storage_mb']] = [10., 20., 30.]
        frame.to_csv(self.measurements, index=False)
        config = {**self.config, 'calibration': False}
        out = self.root / 'constant'
        summary = evaluate(self.candidates, self.measurements, out, config)
        json.dumps(summary, allow_nan=False)
        self.assertIsNone(summary['macro_system_means']['linear']['insert_ms_r2'])
        self.assertIsNone(summary['macro_system_means']['linear']['nrmse'])
        calibration = pd.read_csv(out / 'calibration_predictions.csv')
        self.assertTrue(calibration.empty)
        self.assertIn('schema_id', calibration)
        self.assertEqual(json.loads((out / 'summary.json').read_text()), summary)

    def test_literal_na_source_ids_preserved_and_nonstring_ids_rejected(self):
        records = [json.loads(line) for line in self.candidates.read_text().splitlines()]
        old = records[0]['schema_id']
        records[0]['schema_id'] = 'NA'
        self.candidates.write_text('\n'.join(json.dumps(row) for row in records))
        frame = pd.read_csv(self.measurements)
        frame.loc[frame.schema_id == old, 'schema_id'] = 'NA'
        frame.to_csv(self.measurements, index=False)
        loaded, averaged, _ = load_dataset(self.candidates, self.measurements, self.config)
        self.assertIn('NA', averaged.schema_id.tolist())
        self.assertEqual(next(row for row in loaded if row['schema_id'] == 'NA')['object_model'], 'g0')
        records[0]['object_model'] = 1
        self.candidates.write_text('\n'.join(json.dumps(row) for row in records))
        with self.assertRaisesRegex(ValueError, 'needs schema_id'):
            load_dataset(self.candidates, self.measurements, self.config)

    def test_outcome_feature_rejected(self):
        records = [json.loads(line) for line in self.candidates.read_text().splitlines()]
        records[0]['features']['insert_ms'] = 10
        self.candidates.write_text('\n'.join(json.dumps(row) for row in records))
        with self.assertRaisesRegex(ValueError, 'cannot be prediction features'):
            load_dataset(self.candidates, self.measurements, self.config)


if __name__ == '__main__':
    unittest.main()
