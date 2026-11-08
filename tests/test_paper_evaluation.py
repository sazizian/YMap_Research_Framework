import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ymap_research.evaluation import OBJECTIVES
from ymap_research.paper_evaluation import (
    _paired_family,
    _preference_decisions,
    _realized_regrets,
    candidate_normalization,
    evaluate_paper,
    gaussian_multiplier,
    prediction_metrics,
    preference_weights,
)
from ymap_research.recommendation import normalize_predictions


def _fixture(tmp_path):
    candidates, measured = [], []
    for i, system in enumerate(("one", "two", "three")):
        for j in range(3):
            schema_id = f"{system}-{j}"
            candidates.append({"schema_id": schema_id, "object_model": system,
                               "features": {"tables": j + 1, "fks": i},
                               "ddl": f"CREATE TABLE {system}_{j} (id INT);"})
            for seed, scale in enumerate((1, 2, 99)):
                measured.append({
                    "schema_id": schema_id, "object_model": system, "seed": seed,
                    "workload_id": "w", "environment_id": "e", "label_source": "postgresql_measured",
                    **{objective: (i + j + k + 1) * scale for k, objective in enumerate(OBJECTIVES)},
                })
    c_path, m_path = tmp_path / "candidates.jsonl", tmp_path / "measurements.csv"
    c_path.write_text("".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8")
    pd.DataFrame(measured).to_csv(m_path, index=False)
    return c_path, m_path


class PaperEvaluationTests(unittest.TestCase):
    def test_prediction_range_and_fallback_scale_are_invariant_to_units(self):
        means = np.array([[2, 8, 5], [4, 8, 5]])
        train = np.array([[10, 20, 7], [20, 23, 7]])
        spread = np.array([[1, 2, 3], [2, 4, 6]])
        scales = candidate_normalization(means, train)
        assert scales["scale"] == [2, 3, 7]
        assert scales["scale_source"] == ["prediction_range", "training_range", "training_magnitude"]
        changed = np.array([10, 0.1, 100])
        np.testing.assert_allclose(
            normalize_predictions(means, spread, scales),
            normalize_predictions(means * changed, spread * changed, candidate_normalization(means * changed, train * changed)),
        )


    def test_validation_gaussian_scale_and_degenerate_metrics(self):
        truth = np.array([[3, 5, 7], [5, 5, 7]])
        mean = np.array([[2, 5, 5], [3, 5, 6]])
        spread = np.array([[1, 1, 0], [1, 2, 1]])
        factors = gaussian_multiplier(truth, mean, spread)
        np.testing.assert_allclose(factors, [np.sqrt(2.5), 0.0, np.nan])
        assert np.isclose(prediction_metrics(truth, mean)[0]["nrmse"], np.sqrt(2.5) / 2)
        assert prediction_metrics(truth, mean)[1]["nrmse"] is None
        assert prediction_metrics(truth, mean)[1]["r2"] is None
        assert prediction_metrics(truth, mean)[1]["spearman"] is None


    def test_weighted_regret_uses_selected_candidate_and_feasible_optimum(self):
        truth = np.array([[1, 9, 7], [4, 3, 7], [9, 1, 7]])
        prediction = np.array([[2, 3, 5], [3, 1, 5], [1, 2, 5]])
        normalization = candidate_normalization(prediction, truth)
        scaled, _ = normalize_predictions(prediction, None, normalization)
        weights = preference_weights()
        chosen, _ = _preference_decisions(scaled)
        assert len(weights) == len({tuple(row) for row in weights}) == 66
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        measured_regret, sensitivity = _realized_regrets(truth, chosen, normalization)
        assert min(measured_regret) >= 0 and max(measured_regret) > 0
        # Insert-only chooses predicted row 2, although measured row 0 is optimal.
        only_insert = np.flatnonzero(weights[:, 0] == 1)[0]
        assert chosen[only_insert] == 2
        assert measured_regret[only_insert] == 1
        assert sensitivity[only_insert] == 4
        # The equally weighted attainable optimum is a measured schema, even
        # when it is not the independent per-objective minima.
        balanced = next(i for i, w in enumerate(weights) if np.array_equal(w, [0.5, 0.5, 0]))
        assert chosen[balanced] == 2
        assert np.isclose(measured_regret[balanced], 3 / 16)


    def test_paired_family_requires_all_real_comparators_and_applies_one_holm_family(self):
        systems = [f"s{i}" for i in range(9)]
        rows = [
            {"model": model, "object_model": system, "variant": variant, "kappa": kappa,
             "repeats": 3, "hypervolume": value, "igd_plus": 1 - value, "f1": value}
            for model, variant, kappa, value in (
                ("ymap", "ensemble", 0, 0.8), ("dts", "supplied", None, 0.6),
                ("triorm", "supplied", None, 0.4), ("leant", "supplied", None, 0.2),
            )
            for system in systems
        ]
        df = pd.DataFrame(rows)
        assert _paired_family(df[df.model != "dts"], systems, "paper")["status"] == "not_run"
        result = _paired_family(df, systems, "paper")
        assert result["status"] == "completed"
        assert len(result["tests"]) == 9
        for test in result["tests"]:
            assert test["p_value"] == 2 / 2**9
            assert test["holm_p_value"] == 9 * 2 / 2**9
            assert test["rank_biserial_favoring_reference"] == 1
            lo, hi = test["hodges_lehmann_exact_95_ci"]
            assert lo <= test["hodges_lehmann_ymap_minus_comparator"] <= hi


    def test_structural_smoke_runs_cyclic_loso_and_uses_measured_medians(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)
        candidates, measurements = _fixture(tmp_path)
        out = tmp_path / "out"
        order = ["two", "one", "three"]
        summary = evaluate_paper(candidates, measurements, out, {
            "scope": "smoke", "views": [], "system_order": order,
            "prediction_baselines": ["training_mean", "ridge"],
        })
        assert summary["evaluated_candidates"] == 9
        assert summary["neural_model"] is None
        assert summary["structural_feature_source"] == "declared"
        folds = json.loads((out / "folds.json").read_text())
        assert [fold["validation_system"] for fold in folds] == order[1:] + order[:1]
        assert all(set(fold["training_systems"]) == set(order) - {fold["test_system"], fold["validation_system"]} for fold in folds)
        rows = pd.read_csv(out / "oof_predictions.csv")
        sample = rows[(rows.schema_id == "one-0") & (rows.model == "training_mean")].iloc[0]
        assert sample.true_insert_ms == 2
        assert np.isclose(sample.pred_insert_ms, np.cbrt(4 * 6 * 8))
        assert len((out / "preferences.jsonl").read_text().splitlines()) == 3 * 2 * 66
        assert json.loads((out / "paired_tests.json").read_text())["tests"] == []
        fake = pd.read_csv(measurements)
        fake["label_source"] = "synthetic"
        fake.to_csv(measurements, index=False)
        with self.assertRaisesRegex(ValueError, "documented measured"):
            evaluate_paper(candidates, measurements, out, {"scope": "smoke", "views": []})

    def test_ensemble_passes_only_training_and_validation_labels_to_model(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)
        candidates, measurements = _fixture(tmp_path)
        out = tmp_path / "out"

        def fit(train_records, train_y, val_records, val_y, test_records, config, seeds, *, views):
            train_group, val_group, test_group = (records[0]["object_model"] for records in (train_records, val_records, test_records))
            self.assertEqual(len({train_group, val_group, test_group}), 3)
            self.assertEqual(views, ("graph", "symbolic"))
            self.assertEqual(seeds, (0, 1))
            test_members = np.stack([np.tile([2.0, 3.0, 4.0], (len(test_records), 1)), np.tile([4.0, 5.0, 6.0], (len(test_records), 1))])
            return test_members, np.stack([val_y - 0.25, val_y + 0.75]), {}

        with patch.dict(sys.modules, {"ymap_research.paper_model": SimpleNamespace(fit_predict_paper=fit)}):
            summary = evaluate_paper(candidates, measurements, out, {
                "scope": "ablation", "views": ["graph", "symbolic"], "repetition_seeds": [[0, 1]],
            })
        self.assertEqual(summary["unavailable_risk_folds"], [])
        coverage = pd.read_csv(out / "uncertainty_metrics_per_repetition.csv")
        np.testing.assert_allclose(coverage.gaussian_multiplier, 0.25 / np.sqrt(0.5))
        rows = pd.read_csv(out / "recommendation_metrics_per_repetition.csv")
        self.assertEqual(set(rows[rows.variant == "ensemble"].kappa), {0.0, 0.5, 1.0, 2.0})
        self.assertEqual(set(pd.read_csv(out / "prediction_metrics_per_seed.csv").seed), {0, 1})
