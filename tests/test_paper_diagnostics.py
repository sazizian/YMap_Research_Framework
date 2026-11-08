import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from ymap_research.paper_diagnostics import orm_comparison, protocol_transfer


OBJECTIVES = ("insert_ms", "query_ms", "storage_mb")


class PaperDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def _write(self, name, rows):
        path = self.root / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _measured(self, points, workload, *, system="Store"):
        return [
            {"object_model": system, "schema_id": f"{i:03}", "workload_id": workload,
             "label_source": "postgresql_measured", **dict(zip(OBJECTIVES, point))}
            for i, point in enumerate(points)
        ]

    def _predicted(self, measured, means, *, repeat=1, fold=1):
        return [
            {"object_model": row["object_model"], "schema_id": row["schema_id"],
             "workload_id": row["workload_id"], "model": "ymap", "fold": fold, "repeat": repeat,
             **{f"true_{key}": row[key] for key in OBJECTIVES},
             **{f"pred_{key}": val for key, val in zip(OBJECTIVES, point)}}
            for row, point in zip(measured, means)
        ]

    def test_transfer_evaluates_predicted_ids_at_variant_measurements(self):
        increasing = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        reference = self._measured(increasing, "reference")
        variant = self._measured(increasing[::-1], "variant")
        predictions = self._predicted(reference, increasing)
        result = protocol_transfer(
            self._write("reference", reference), self._write("variant", variant[::-1]),
            self._write("predictions", predictions),
        )
        system = result["per_system"][0]
        self.assertEqual(result["matched_candidates"], 3)
        self.assertEqual(system["measured_front_jaccard"], 0.0)
        self.assertAlmostEqual(system["measured_rank_kendall_tau_b"]["insert_ms"], -1.0)
        self.assertAlmostEqual(system["prediction_transfer_spearman"]["query_ms"], -1.0)
        self.assertAlmostEqual(system["hypervolume_ratio"], 1 / 1331)
        json.dumps(result, allow_nan=False)

    def test_explicit_variant_sample_and_independent_repeats(self):
        all_points = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (0, 0, 0)]
        reference = self._measured(all_points, "reference")
        variant = self._measured(all_points[:3][::-1], "variant")
        predicted = [
            *self._predicted(reference, all_points, repeat=1),
            *self._predicted(reference, all_points[:3][::-1] + [all_points[-1]], repeat=2),
        ]
        paths = (
            self._write("reference", reference), self._write("variant", variant),
            self._write("predicted", predicted),
        )
        with self.assertRaisesRegex(ValueError, "candidate identities"):
            protocol_transfer(*paths)
        result = protocol_transfer(*paths, allow_reference_superset=True)
        self.assertEqual(result["prediction_repeat_count"], 2)
        self.assertAlmostEqual(result["per_system"][0]["hypervolume_ratio"], (1 / 1331 + 1) / 2)
        self.assertAlmostEqual(result["per_system"][0]["prediction_transfer_spearman"]["insert_ms"], 0)
        with self.assertRaisesRegex(ValueError, "repeat .*1 missing"):
            protocol_transfer(
                *paths[:2], self._write("missing", predicted[:-2]), allow_reference_superset=True,
            )

    def test_rejects_bad_provenance_duplicate_rows_and_reference_disagreement(self):
        points = [(1, 1, 1), (2, 2, 2)]
        reference = self._measured(points, "reference")
        variant = self._measured(points, "variant")
        predictions = self._predicted(reference, points)
        reference_path, variant_path, prediction_path = (
            self._write("reference", reference), self._write("variant", variant),
            self._write("predictions", predictions),
        )
        with self.assertRaisesRegex(ValueError, "pools different workload_id"):
            protocol_transfer(reference_path, self._write("pooled", [variant[0], reference[1]]), prediction_path)
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            protocol_transfer(reference_path, self._write("duplicate", variant + variant[:1]), prediction_path)
        with self.assertRaisesRegex(ValueError, "actual measured"):
            protocol_transfer(
                reference_path, self._write("demo", [{**row, "label_source": "synthetic_demo"} for row in variant]),
                prediction_path,
            )
        with self.assertRaisesRegex(ValueError, "saved true objectives disagree"):
            protocol_transfer(
                reference_path, variant_path,
                self._write("bad_predictions", [{**predictions[0], "true_insert_ms": 20}, predictions[1]]),
            )
        with self.assertRaisesRegex(ValueError, "must match their recorded environment_id"):
            protocol_transfer(
                self._write("reference_host", [{**row, "environment_id": "host1"} for row in reference]),
                self._write("variant_host", [{**row, "environment_id": "host2"} for row in variant]),
                prediction_path,
            )

    def test_checks_native_held_out_split_audit(self):
        points = [(1, 1, 1), (2, 2, 2)]
        reference = self._measured(points, "reference")
        predicted = [{**row, "scope": "paper", "validation_system": "Bank"}
                     for row in self._predicted(reference, points)]
        reference_path = self._write("reference", reference)
        variant_path = self._write("variant", self._measured(points, "variant"))
        prediction_path = self._write("predicted", predicted)
        fold_path = self.root / "folds.json"
        audit = {"fold": 1, "test_system": "Store", "validation_system": "Bank", "training_systems": ["Library"]}
        fold_path.write_text(json.dumps([audit]))
        result = protocol_transfer(reference_path, variant_path, prediction_path)
        self.assertEqual(result["inputs"]["reference_predictions"]["fold_audit"]["folds"], 1)
        fold_path.write_text(json.dumps([{**audit, "training_systems": ["Store"]}]))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            protocol_transfer(reference_path, variant_path, prediction_path)

    def test_orm_reports_matched_rank_front_and_undefined_ties(self):
        sql = self._measured([(1, 1, 1), (2, 1, 2), (3, 1, 3)], "operations")
        orm = self._measured([(6, 2, 3), (4, 2, 2), (2, 2, 1)], "operations")
        result = orm_comparison(
            self._write("sql", [{**row, "execution_layer": "sql"} for row in sql]),
            self._write("hibernate", [{**row, "execution_layer": "hibernate"} for row in orm[::-1]]),
        )
        summary = result["median_over_systems"]
        self.assertEqual(result["matched_candidates"], 3)
        self.assertEqual(summary["latency_rank_spearman"]["insert_ms"]["value"], -1.0)
        self.assertEqual(summary["latency_rank_spearman"]["query_ms"], {"value": None, "defined": 0})
        self.assertEqual(summary["measured_front_jaccard"]["value"], 0.0)
        self.assertEqual(summary["median_candidate_latency_ratio"]["query_ms"]["value"], 2.0)
        json.dumps(result, allow_nan=False)

    def test_orm_rejects_misaligned_system_and_same_input(self):
        sql = self._write("sql", self._measured([(1, 1, 1)], "operations"))
        orm = self._write("hibernate", self._measured([(2, 2, 1)], "operations", system="Bank"))
        with self.assertRaisesRegex(ValueError, "candidate identities"):
            orm_comparison(sql, orm)
        with self.assertRaisesRegex(ValueError, "separate"):
            orm_comparison(sql, sql)
