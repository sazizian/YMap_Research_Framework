import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ymap_research.common import TARGETS, validate_dataset, write_jsonl
from ymap_research.inference import train, predict_and_recommend


class ProvenanceAndInferenceTests(unittest.TestCase):
    def fixture(self, root):
        records, labels = [], []
        for group in range(3):
            for variant in range(4):
                sid = f"system{group}_schema{variant}"
                records.append({"schema_id": sid, "object_model": f"system{group}", "features": {"tables": group + 1, "width": variant + 2}, "ddl": "SELECT 1;"})
                labels.append({"schema_id": sid, "object_model": f"system{group}", "insert_ms": 1 + group + variant, "query_ms": 2 + group + variant / 2, "storage_mb": 3 + group * 2 + variant,
                               "label_source": "synthetic_demo", "workload_id": "fixture", "environment_id": "no_database", "seed": 42})
        candidates, measurements = root / "candidates.jsonl", root / "measurements.csv"
        write_jsonl(candidates, records)
        pd.DataFrame(labels).to_csv(measurements, index=False)
        return candidates, measurements, records

    def test_research_mode_rejects_synthetic_and_default_is_research(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "rejects"):
                validate_dataset(*paths[:2])
            self.assertEqual(validate_dataset(*paths[:2], "demo")["source_systems"], 3)

    def test_literal_na_identifier_survives_validation_and_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidates, measurements, records = self.fixture(root)
            original_id = records[0]["schema_id"]
            records[0]["schema_id"] = "NA"
            write_jsonl(candidates, records)
            rows = pd.read_csv(measurements)
            rows.loc[rows.schema_id == original_id, "schema_id"] = "NA"
            rows.to_csv(measurements, index=False)
            self.assertEqual(validate_dataset(candidates, measurements, "demo")["measured_unique_schemas"], 12)
            metadata = train(candidates, measurements, root / "model.joblib", {"mode": "demo", "trees": 3, "ensemble_size": 1})
            self.assertIn("NA", metadata["training_schema_ids"])

    def test_invalid_identifiers_cannot_escape_recommendation_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidates, measurements, records = self.fixture(root)
            train(candidates, measurements, root / "model.joblib", {"mode": "demo", "trees": 3, "ensemble_size": 1})
            invalid = [["../outside"], ["/absolute"], ["sub\\outside"], ["."], [".."], [""], ["CON"], ["LPT1.backup"], ["same", "same"], ["Schema", "schema"]]
            for identifiers in invalid:
                with self.subTest(identifiers=identifiers):
                    changed = [dict(records[0], schema_id=sid) for sid in identifiers]
                    write_jsonl(root / "bad.jsonl", changed)
                    with self.assertRaisesRegex(ValueError, "safe|reserved|unique"):
                        predict_and_recommend(root / "bad.jsonl", root / "model.joblib", root / "out", allow_demo=True)
                    self.assertFalse((root / "out").exists())
            self.assertFalse((root / "outside.sql").exists())

    def test_inference_never_needs_measurement_file_and_marks_demo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidates, measurements, records = self.fixture(root)
            train(candidates, measurements, root / "model.joblib", {"mode": "demo", "trees": 5, "ensemble_size": 2})
            measurements.unlink()
            single = root / "unseen.jsonl"
            # No target columns or training IDs are required in a new-model candidate file.
            changed = [dict(row, schema_id=f"unseen_{i}", object_model="unseen") for i, row in enumerate(records[:4])]
            write_jsonl(single, changed)
            with self.assertRaisesRegex(ValueError, "Demo-trained"):
                predict_and_recommend(single, root / "model.joblib", root / "rejected")
            result = predict_and_recommend(single, root / "model.joblib", root / "predictions", allow_demo=True)
            self.assertEqual(result["source_model"], "unseen")
            self.assertEqual(result["model_mode"], "demo")
            predicted = pd.read_csv(root / "predictions/predictions.csv")
            self.assertEqual(len(predicted), 4)
            self.assertTrue(np.isfinite(predicted[[f"pred_{t}" for t in TARGETS]]).all().all())


if __name__ == "__main__":
    unittest.main()
