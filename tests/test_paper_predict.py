import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ymap_research.common import sha256_file
from ymap_research.paper_model import PAPER_VIEWS
from ymap_research.paper_predict import predict_paper
from ymap_research.paper_schema import PAPER_FEATURES


def _candidate(identifier):
    ddl = "CREATE TABLE person (id BIGINT PRIMARY KEY);"
    return {
        "schema_id": identifier,
        "object_model": "people",
        "paper_ddl": ddl,
        "paper_ddl_sha256": hashlib.sha256(ddl.encode("utf-8")).hexdigest(),
        "paper_graph": {
            "nodes": [
                {"id": "t", "type": "table", "column_count": 1},
                {"id": "c", "type": "column", "data_type": "bigint",
                 "primary_key": True, "foreign_key": False, "nullable": False,
                 "unique": True, "discriminator": False},
            ],
            "edges": [
                {"type": "hasColumn", "source": "t", "target": "c"},
                {"type": "pkOf", "source": "c", "target": "t"},
            ],
        },
        "paper_features": {key: 1 for key in PAPER_FEATURES},
    }


class PaperPredictTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidates = self.root / "candidate.jsonl"
        self.model = self.root / "model"
        self.out = self.root / "out"
        self.ensemble_dir = self.model / "ensemble"
        self.ensemble_dir.mkdir(parents=True)
        (self.ensemble_dir / "manifest.json").write_text("{}", encoding="utf-8")
        for i in range(2):
            (self.ensemble_dir / f"member-{i}.npz").write_bytes(bytes([i]))
        self.fit = {
            "status": "fitted", "calibration_factors": [2.0, None, 0.0],
            "ensemble_manifest_sha256": sha256_file(self.ensemble_dir / "manifest.json"),
            "ensemble_member_sha256": {
                path.name: sha256_file(path) for path in self.ensemble_dir.glob("member-*.npz")
            },
            "validation_system": "validation",
        }
        self._save_fit()
        self.ensemble = Mock(
            views=PAPER_VIEWS, seeds=(0, 1),
            metadata={"model_version": "fixture", "text_encoders": {"llama": {"checkpoint": "test"}}},
        )
        self.ensemble.predict_members.side_effect = lambda records: np.tile(
            np.asarray([[[1.0, 2.0, 5.0]], [[3.0, 6.0, 9.0]]]), (1, len(records), 1),
        )
        loader = patch("ymap_research.paper_predict.PaperEnsemble.load", return_value=self.ensemble)
        self.loader = loader.start()
        self.addCleanup(loader.stop)

    def _save_fit(self):
        (self.model / "fit_manifest.json").write_text(json.dumps(self.fit), encoding="utf-8")

    def _write(self, *records):
        self.candidates.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    def test_scores_prepared_candidates_and_records_origin_and_uncertainty(self):
        self._write(_candidate("z"), _candidate("a"))
        config = {"neural": {"device": "cpu", "cache_dir": "cached"}}
        report = predict_paper(self.candidates, self.model, self.out, config=config, batch_size=1)

        self.loader.assert_called_once_with(self.ensemble_dir, device="cpu", cache_dir="cached")
        self.assertEqual([len(call.args[0]) for call in self.ensemble.predict_members.call_args_list], [1, 1])
        with (self.out / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["schema_id"] for row in rows], ["z", "a"])
        for row in rows:
            self.assertAlmostEqual(float(row["mean_insert_ms"]), 2)
            self.assertAlmostEqual(float(row["sample_std_insert_ms"]), math.sqrt(2))
            self.assertAlmostEqual(float(row["calibrated_std_insert_ms"]), 2 * math.sqrt(2))
            self.assertAlmostEqual(float(row["sample_std_query_ms"]), math.sqrt(8))
            self.assertEqual(row["calibrated_std_query_ms"], "")
            self.assertEqual(float(row["calibrated_std_storage_mb"]), 0)
            self.assertEqual(float(row["member_seed_1_query_ms"]), 6)

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["input"]["sha256"], sha256_file(self.candidates))
        self.assertTrue(report["model"]["calibration_bound_to_member_files"])
        self.assertFalse(report["full_paper_configuration"])
        manifest = json.loads((self.out / "prediction_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["calibration_factors"], [2.0, None, 0.0])
        self.assertEqual(manifest["model"]["text_encoders"], {"llama": {"checkpoint": "test"}})

    def test_duplicate_or_missing_views_never_leaves_a_partial_prediction(self):
        self._write(_candidate("duplicate"), _candidate("duplicate"))
        with self.assertRaisesRegex(ValueError, "line 2: duplicate schema_id"):
            predict_paper(self.candidates, self.model, self.out, batch_size=1)
        self.assertEqual(list(self.out.iterdir()), [])
        self._write(_candidate("valid"), {"schema_id": "raw-only", "object_model": "people", "ddl": "CREATE TABLE x (id INT);"})
        with self.assertRaisesRegex(ValueError, "line 2: text views require.*paper_ddl"):
            predict_paper(self.candidates, self.model, self.out)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_model_binding_and_member_output_validation(self):
        self._write(_candidate("x"))
        (self.ensemble_dir / "member-1.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "members do not match"):
            predict_paper(self.candidates, self.model, self.out)
        self.ensemble.predict_members.assert_not_called()
        (self.ensemble_dir / "member-1.npz").write_bytes(b"\x01")
        self.ensemble.predict_members.side_effect = None
        self.ensemble.predict_members.return_value = np.asarray([[[0.0, 1, 1]], [[1.0, 1, 1]]])
        with self.assertRaisesRegex(ValueError, "finite positive prediction"):
            predict_paper(self.candidates, self.model, self.out)
        self.assertEqual(list(self.out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
