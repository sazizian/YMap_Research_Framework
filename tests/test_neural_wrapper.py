"""Deployment-wrapper filename and fixed-target-order regression checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


WRAPPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "neural_train_recommend.py"
SPEC = importlib.util.spec_from_file_location("ymap_neural_deployment_wrapper", WRAPPER_PATH)
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class NeuralWrapperTests(unittest.TestCase):
    def test_recommend_rejects_unsafe_or_duplicate_ids_before_model_loading(self):
        examples = [["../escaped"], ["/absolute"], ["sub\\escaped"], ["."], [".."], [""], ["CON"], ["com1.backup"], ["same", "same"], ["Schema", "schema"]]
        for identifiers in examples:
            with self.subTest(identifiers=identifiers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidates = root / "candidates.jsonl"
                candidates.write_text("\n".join(json.dumps({"schema_id": sid, "object_model": "Example", "ddl": "CREATE TABLE t (id INT)"}) for sid in identifiers), encoding="utf-8")
                argv = [str(WRAPPER_PATH), "recommend", "--candidates", str(candidates), "--model", str(root / "model"), "--out", str(root / "output")]
                with patch.object(WRAPPER.sys, "argv", argv), patch.object(WRAPPER.NeuralMultiViewRegressor, "load") as load:
                    with self.assertRaises(ValueError):
                        WRAPPER.main()
                    load.assert_not_called()
                self.assertFalse((root / "output").exists())
                self.assertFalse((root / "escaped.sql").exists())

    def test_target_names_cannot_relabel_measurement_columns(self):
        WRAPPER.validate_target_order(["insert_ms", "query_ms", "storage_mb"])
        with self.assertRaisesRegex(ValueError, "measurement-column order"):
            WRAPPER.validate_target_order(["query_ms", "insert_ms", "storage_mb"])

    def test_safe_ids_write_only_selected_ddl_inside_output_directory(self):
        # Stubs isolate wrapper I/O from model correctness, which is tested using
        # actual Torch/HF models in test_neural.py. No fake framework fallback.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"schema_id": sid, "object_model": "Example", "ddl": f'CREATE TABLE "{sid}" (id INT);'} for sid in ("schema-01", "schema_02.v1")]
            candidates = root / "candidates.jsonl"
            candidates.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "deployment_provenance.json").write_text(json.dumps({"normalization": {"minimum": [0, 0, 0], "scale": [1, 1, 1]}}), encoding="utf-8")
            model = SimpleNamespace(config=SimpleNamespace(target_names=("insert_ms", "query_ms", "storage_mb")), predict_members=lambda _: np.asarray([[[1., 2., 3.], [2., 3., 4.]]]))
            out = root / "output"
            argv = [str(WRAPPER_PATH), "recommend", "--candidates", str(candidates), "--model", str(model_dir), "--out", str(out)]
            with patch.object(WRAPPER.sys, "argv", argv), patch.object(WRAPPER.NeuralMultiViewRegressor, "load", return_value=model):
                WRAPPER.main()
            self.assertEqual((out / "schema-01.sql").read_text(encoding="utf-8"), records[0]["ddl"])
            self.assertFalse((out / "schema_02.v1.sql").exists())
            self.assertEqual({path.name for path in out.iterdir()}, {"schema-01.sql", "selection.json", "neural_predictions.csv"})


if __name__ == "__main__":
    unittest.main()
