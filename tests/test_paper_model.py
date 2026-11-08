"""Protocol and CPU smoke checks for the manuscript neural path."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ymap_research.paper_model import (
    PAPER_VIEWS,
    PaperEnsemble,
    PaperModelConfig,
    _check_protocol,
    _ddl_chunks,
    _graph_arrays,
    _invert_targets,
    _paper_views,
    _standardizer,
    fit_paper_ensemble,
    fit_predict_paper,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


class PaperModelProtocolTests(unittest.TestCase):
    def test_full_model_rejects_smoke_overrides_and_bad_repetition_seeds(self):
        config = PaperModelConfig()
        self.assertEqual(_paper_views(None), PAPER_VIEWS)
        self.assertEqual(_check_protocol(config, [10, 11, 12, 13, 14], None), (10, 11, 12, 13, 14))
        with self.assertRaisesRegex(ValueError, "seeds"):
            _check_protocol(config, [1, 2, 3, 4, 5], None)
        with self.assertRaisesRegex(ValueError, "50 maximum epochs"):
            _check_protocol(PaperModelConfig(epochs=1), range(5), None)
        self.assertEqual(_check_protocol(PaperModelConfig(epochs=1), [2], ("graph",)), (2,))
        self.assertEqual(_paper_views(("descriptors", "ct5")), ("codet5", "symbolic"))

    def test_missing_weights_never_silently_switches_to_other_views(self):
        with self.assertRaisesRegex(FileNotFoundError, "neural.checkpoints.graphcodebert"):
            fit_predict_paper([], [], [], [], [], {}, range(5))

    def test_neural_api_refuses_overlap_of_train_and_validation_system(self):
        same_system = [{"object_model": "same"}]
        with self.assertRaisesRegex(ValueError, "object models must be disjoint"):
            fit_predict_paper(
                same_system, np.ones((1, 3)), same_system, np.ones((1, 3)), [],
                {}, [0], views=("symbolic",),
            )

    def test_chunking_preserves_statements_and_rejects_truncation(self):
        statements = ["CREATE TABLE a (id int);", "CREATE TABLE b (id int);", "ALTER TABLE b ADD FOREIGN KEY (id) REFERENCES a;"]

        def tokens(text):
            return ["BOS", *text.split(), "EOS"]

        ddl = "\n".join(statements)
        self.assertEqual(_ddl_chunks(ddl, tokens, 13), [tokens("\n".join(statements[:2])), tokens(statements[2])])
        with self.assertRaisesRegex(ValueError, "cannot split"):
            _ddl_chunks(ddl, tokens, 6)

    def test_inverting_standardized_logs_preserves_original_units(self):
        measured = np.array([[2.0, 5.0, 10.0], [8.0, 5.0, 40.0]])
        logged = np.log(measured)
        mean, scale = _standardizer(logged)
        self.assertEqual(scale[1], 1)
        np.testing.assert_allclose(_invert_targets((logged - mean) / scale, mean, scale), measured)

    def test_typed_edges_and_unseen_type_encoding_are_distinct(self):
        graph = {
            "nodes": [
                {"id": "t", "type": "table", "column_count": 1},
                {"id": "c", "type": "column", "data_type": "numeric", "primary_key": True,
                 "foreign_key": False, "nullable": False, "unique": True, "discriminator": False},
            ],
            "edges": [
                {"source": "t", "target": "c", "type": "hasColumn"},
                {"source": "c", "target": "t", "type": "pkOf"},
            ],
        }
        features, (has_column, pk_of) = _graph_arrays(graph, {"int": 0}, ("hasColumn", "pkOf"))
        np.testing.assert_array_equal(features[0], [1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(features[1], [0, 1, 0, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(has_column, [[0, 1]])
        np.testing.assert_array_equal(pk_of, [[1, 0]])


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is not installed")
class PaperModelCPUTests(unittest.TestCase):
    def test_real_graph_and_symbolic_training_and_safe_roundtrip(self):
        import torch
        from ymap_research.paper_schema import PAPER_FEATURES

        def candidate(group, i):
            return {
                "object_model": group,
                "schema_id": f"{group}-{i}",
                "paper_features": {name: i + offset for offset, name in enumerate(PAPER_FEATURES)},
                "paper_graph": {
                    "nodes": [
                        {"id": "t", "type": "table", "column_count": 1},
                        {"id": "t.id", "type": "column", "data_type": "bigint", "primary_key": True,
                         "foreign_key": False, "nullable": False, "unique": True, "discriminator": False},
                    ],
                    "edges": [
                        {"source": "t", "target": "t.id", "type": "hasColumn"},
                        {"source": "t.id", "target": "t", "type": "pkOf"},
                    ],
                },
            }

        train_records = [candidate("train", i) for i in (0, 1, 2)]
        validation_records = [candidate("validation", 100)]
        test_records = [candidate("test", 1), candidate("test", 3)]
        train_y = np.array([[1, 2, 4], [2, 4, 6], [4, 6, 8]])
        validation_y = np.array([[3, 7, 9]])
        config = {"neural": {"device": "cpu", "epochs": 2, "batch_size": 2}}
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model = fit_paper_ensemble(
                train_records, train_y, validation_records, validation_y,
                config, [0, 1], views=("graph", "symbolic"),
            )
            predictions = model.predict_members(test_records)
            self.assertEqual(predictions.shape, (2, 2, 3))
            self.assertTrue(np.isfinite(predictions).all() and (predictions > 0).all())
            self.assertFalse(np.allclose(predictions[0], predictions[1]))
            stats = model.metadata["training_fold_statistics"]
            np.testing.assert_allclose(stats["log_target_mean"], np.log(train_y).mean(axis=0))
            self.assertEqual(stats["descriptor_mean"][0], 1.0)
            self.assertFalse(model.metadata["full_paper_configuration"])
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "ensemble"
                model.save(path)
                reloaded = PaperEnsemble.load(path)
                np.testing.assert_allclose(reloaded.predict_members(test_records), predictions, rtol=1e-7)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
