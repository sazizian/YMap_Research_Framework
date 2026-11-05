"""Small CPU graph/symbolic tests; no pretrained weights or outcome claims."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ymap_research.neural import EncoderConfig, FrozenTextEncoder, NeuralConfig, NeuralMultiViewRegressor, _encoder_identity, _usable_token_limit


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def candidate(index, group="train"):
    table = f"t{index}"
    return {"schema_id": str(index), "object_model": group, "ddl": f"CREATE TABLE {table} (id BIGINT PRIMARY KEY)", "features": {"table_count": 1, "sample_feature": index}, "graph": {"nodes": [{"id": f"table:{table}", "type": "table", "kind": "entity"}, {"id": f"column:{table}.id", "type": "column", "data_type": "BIGINT", "primary_key": True}], "edges": [{"source": f"table:{table}", "target": f"column:{table}.id", "type": "contains"}]}}


class ConfigurationTests(unittest.TestCase):
    def test_explicit_ablation_configuration(self):
        config = NeuralConfig()
        self.assertEqual(config.text_encoders, ())
        with self.assertRaises(ValueError):
            NeuralConfig(use_graph=False, use_symbolic=False)
        with self.assertRaises(ValueError):
            EncoderConfig("fake", "not-a-checkpoint")

    def test_invalid_numeric_configuration_fails_early(self):
        for arguments in ({"learning_rate": float("nan")}, {"weight_decay": float("inf")}, {"hidden_dim": 2.5}, {"seed": -1}, {"seed": 2**32 - 1, "ensemble_size": 2}, {"use_graph": "false"}, {"text_encoders": [{"family": "llama"}]}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                NeuralConfig(**arguments)

    def test_roberta_position_offset_limits_actual_tokens(self):
        configuration = SimpleNamespace(max_position_embeddings=514, pad_token_id=1)
        tokenizer = SimpleNamespace(pad_token_id=1, model_max_length=10**30)
        self.assertEqual(_usable_token_limit(configuration, tokenizer, "graphcodebert"), 512)
        tokenizer.model_max_length = 256
        self.assertEqual(_usable_token_limit(configuration, tokenizer, "graphcodebert"), 256)
        self.assertEqual(_usable_token_limit(configuration, tokenizer, "llama"), 256)

    def test_checkpoint_identity_excludes_diagnostic_order_and_moving_request(self):
        first = {"config": {"family": "llama", "model_name_or_path": "example/model", "max_length": 32, "revision": "main", "batch_size": 1}, "resolved_commit": "a" * 40, "loading_diagnostics": {"unexpected_keys": ["a", "b"]}}
        second = {**first, "config": {**first["config"], "revision": "a" * 40, "batch_size": 8}, "loading_diagnostics": {"unexpected_keys": ["b", "a"]}}
        self.assertEqual(_encoder_identity(first), _encoder_identity(second))
        self.assertNotEqual(_encoder_identity(first), _encoder_identity({**second, "resolved_commit": "b" * 40}))

    def test_malformed_candidate_has_clear_validation_error(self):
        with self.assertRaisesRegex(ValueError, "schema_id"):
            NeuralMultiViewRegressor().fit([None, candidate(1)], np.ones((2, 3)))

    def test_prediction_requires_fit_and_load_requires_trust(self):
        with self.assertRaises(RuntimeError):
            NeuralMultiViewRegressor().predict([candidate(0)])
        with self.assertRaisesRegex(ValueError, "trusted"):
            NeuralMultiViewRegressor.load("does-not-exist")

    def test_group_validation_rejects_leakage_before_optional_dependencies(self):
        model = NeuralMultiViewRegressor()
        with self.assertRaisesRegex(ValueError, "overlap"):
            model.fit([candidate(0), candidate(1)], np.ones((2, 3)), [candidate(2)], np.ones((1, 3)))

    @unittest.skipIf(HAS_TORCH, "tests missing-dependency behavior only")
    def test_missing_torch_does_not_fall_back_to_hashes(self):
        with self.assertRaisesRegex(ImportError, "No hash/fake fallback"):
            NeuralMultiViewRegressor().fit([candidate(0), candidate(1)], np.ones((2, 3)))


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is not installed")
class NeuralCpuTests(unittest.TestCase):
    def test_graph_symbolic_fit_train_scalers_and_tensor_only_roundtrip(self):
        records = [candidate(i) for i in range(5)]
        y = np.asarray([[i + 1, i + 2, i + 10] for i in range(5)], dtype=float)
        config = NeuralConfig(ensemble_size=2, epochs=2, hidden_dim=8, graph_layers=1, batch_size=3, dropout=0.0)
        model = NeuralMultiViewRegressor(config).fit(records, y, [candidate(10, "validation")], [[11, 12, 20]])
        np.testing.assert_allclose(model.target_mean_, y.mean(axis=0))
        np.testing.assert_allclose(model.feature_mean_, [2, 1])
        members = model.predict_members(records)
        self.assertEqual(members.shape, (2, 5, 3))
        self.assertTrue(np.isfinite(members).all())
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = NeuralMultiViewRegressor.load(directory, trusted=True)
            np.testing.assert_allclose(restored.predict_members(records), members, atol=1e-6)
        self.assertEqual(model.predict([]).shape, (0, 3))

    def test_reproducible_symbolic_only_fit(self):
        records = [candidate(i) for i in range(3)]
        y = np.asarray([[i, i + 2, i + 5] for i in range(3)], dtype=float)
        config = NeuralConfig(use_graph=False, ensemble_size=1, epochs=2, hidden_dim=8, dropout=0.0)
        first = NeuralMultiViewRegressor(config).fit(records, y).predict(records)
        second = NeuralMultiViewRegressor(config).fit(records, y).predict(records)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)

    def test_explicit_float32_tensors_ignore_global_float64_default(self):
        import torch
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            records = [candidate(i) for i in range(3)]
            config = NeuralConfig(ensemble_size=1, epochs=1, hidden_dim=8, graph_layers=1, dropout=0.0)
            model = NeuralMultiViewRegressor(config).fit(records, [[i, i + 1, i + 2] for i in range(3)])
            symbolic, _, graphs = model._prepare(records)
            self.assertEqual(symbolic.dtype, torch.float32)
            self.assertEqual(graphs[0][0].dtype, torch.float32)
            self.assertTrue(all(tensor.dtype == torch.int64 for tensor in graphs[0][1:]))
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.members_[0].parameters()))
            self.assertTrue(np.isfinite(model.predict(records)).all())
        finally:
            torch.set_default_dtype(old_dtype)

    def test_large_finite_features_use_float64_moment_accumulation(self):
        records = [candidate(i) for i in range(3)]
        for record, value in zip(records, (1e20, 2e20, 3e20)):
            record["features"]["sample_feature"] = value
        config = NeuralConfig(use_graph=False, ensemble_size=1, epochs=1, hidden_dim=8)
        model = NeuralMultiViewRegressor(config).fit(records, [[i, i + 1, i + 2] for i in range(3)])
        self.assertTrue(np.isfinite(model.feature_scale_).all())
        self.assertGreater(model.feature_scale_[0], 1e19)
        self.assertTrue(np.isfinite(model.predict(records)).all())


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "optional torch/transformers dependencies are not installed")
class LocalCheckpointTests(unittest.TestCase):
    """Execute real model APIs with tiny local random checkpoints, never Hub weights.

    These are software integration tests, not evidence about pretrained semantic
    quality or the submitted manuscript's experimental results.
    """

    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast, RobertaConfig, RobertaModel, T5Config, T5Model
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "create": 3, "table": 4, "id": 5, "bigint": 6, "primary": 7, "key": 8, "(": 9, ")": 10, ";": 11}
        raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", model_max_length=32)
        torch.manual_seed(1234)
        models = {
            "graphcodebert": RobertaModel(RobertaConfig(vocab_size=len(vocabulary), hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, max_position_embeddings=34, pad_token_id=0)),
            "codet5": T5Model(T5Config(vocab_size=len(vocabulary), d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=4, pad_token_id=0, eos_token_id=2, decoder_start_token_id=0)),
            "llama": LlamaModel(LlamaConfig(vocab_size=len(vocabulary), hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)),
        }
        for family, model in models.items():
            directory = cls.root / family
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
        cls.encoders = tuple(EncoderConfig(family, str(cls.root / family), max_length=32, batch_size=2) for family in models)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_local_backbones_pool_cache_and_preserve_batch_results(self):
        ddls = [candidate(0)["ddl"], candidate(12)["ddl"] + "; " + "CREATE TABLE extra (id BIGINT); " * 8]
        for configuration in self.encoders:
            with self.subTest(family=configuration.family):
                encoder = FrozenTextEncoder(configuration, cache_dir=str(self.root / "cache"))
                vectors = encoder.encode(ddls)
                self.assertEqual(vectors.shape, (2, 8))
                self.assertTrue(np.isfinite(vectors).all())
                self.assertGreater(encoder.truncation_count, 0)
                self.assertTrue(all(not parameter.requires_grad for parameter in encoder.model.parameters()))
                with patch.object(encoder.model, "forward", side_effect=AssertionError("cache should avoid forward")):
                    np.testing.assert_array_equal(encoder.encode(ddls), vectors)
                uncached = FrozenTextEncoder(configuration)
                separate = np.vstack([uncached.encode([ddl]) for ddl in ddls])
                np.testing.assert_allclose(vectors, separate, rtol=1e-5, atol=1e-6)

    def test_all_real_views_train_and_roundtrip_together(self):
        records = [candidate(i) for i in range(4)]
        y = np.asarray([[i + 1, i + 2, i + 10] for i in range(4)], dtype=float)
        config = NeuralConfig(text_encoders=self.encoders, ensemble_size=2, epochs=2, hidden_dim=8, graph_layers=1, batch_size=2, dropout=0.0, cache_dir=str(self.root / "fusion-cache"))
        model = NeuralMultiViewRegressor(config).fit(records, y)
        before = model.predict_members(records)
        self.assertEqual(before.shape, (2, 4, 3))
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = NeuralMultiViewRegressor.load(directory, trusted=True)
            np.testing.assert_allclose(restored.predict_members(records), before, rtol=0, atol=1e-6)

    def test_real_checkpoint_rejects_token_limit_above_capacity(self):
        with self.assertRaisesRegex(ValueError, "positional limit"):
            FrozenTextEncoder(EncoderConfig("graphcodebert", str(self.root / "graphcodebert"), max_length=33))

    def test_real_checkpoint_rejects_missing_used_encoder_weights(self):
        from transformers import AutoTokenizer, RobertaModel
        source = self.root / "graphcodebert"
        model = RobertaModel.from_pretrained(source, local_files_only=True)
        state = {name: value for name, value in model.state_dict().items() if name != "embeddings.word_embeddings.weight"}
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, state_dict=state, safe_serialization=True)
            AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "required frozen encoder weights"):
                FrozenTextEncoder(EncoderConfig("graphcodebert", directory, max_length=32))


if __name__ == "__main__":
    unittest.main()
