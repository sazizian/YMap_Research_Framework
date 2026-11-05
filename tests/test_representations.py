"""Pure preprocessing checks; these tests never impersonate model evaluation."""
import unittest

import numpy as np

from ymap_research.representations import GRAPH_NODE_FEATURES, canonical_ddl, ddl_tokens, representation_key, schema_graph, symbolic_feature_names, symbolic_matrix


def example_graph():
    return {"nodes": [{"id": "table:t", "type": "table", "kind": "entity"}, {"id": "column:t.id", "type": "column", "data_type": "BIGINT", "primary_key": True, "nullable": False}, {"id": "column:t.parent_id", "type": "column", "data_type": "BIGINT", "nullable": True}], "edges": [{"source": "table:t", "target": "column:t.id", "type": "contains"}, {"source": "table:t", "target": "column:t.parent_id", "type": "contains"}, {"source": "column:t.parent_id", "target": "column:t.id", "type": "foreign_key"}]}


class RepresentationTests(unittest.TestCase):
    def test_canonicalization_preserves_quoted_text_and_removes_nested_comments(self):
        ddl = 'CREATE /* outer /* nested */ end */ TABLE "Order" (x TEXT DEFAULT \'--A\'); -- note\n'
        self.assertEqual(canonical_ddl(ddl), 'create table "Order" ( x text default \'--A\' ) ;')
        self.assertEqual(canonical_ddl("CREATE TABLE a(x INT);"), canonical_ddl("create\n table A ( X int ) ;"))
        self.assertNotEqual(canonical_ddl('CREATE TABLE "A"(x INT)'), canonical_ddl('CREATE TABLE "a"(x INT)'))

    def test_escaped_and_dollar_quotes_are_not_comments(self):
        self.assertIn("$body$ -- literal /* literal */ $body$", ddl_tokens("CREATE FUNCTION f() RETURNS TEXT AS $body$ -- literal /* literal */ $body$ LANGUAGE sql;"))
        self.assertIn("e'a\\'b'", ddl_tokens("SELECT E'a\\'b';"))
        self.assertIn("'it''s'", ddl_tokens("SELECT 'it''s';"))

    def test_invalid_quotes_and_comments_fail(self):
        for ddl in ("", "-- only a comment", "CREATE /* unfinished", "SELECT 'unfinished", 'SELECT "unfinished', "SELECT $tag$unfinished"):
            with self.subTest(ddl=ddl), self.assertRaises(ValueError):
                canonical_ddl(ddl)

    def test_cache_identity_changes_with_checkpoint_and_literal(self):
        first = representation_key("CREATE TABLE a (id INT)", {"checkpoint": "v1", "max_length": 512})
        self.assertEqual(first, representation_key("create\n table A(id int)", {"max_length": 512, "checkpoint": "v1"}))
        self.assertNotEqual(first, representation_key("CREATE TABLE a (id INT)", {"checkpoint": "v2", "max_length": 512}))
        self.assertNotEqual(representation_key("SELECT 'A'", {}), representation_key("SELECT 'a'", {}))

    def test_extensible_operator_tokens_do_not_collide_with_spaced_operators(self):
        self.assertIn("@>", ddl_tokens("SELECT payload @> '{\"a\": 1}'"))
        self.assertNotEqual(canonical_ddl("SELECT a @> b"), canonical_ddl("SELECT a @ > b"))
        self.assertEqual(ddl_tokens("SELECT a +/* note */ b"), ["select", "a", "+", "b"])
        self.assertIn("::", ddl_tokens("SELECT a::int"))

    def test_graph_is_order_invariant_with_actual_reverse_edges(self):
        original = example_graph()
        graph = schema_graph(original)
        reordered = {"nodes": list(reversed(original["nodes"])), "edges": list(reversed(original["edges"])) + [original["edges"][0]]}
        other = schema_graph(reordered)
        self.assertEqual(graph.node_ids, other.node_ids)
        np.testing.assert_array_equal(graph.node_features, other.node_features)
        np.testing.assert_array_equal(graph.edge_sources, other.edge_sources)
        np.testing.assert_array_equal(graph.edge_types, other.edge_types)
        self.assertEqual(graph.node_features.shape, (3, len(GRAPH_NODE_FEATURES)))
        self.assertEqual(graph.edge_types.tolist().count(0), 2)
        self.assertEqual(graph.edge_types.tolist().count(1), 2)
        self.assertEqual(graph.edge_types.tolist().count(2), 1)
        self.assertEqual(graph.edge_types.tolist().count(3), 1)

    def test_graph_changes_when_foreign_key_changes(self):
        with_fk = schema_graph(example_graph())
        without = example_graph()
        without["edges"] = without["edges"][:-1]
        without_fk = schema_graph(without)
        self.assertFalse(np.array_equal(with_fk.node_features, without_fk.node_features))

    def test_sql_types_are_not_classified_by_ambiguous_substrings(self):
        graph = example_graph()
        graph["nodes"][1]["data_type"] = "INTERVAL"
        graph["nodes"][2]["data_type"] = "POINT"
        encoded = schema_graph(graph)
        interval = encoded.node_features[encoded.node_ids.index("column:t.id")]
        point = encoded.node_features[encoded.node_ids.index("column:t.parent_id")]
        self.assertEqual(interval[GRAPH_NODE_FEATURES.index("temporal_type")], 1)
        self.assertEqual(interval[GRAPH_NODE_FEATURES.index("numeric_type")], 0)
        self.assertEqual(point[GRAPH_NODE_FEATURES.index("other_type")], 1)

    def test_invalid_graph_fails_explicitly(self):
        graph = example_graph()
        graph["edges"][0]["target"] = "missing"
        with self.assertRaisesRegex(ValueError, "dangling"):
            schema_graph(graph)
        with self.assertRaises(ValueError):
            schema_graph({"nodes": []})

    def test_graph_flags_and_column_ownership_are_validated(self):
        graph = example_graph()
        graph["nodes"][1]["nullable"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            schema_graph(graph)
        graph = example_graph()
        graph["edges"] = graph["edges"][1:]
        with self.assertRaisesRegex(ValueError, "exactly one table"):
            schema_graph(graph)

    def test_symbolic_vocabulary_is_train_only_and_numerical(self):
        train = [{"features": {"b": 2, "a": 1}}, {"features": {"a": 3}}]
        vocabulary = symbolic_feature_names(train)
        self.assertEqual(vocabulary, ("a", "b"))
        np.testing.assert_array_equal(symbolic_matrix(train, vocabulary), [[1, 2], [3, 0]])
        np.testing.assert_array_equal(symbolic_matrix([{"features": {"a": 4, "new": 99}}], vocabulary), [[4, 0]])
        for value in ("nan", float("nan"), float("inf"), 1e100):
            with self.assertRaises(ValueError):
                symbolic_matrix([{"features": {"a": value}}], vocabulary)


if __name__ == "__main__":
    unittest.main()
