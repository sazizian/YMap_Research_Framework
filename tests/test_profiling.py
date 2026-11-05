import csv
import importlib.util
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from ymap_research.profiling import ProfilingError, build_load_plan, build_logical_fixture, build_query_plan, expected_query_results, profile_candidates, validate_logical_fixture, validate_postgresql_syntax
from ymap_research.synthesis import generate_candidates, quote


def model_fixture():
    return {"name": "ProfileFixture", "classes": [{"name": "Person", "attributes": [{"name": "name", "type": "string"}]}, {"name": "Student", "extends": "Person", "attributes": [{"name": "score", "type": "float"}]}, {"name": "DoctoralStudent", "extends": "Student", "attributes": [{"name": "topic", "type": "string"}]}, {"name": "Course", "attributes": [{"name": "title", "type": "string"}]}], "associations": [{"name": "Teaching", "source": "Student", "target": "Course", "multiplicity": "one-to-many"}]}


class ProfilingTests(unittest.TestCase):
    def test_logical_fixture_deterministic_and_mapping_independent(self):
        fixture = build_logical_fixture(model_fixture(), 3, 7)
        self.assertEqual(fixture, build_logical_fixture(model_fixture(), 3, 7))
        self.assertNotEqual(fixture["realization_id"], build_logical_fixture(model_fixture(), 3, 8)["realization_id"])
        self.assertEqual(len(fixture["objects"]), 12)
        candidates = list(generate_candidates(model_fixture()))
        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            plan = build_load_plan(candidate, fixture)
            self.assertTrue(all(len(row) == len(t["columns"]) for t in plan.values() for row in t["rows"]))
            self.assertEqual(len(build_query_plan(candidate, 5, fixture)), 9)

    def test_joined_loader_emits_all_ancestor_rows_for_grandchild(self):
        model = model_fixture()
        fixture = build_logical_fixture(model, 1, 1)
        candidate = next(c for c in generate_candidates(model) if c["decisions"]["inheritance:Person"] == "joined" and c["decisions"]["association:Teaching"] == "direct_fk")
        plan = build_load_plan(candidate, fixture)
        obj = next(o for o in fixture["objects"] if o["class"] == "DoctoralStudent")
        for table in ("c_Person", "c_Student", "c_DoctoralStudent"):
            ids = [row[plan[table]["columns"].index("id")] for row in plan[table]["rows"]]
            self.assertIn(obj["id"], ids)

    def test_required_one_to_one_infeasible_fixture_rejected(self):
        model = {"name": "ImpossibleCounts", "classes": [{"name": "A"}, {"name": "B"}], "associations": [{"name": "AB", "source": "A", "target": "B", "multiplicity": "one-to-one"}]}
        with self.assertRaisesRegex(ProfilingError, "more target"):
            build_logical_fixture(model, {"A": 1, "B": 2})

    def test_generation_requires_explicit_fixture_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProfilingError, "fixture_mode"):
                profile_candidates([], {}, Path(tmp))

    def test_missing_database_fails_without_creating_labels(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProfilingError, "unset"):
                profile_candidates(generate_candidates(model_fixture()), {"fixture_mode": "deterministic_generated", "dsn_env": "MISSING_YMAP_DSN"}, Path(tmp))
            self.assertFalse((Path(tmp) / "measurements.csv").exists())

    def test_duplicate_object_identity_and_wrong_subtype_link_rejected(self):
        model = model_fixture()
        fixture = build_logical_fixture(model, 1)
        fixture["objects"][1]["id"] = fixture["objects"][0]["id"]
        with self.assertRaisesRegex(ProfilingError, "globally unique"):
            validate_logical_fixture(model, fixture)
        fixture = build_logical_fixture(model, 1)
        person = next(o for o in fixture["objects"] if o["class"] == "Person")
        fixture["links"]["Teaching"][0][0] = person["id"]
        with self.assertRaisesRegex(ProfilingError, "endpoint type"):
            validate_logical_fixture(model, fixture)

    def test_logical_oracle_is_mapping_independent_and_checks_ordered_limits(self):
        model = model_fixture()
        fixture = build_logical_fixture(model, 3, 7)
        candidates = list(generate_candidates(model))
        signatures = [expected_query_results(candidate, fixture, 2) for candidate in candidates]
        self.assertTrue(all(signature == signatures[0] for signature in signatures))
        self.assertEqual(signatures[0]["scan:Person"]["rows"], 2)
        self.assertEqual(signatures[0]["association:Teaching"]["rows"], 2)

    @unittest.skipUnless(importlib.util.find_spec("pglast"), "Install optional pglast for static PostgreSQL/PLpgSQL parsing.")
    def test_static_postgresql_parse_reserved_and_long_identifiers(self):
        model = {"name": "QuotedNames", "classes": [{"name": "Order", "attributes": [{"name": "select", "type": "decimal"}]}, {"name": "MixedCase" + "x" * 35, "attributes": [{"name": "LongAttribute" + "y" * 30, "type": "datetime"}]}], "associations": [{"name": "Association" + "Z" * 35, "source": "Order", "target": "MixedCase" + "x" * 35, "multiplicity": "one-to-one"}]}
        fixture = build_logical_fixture(model, 2)
        for candidate in generate_candidates(model):
            report = validate_postgresql_syntax(candidate, fixture)
            self.assertFalse(report["database_execution"])
            self.assertGreater(report["ddl_statements"], 0)

    @unittest.skipUnless(os.environ.get("YMAP_TEST_POSTGRES_DSN") and importlib.util.find_spec("psycopg"), "Set YMAP_TEST_POSTGRES_DSN and install psycopg to run isolated real-database integration verification.")
    def test_real_postgres_all_supported_customer_mappings(self):
        model = model_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = profile_candidates(generate_candidates(model), {"fixture_mode": "deterministic_generated", "dsn_env": "YMAP_TEST_POSTGRES_DSN", "rows_per_class": 3, "repetitions": 1, "warmup": 0, "seed": 9}, Path(tmp))
            with output.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["label_source"] == "postgresql_measured" and float(row["insert_ms"]) > 0 for row in rows))
            self.assertEqual(len({row["realization_id"] for row in rows}), 1)
            records = [json.loads(line) for line in (Path(tmp) / "raw_measurements.jsonl").read_text().splitlines()]
            self.assertTrue(all(abs(r["storage_mb"] - r["storage_bytes"] / 1_000_000) < 1e-12 for r in records))
            self.assertTrue(all(r["query_results"] == records[0]["query_results"] for r in records))

    @unittest.skipUnless(os.environ.get("YMAP_TEST_POSTGRES_DSN") and importlib.util.find_spec("psycopg"), "Set YMAP_TEST_POSTGRES_DSN and install psycopg for live PostgreSQL verification.")
    def test_real_postgres_tpc_leaf_associations_and_scalar_roundtrip(self):
        model = {"name": "TypedTPC", "classes": [{"name": "Base", "abstract": True, "attributes": [{"name": "amount", "type": "decimal"}, {"name": "date", "type": "date"}]}, {"name": "Leaf", "extends": "Base", "attributes": [{"name": "time", "type": "datetime"}, {"name": "enabled", "type": "bool"}]}, {"name": "OtherLeaf", "extends": "Base", "attributes": [{"name": "number", "type": "int"}]}, {"name": "Target", "attributes": [{"name": "text", "nullable": True}]}], "associations": [{"name": "LeafTargets", "source": "Leaf", "target": "Target"}]}
        with tempfile.TemporaryDirectory() as tmp:
            output = profile_candidates(generate_candidates(model), {"fixture_mode": "deterministic_generated", "dsn_env": "YMAP_TEST_POSTGRES_DSN", "rows_per_class": 3, "repetitions": 1, "warmup": 0, "seed": 9}, Path(tmp))
            with output.open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)
            for lifecycle in Path(tmp).glob("candidates/*/schema_lifecycle.json"):
                self.assertEqual(json.loads(lifecycle.read_text())["cleanup_status"], "dropped")

    @unittest.skipUnless(os.environ.get("YMAP_TEST_POSTGRES_DSN") and importlib.util.find_spec("psycopg"), "Set YMAP_TEST_POSTGRES_DSN and install psycopg for live constraint verification.")
    def test_real_postgres_rejects_wrong_subtype_duplicate_one_to_one_and_missing_required_link(self):
        import psycopg
        model = {"name": "ConstraintChecks", "classes": [{"name": "Identity", "abstract": True}, {"name": "Allowed", "extends": "Identity"}, {"name": "Forbidden", "extends": "Identity"}, {"name": "Target"}], "associations": [{"name": "Pair", "source": "Allowed", "target": "Target", "multiplicity": "one-to-one"}]}
        fixture = build_logical_fixture(model, 2, 3)
        forbidden = next(o["id"] for o in fixture["objects"] if o["class"] == "Forbidden")
        targets = sorted(o["id"] for o in fixture["objects"] if o["class"] == "Target")
        for candidate in generate_candidates(model, max_candidates=2):
            with self.subTest(strategy=candidate["decisions"]), psycopg.connect(os.environ["YMAP_TEST_POSTGRES_DSN"], autocommit=True) as conn:
                conn.prepare_threshold = None
                schema = "ymap_test_" + uuid.uuid4().hex
                created = False
                try:
                    conn.execute("CREATE SCHEMA " + quote(schema))
                    created = True
                    conn.execute("SET search_path TO " + quote(schema) + ", pg_catalog")
                    conn.execute(candidate["ddl"], prepare=False)
                    with conn.transaction():
                        with conn.cursor() as cur:
                            for plan in build_load_plan(candidate, fixture).values():
                                cur.executemany(plan["sql"], plan["rows"])
                    am = candidate["association_mapping"]["Pair"]
                    source_id, source_type = am["source_columns"]
                    target_id = am["target_columns"][0]
                    # Root-table FKs alone would accept this sibling object;
                    # endpoint discriminator constraints must reject it.
                    with self.assertRaises(psycopg.Error):
                        with conn.transaction():
                            conn.execute("UPDATE " + quote(am["table"]) + " SET " + quote(source_id) + "=%s, " + quote(source_type) + "=%s WHERE " + quote(target_id) + "=%s", (forbidden, "Forbidden", targets[0]), prepare=False)
                    source_for_second = next(s for s, t in fixture["links"]["Pair"] if t == targets[1])
                    with self.assertRaises(psycopg.Error):
                        with conn.transaction():
                            conn.execute("UPDATE " + quote(am["table"]) + " SET " + quote(source_id) + "=%s WHERE " + quote(target_id) + "=%s", (source_for_second, targets[0]), prepare=False)
                    if am["strategy"] == "association_table":
                        with self.assertRaises(psycopg.Error):
                            with conn.transaction():
                                conn.execute("DELETE FROM " + quote(am["table"]) + " WHERE " + quote(target_id) + "=%s", (targets[0],), prepare=False)
                    else:
                        with self.assertRaises(psycopg.Error):
                            with conn.transaction():
                                conn.execute("UPDATE " + quote(am["table"]) + " SET " + quote(source_type) + "=NULL WHERE " + quote(target_id) + "=%s", (targets[0],), prepare=False)
                finally:
                    if created:
                        conn.execute("SET search_path TO pg_catalog")
                        conn.execute("DROP SCHEMA " + quote(schema) + " CASCADE")


if __name__ == "__main__":
    unittest.main()
