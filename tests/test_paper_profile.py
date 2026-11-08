import csv
from contextlib import closing
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ymap_research.paper_profile import (
    ProfilingError, TEMPLATES, _configuration, _fixture, _fresh_database, _insert_rows, _query_sql,
    _realization, _same_rows, _schedule, _system_config, profile_paper,
)
from ymap_research.paper_schema import paper_candidates, paper_executable_candidate
from ymap_research.profiling import build_load_plan
from ymap_research.synthesis import model_index, quote


def fixture():
    model = {
        "name": "PaperFixture",
        "classes": [
            {"name": "Parent", "attributes": [{"name": "flag", "type": "int"}]},
            {"name": "Child", "extends": "Parent", "attributes": [{"name": "score", "type": "int"}]},
            {"name": "Item", "attributes": [{"name": "value", "type": "int"}]},
            {"name": "Part", "attributes": [{"name": "amount", "type": "int"}]},
        ],
        "associations": [
            {"name": "Items", "source": "Child", "target": "Item"},
            {"name": "Parts", "source": "Item", "target": "Part"},
        ],
    }
    config = {
        "mode": "smoke", "confirm_disposable_server": True, "seed": 42,
        "population_by_system": {"PaperFixture": {"Parent": 5, "Child": 5, "Item": 7, "Part": 14}},
        "workload_by_system": {"PaperFixture": {
            "T1": {"class": "Parent"}, "T2": {"class": "Child"},
            "T3": {"association": "Items"}, "T4": {"associations": ["Items", "Parts"]},
            "T5": {"association": "Parts", "attribute": "amount", "operation": "sum"},
            "insert_classes": ["Child", "Part"],
        }},
    }
    return model, config


class PaperProfileTests(unittest.TestCase):
    def test_strict_procedure_guards_and_interleaving(self):
        _, config = fixture()
        config.update(mode="reference", warmup_inserts=500)
        with self.assertRaisesRegex(ProfilingError, "cache_reset_command"):
            _configuration(config)
        config.update(confirm_dedicated_host=True, cache_reset_command=[sys.executable],
                      environment_notes="Dedicated profiling host", repetitions=2)
        with self.assertRaisesRegex(ProfilingError, "repetitions must be 5"):
            _configuration(config)
        config["repetitions"] = 5
        validated = _configuration(config)
        self.assertEqual((validated["warmup_operations"], validated["insert_operations"],
                          validated["query_operations"]), (1000, 10000, 10000))
        order = list(_schedule(2, 5, 7))
        self.assertTrue(all(a[1] != b[1] for a, b in zip(order, order[1:])))
        self.assertEqual(order, list(_schedule(2, 5, 7)))

    def test_unsupported_intents_and_variant_mixture_fail_closed(self):
        model, config = fixture()
        config["workload_by_system"]["PaperFixture"]["T4"]["directions"] = ["reverse", "forward"]
        with self.assertRaisesRegex(ProfilingError, "intermediate"):
            _system_config(model, _configuration(config))
        config["variant"] = "W_join"
        with self.assertRaisesRegex(ProfilingError, "template_weights|percentages"):
            _configuration(config)
        config["template_weights"] = dict(zip(TEMPLATES, [7, 7, 40, 40, 6]))
        self.assertEqual(_configuration(config)["template_weights"], config["template_weights"])
        config["template_weights"] = dict(zip(TEMPLATES, [6.67, 6.67, 40, 40, 6.66]))
        self.assertEqual(_configuration(config)["template_weights"], config["template_weights"])

    def test_database_identity_mismatch_is_never_dropped(self):
        _, config = fixture()
        config = _configuration(config)
        admin, cleanup, psycopg = MagicMock(), MagicMock(), MagicMock()
        create_cursor = admin.__enter__.return_value.cursor.return_value.__enter__.return_value
        clean_cursor = cleanup.__enter__.return_value.cursor.return_value.__enter__.return_value
        create_cursor.fetchone.side_effect = [("pre_restart",), (42,)]
        clean_cursor.fetchone.return_value = (43,)
        events = io.StringIO()
        with patch("ymap_research.paper_profile._admin_connect", side_effect=(admin, cleanup)), \
                patch("ymap_research.paper_profile._environment", return_value=({}, "new_start", "wrong_db")):
            with self.assertRaisesRegex(ProfilingError, "Cleanup failed"):
                _fresh_database(psycopg, "opaque", {"schema_id": "paper_fixture"}, {}, {}, config, {}, events, 0)
        self.assertFalse(any("DROP DATABASE" in str(call) for call in clean_cursor.execute.call_args_list))
        self.assertIn("database_cleanup_failed", events.getvalue())

    def test_tenfold_population_preserves_every_physical_table_count(self):
        model, config = fixture()
        population = config["population_by_system"][model["name"]]
        initial = _fixture(model, population, 9, "reference")
        larger = _fixture(model, population, 9, "D_10x")
        skew = _fixture(model, population, 9, "D_skew")
        self.assertNotEqual(skew["links"], initial["links"])
        self.assertEqual(larger, _fixture(model, population, 9, "D_10x"))
        for generated in paper_candidates(model, max_tables=12, backend="python"):
            candidate = paper_executable_candidate(generated)
            before, after = (build_load_plan(candidate, f) for f in (initial, larger))
            self.assertEqual({k: 10 * len(v["rows"]) for k, v in before.items()},
                             {k: len(v["rows"]) for k, v in after.items()})

    def test_all_query_intents_match_logical_oracle_across_mapping_strategies(self):
        model, config = fixture()
        config = _configuration(config)
        workload = _system_config(model, config)
        data = _realization(model, workload, config, 0)
        idx = model_index(model)
        tested = set()
        for generated in paper_candidates(model, max_tables=12, backend="python"):
            candidate = paper_executable_candidate(generated)
            tested.add(candidate["mapping"]["Child"]["strategy"])
            plan = build_load_plan(candidate, data["fixture"])
            sqls = _query_sql(candidate, workload)
            # SQLite checks the logical joins against the independently built
            # oracle, not PostgreSQL DDL, prepare, timing or constraint behavior.
            with closing(sqlite3.connect(":memory:")) as conn:
                for table, spec in plan.items():
                    conn.execute(f"CREATE TABLE {quote(table)} (" + ", ".join(map(quote, spec["columns"])) + ")")
                    conn.executemany(spec["sql"].replace("%s", "?"), spec["rows"])
                for record in data["inserts"]:
                    for table, row in _insert_rows(candidate, idx, record).items():
                        spec = plan[table]
                        conn.execute(spec["sql"].replace("%s", "?"), tuple(row.get(c) for c in spec["columns"]))
                for template, oid in set(data["timed"]):
                    sql = re.sub(r"::(?:DOUBLE PRECISION|[A-Za-z]+)", "", sqls[template]).replace("%s", "?")
                    observed = conn.execute(sql, (oid,)).fetchall()
                    self.assertTrue(_same_rows(observed, data["oracle_timed"][template, oid]),
                                    (candidate["mapping"], template, oid, observed))
        self.assertEqual(tested, {"single_table", "joined", "table_per_class"})

    @unittest.skipUnless(os.environ.get("YMAP_TEST_DISPOSABLE_POSTGRES_DSN") and importlib.util.find_spec("psycopg"),
                         "Requires opt-in disposable PostgreSQL with CREATE DATABASE privileges and psycopg.")
    def test_real_postgres_smoke_produces_medians_and_cleans_up_own_databases(self):
        model, config = fixture()
        config.update(dsn_env="YMAP_TEST_DISPOSABLE_POSTGRES_DSN")
        candidates = list(paper_candidates(model, max_tables=12, backend="python"))[:2]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            csv_path = profile_paper(candidates, config, out)
            with csv_path.open() as handle:
                labels = list(csv.DictReader(handle))
            self.assertEqual(len(labels), 2)
            self.assertEqual({row["label_source"] for row in labels}, {"postgresql_smoke"})
            self.assertEqual({row["repetitions"] for row in labels}, {"2"})
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            created = {e["database"] for e in events if e["event"] == "database_created"}
            dropped = {e["database"] for e in events if e["event"] == "database_dropped"}
            self.assertEqual(len(created), 4)
            self.assertEqual(created, dropped)
            with gzip.open(out / "raw_measurements.jsonl.gz", "rt") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(sum(record["timed_query_counts"].values()) == 30 for record in records))
