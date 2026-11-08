import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from ymap_research.paper_schema import (
    PAPER_FEATURES,
    PAPER_RELATIONS,
    paper_candidates,
    paper_executable_candidate,
    paper_features,
    paper_graph,
)
from ymap_research.synthesis import UnsupportedMappingError


def _model():
    return {
        "name": "PaperExample",
        "classes": [
            {"name": "Account", "attributes": [{"name": "z"}, {"name": "a"}]},
            {"name": "Special", "extends": "Account", "attributes": [{"name": "benefit"}]},
            {"name": "Other"},
        ],
        "associations": [
            {"name": "account", "source": "Other", "target": "Account", "multiplicity": "one-to-many"},
        ],
    }


class PaperSchemaTests(TestCase):
    def test_scope_descriptors_and_graph(self):
        counts = {}
        bounded = list(paper_candidates(_model(), max_tables=3, backend="python", counts=counts))
        self.assertEqual(counts, dict(decision_vectors=4, out_of_scope=1, enumerated=3, duplicates=0, unique=3))
        self.assertEqual({r["paper_features"]["table_count"] for r in bounded}, {2, 3})
        self.assertNotIn({"inheritance:Account": "joined", "association:account": "association_table"}, [r["decisions"] for r in bounded])
        self.assertTrue(any("polymorphic association endpoints" in note for note in bounded[0]["generation"]["excluded_choices"]))

        joined = next(r for r in bounded if r["decisions"]["inheritance:Account"] == "joined")
        descriptors = paper_features(joined)
        self.assertEqual(tuple(descriptors), PAPER_FEATURES)
        self.assertEqual(tuple(descriptors.values()), (3, 9, 3, 2, 0, 1, 1.0, 3.0, 5, 2 / 3, 0.0, 1 / 3))
        graph = paper_graph(joined)
        self.assertEqual({edge["type"] for edge in graph["edges"]}, set(PAPER_RELATIONS))
        self.assertIn({"source": "table:c_Special", "target": "table:c_Account", "type": "inheritsFrom"}, graph["edges"])
        self.assertIn({"source": "column:c_Special.id", "target": "table:c_Account", "type": "references"}, graph["edges"])
        self.assertIn({"id": "column:c_Special.dtype", "type": "column", "data_type": "TEXT", "primary_key": False, "foreign_key": True, "nullable": False, "unique": False, "discriminator": True}, graph["nodes"])

    def test_paper_association_and_nullability_rules(self):
        model = _model()
        model["associations"].append({"name": "one", "source": "Other", "target": "Special", "multiplicity": "one-to-one"})
        candidates = list(paper_candidates(model, max_tables=5, backend="python"))
        self.assertTrue(all(c["decisions"]["association:one"] == "direct_fk" for c in candidates))
        candidate = next(c for c in candidates if c["decisions"]["inheritance:Account"] == "single_table" and c["decisions"]["association:account"] == "association_table")
        ddl = candidate["paper_ddl"]
        self.assertIn('"a_Account_a" TEXT NOT NULL', ddl)
        self.assertIn('"a_Special_benefit" TEXT,', ddl)
        self.assertIn('"r_one_source_id" BIGINT,', ddl)
        self.assertIn('UNIQUE ("r_one_source_id")', ddl)
        self.assertIn('PRIMARY KEY ("r_account_target_id")', ddl)
        self.assertNotIn('UNIQUE ("r_account_target_id")', ddl)
        self.assertIn('CREATE CONSTRAINT TRIGGER', ddl)

    def test_canonical_order_delays_only_forward_cycle_references(self):
        model = {
            "name": "Cycle", "classes": [{"name": name} for name in ("Z", "C", "B", "A")],
            "associations": [
                {"name": "back", "source": "A", "target": "B"},
                {"name": "forward", "source": "B", "target": "A"},
                {"name": "external", "source": "Z", "target": "B"},
                {"name": "self", "source": "C", "target": "C", "nullable": True},
            ],
        }
        candidate = next(paper_candidates(model, max_tables=4, backend="python"))
        ddl = candidate["paper_ddl"]
        lines = ddl.splitlines()
        self.assertEqual([line.split('"')[1] for line in lines[:4]], ["c_C", "c_Z", "c_A", "c_B"])
        self.assertIn('REFERENCES "c_C"', lines[0])
        self.assertIn('REFERENCES "c_A"', lines[3])
        self.assertIn('REFERENCES "c_Z"', lines[3])
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[4].startswith('ALTER TABLE "c_A"'))
        self.assertIn('REFERENCES "c_B"', lines[4])
        self.assertEqual(candidate["paper_ddl_sha256"], sha256(ddl.encode()).hexdigest())

        reversed_model = {**model, "classes": model["classes"][::-1], "associations": model["associations"][::-1]}
        reordered = next(paper_candidates(reversed_model, max_tables=4, backend="python"))
        self.assertEqual(reordered["paper_ddl_sha256"], candidate["paper_ddl_sha256"])
        self.assertEqual(reordered["schema_id"], candidate["schema_id"])
        self.assertNotEqual(candidate["compiler_schema_id"], candidate["schema_id"])
        executable = paper_executable_candidate(candidate)
        self.assertEqual(executable["ddl"], ddl)
        self.assertEqual(paper_executable_candidate(executable), executable)
        tampered = deepcopy(candidate)
        tampered["paper_ddl"] += "\nDROP TABLE c_A;"
        tampered["paper_ddl_sha256"] = sha256(tampered["paper_ddl"].encode()).hexdigest()
        with self.assertRaises(UnsupportedMappingError):
            paper_executable_candidate(tampered)

    def test_seed_and_duplicate_solver_vectors(self):
        args = dict(max_tables=5, backend="python")
        def keys(seed):
            return [record["paper_ddl_sha256"] for record in paper_candidates(_model(), **args, seed=seed)]
        self.assertEqual(keys(37), keys(37))
        self.assertNotEqual(keys(37), keys(17))
        self.assertEqual(set(keys(37)), set(keys(17)))

        def repeated(*_):
            yield {"inheritance:Account": "single_table", "association:account": "direct_fk"}
            yield {"inheritance:Account": "single_table", "association:account": "direct_fk"}

        with patch("ymap_research.paper_schema._alloy_decisions", repeated):
            for deduplicate, expected in [(True, 1), (False, 2)]:
                counts = {}
                records = list(paper_candidates(_model(), max_tables=5, backend="alloy", counts=counts, deduplicate=deduplicate))
                self.assertEqual(len(records), expected)
                self.assertEqual(counts, dict(decision_vectors=2, out_of_scope=0, enumerated=2, duplicates=1, unique=1))

    @skipUnless(os.environ.get("YMAP_TEST_ALLOY_JAR"), "requires external Alloy JAR and JDK")
    def test_alloy_matches_bounded_python_choices(self):
        with TemporaryDirectory() as directory:
            args = dict(max_tables=3)
            python = list(paper_candidates(_model(), **args, backend="python"))
            alloy = list(paper_candidates(_model(), **args, backend="alloy", alloy_jar=os.environ["YMAP_TEST_ALLOY_JAR"], artifact_dir=Path(directory)))
            self.assertEqual({record["schema_id"] for record in python}, {record["schema_id"] for record in alloy})
