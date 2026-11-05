import copy
import os
from pathlib import Path
import tempfile
import unittest

from ymap_research.synthesis import AlloyUnavailableError, ModelValidationError, UnsupportedMappingError, canonical_candidate, generate_candidates, validate_candidate, validate_model


def hierarchy_model():
    return {"name": "DeepHierarchy", "classes": [{"name": "Person", "attributes": [{"name": "name", "type": "string"}]}, {"name": "Student", "extends": "Person", "attributes": [{"name": "score", "type": "float"}]}, {"name": "DoctoralStudent", "extends": "Student", "attributes": [{"name": "topic", "type": "string"}]}], "associations": []}


class SynthesisTests(unittest.TestCase):
    def test_streaming_prefix_stable_and_model_sensitive_identity(self):
        model = hierarchy_model()
        candidates = list(generate_candidates(model))
        self.assertEqual(len(candidates), 3)
        self.assertEqual(list(generate_candidates(model, 1)), candidates[:1])
        changed = copy.deepcopy(model)
        changed["classes"][0]["attributes"][0]["type"] = "int"
        self.assertNotEqual(candidates[0]["schema_id"], next(generate_candidates(changed))["schema_id"])

    def test_multilevel_joined_references_direct_parent(self):
        joined = next(c for c in generate_candidates(hierarchy_model()) if c["decisions"]["inheritance:Person"] == "joined")
        tables = {t["name"]: t for t in joined["tables"]}
        self.assertEqual(tables["c_DoctoralStudent"]["foreign_keys"][0]["references"]["table"], "c_Student")
        self.assertEqual(tables["c_Student"]["foreign_keys"][0]["references"]["table"], "c_Person")
        self.assertIn("a_DoctoralStudent_topic", [c["name"] for c in tables["c_DoctoralStudent"]["columns"]])
        self.assertNotIn("a_Person_name", [c["name"] for c in tables["c_DoctoralStudent"]["columns"]])

    def test_single_table_includes_grandchild_and_conditional_not_null(self):
        single = next(generate_candidates(hierarchy_model()))
        self.assertEqual(len(single["tables"]), 1)
        self.assertIn("a_DoctoralStudent_topic", [c["name"] for c in single["tables"][0]["columns"]])
        self.assertIn('"a_DoctoralStudent_topic" IS NOT NULL', single["ddl"])
        self.assertIn("'DoctoralStudent'", single["ddl"])

    def test_tpc_copies_all_ancestors(self):
        tpc = next(c for c in generate_candidates(hierarchy_model()) if c["decisions"]["inheritance:Person"] == "table_per_class")
        table = next(t for t in tpc["tables"] if t["name"] == "c_DoctoralStudent")
        self.assertEqual({c["name"] for c in table["columns"]}, {"id", "a_Person_name", "a_Student_score", "a_DoctoralStudent_topic"})

    def test_polymorphic_endpoint_rejects_tpc_explicitly_and_types_sti_fk(self):
        model = hierarchy_model()
        model["classes"].append({"name": "Course", "attributes": []})
        model["associations"] = [{"name": "Mentoring", "source": "Student", "target": "Course", "multiplicity": "one-to-many"}]
        candidates = list(generate_candidates(model))
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(c["decisions"]["inheritance:Person"] != "table_per_class" for c in candidates))
        self.assertTrue(all(c["generation"]["excluded_choices"] for c in candidates))
        single = candidates[0]
        am = single["association_mapping"]["Mentoring"]
        self.assertEqual(len(am["source_columns"]), 2)
        self.assertIn("'Student','DoctoralStudent'", single["ddl"].replace("'DoctoralStudent','Student'", "'Student','DoctoralStudent'"))

    def test_one_to_one_has_unique_source_and_target_and_required_trigger(self):
        model = {"name": "Pair", "classes": [{"name": "A"}, {"name": "B"}], "associations": [{"name": "AB", "source": "A", "target": "B", "multiplicity": "one-to-one"}]}
        candidates = list(generate_candidates(model))
        direct, link = candidates
        self.assertIn([direct["association_mapping"]["AB"]["source_columns"][0]], next(t for t in direct["tables"] if t["name"] == "c_B")["unique"])
        link_table = next(t for t in link["tables"] if t["kind"] == "association")
        self.assertEqual(len(link_table["unique"]), 2)
        self.assertIn("CREATE CONSTRAINT TRIGGER", link["ddl"])
        self.assertIn("DEFERRABLE INITIALLY DEFERRED", link["ddl"])

    def test_bad_models_rejected(self):
        for mutation in ("cycle", "type", "shadow"):
            model = hierarchy_model()
            if mutation == "cycle":
                model["classes"][0]["extends"] = "DoctoralStudent"
            elif mutation == "type":
                model["classes"][0]["attributes"][0]["type"] = "sql injection"
            else:
                model["classes"][1]["attributes"].append({"name": "name"})
            with self.assertRaises(ModelValidationError):
                validate_model(model)

    def test_structural_validator_detects_dangling_fk(self):
        candidate = list(generate_candidates(hierarchy_model()))[1]
        candidate["tables"][0]["foreign_keys"][0]["references"]["table"] = "missing"
        with self.assertRaises(UnsupportedMappingError):
            validate_candidate(candidate)

    def test_alloy_never_silently_falls_back(self):
        with self.assertRaises(AlloyUnavailableError):
            list(generate_candidates(hierarchy_model(), backend="alloy"))

    def test_nonstring_names_and_parent_types_rejected(self):
        for key, value in (("name", None), ("name", True), ("extends", [])):
            model = hierarchy_model()
            model["classes"][0][key] = value
            with self.assertRaises(ModelValidationError):
                validate_model(model)
        model = hierarchy_model()
        model["associations"] = [{"name": "BadEndpoint", "source": [], "target": "Person"}]
        with self.assertRaises(ModelValidationError):
            validate_model(model)

    def test_unsupported_semantics_not_silently_ignored(self):
        model = hierarchy_model()
        model["classes"][0]["attributes"][0]["unique"] = True
        with self.assertRaisesRegex(ModelValidationError, "Unsupported attribute"):
            validate_model(model)

    def test_ambiguous_physical_attribute_names_rejected(self):
        model = {"name": "AmbiguousColumns", "classes": [{"name": "A", "attributes": [{"name": "B_C"}]}, {"name": "A_B", "extends": "A", "attributes": [{"name": "C"}]}]}
        with self.assertRaisesRegex(ModelValidationError, "collide"):
            validate_model(model)

    def test_canonical_candidate_rejects_tampered_or_stale_ddl(self):
        candidate = next(generate_candidates(hierarchy_model()))
        self.assertEqual(canonical_candidate(candidate), candidate)
        candidate["ddl"] += "\nDROP SCHEMA public CASCADE;"
        with self.assertRaisesRegex(UnsupportedMappingError, "stale or modified"):
            canonical_candidate(candidate)

    @unittest.skipUnless(os.environ.get("YMAP_TEST_ALLOY_JAR"), "Set YMAP_TEST_ALLOY_JAR and add a JDK's bin directory to PATH for actual Alloy API integration.")
    def test_real_alloy_enumerates_same_finite_choices_including_empty_product(self):
        empty_product = {"name": "NoChoices", "classes": [{"name": "Entity"}]}
        assoc_model = hierarchy_model()
        assoc_model["classes"].append({"name": "Course"})
        assoc_model["associations"] = [{"name": "Enroll", "source": "Student", "target": "Course"}]
        for model in (empty_product, hierarchy_model(), assoc_model):
            with self.subTest(model=model["name"]), tempfile.TemporaryDirectory() as tmp:
                python = list(generate_candidates(model))
                alloy = list(generate_candidates(model, backend="alloy", alloy_jar=os.environ["YMAP_TEST_ALLOY_JAR"], artifact_dir=Path(tmp)))
                self.assertEqual({c["schema_id"] for c in python}, {c["schema_id"] for c in alloy})
                self.assertEqual(len(python), len(alloy))


if __name__ == "__main__":
    unittest.main()
