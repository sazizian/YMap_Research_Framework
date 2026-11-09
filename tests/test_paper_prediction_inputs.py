from copy import deepcopy
from unittest import TestCase

from ymap_research.paper_prediction_inputs import native_prediction_records
from ymap_research.synthesis import UnsupportedMappingError, generate_candidates


def _native_candidate():
    model = {
        "name": "Measured",
        "classes": [
            {"name": "Account", "attributes": [{"name": "number"}]},
            {"name": "Special", "extends": "Account"},
            {"name": "Other"},
        ],
        "associations": [
            {"name": "account", "source": "Other", "target": "Account", "multiplicity": "one-to-many"},
        ],
    }
    return next(
        record for record in generate_candidates(model)
        if record["decisions"] == {
            "inheritance:Account": "single_table",
            "association:account": "association_table",
        }
    )


class PaperPredictionInputsTests(TestCase):
    def test_preserves_measured_schema_and_uses_its_physical_column_flags(self):
        record = _native_candidate()
        snapshot = deepcopy(record)
        adapted, = native_prediction_records([record])

        self.assertEqual(record, snapshot)
        self.assertEqual({key: adapted[key] for key in snapshot}, snapshot)
        self.assertNotIn("paper_ddl", adapted)

        columns = {node["id"]: node for node in adapted["paper_graph"]["nodes"] if node["type"] == "column"}
        for table in record["tables"]:
            for column in table["columns"]:
                with self.subTest(table=table["name"], column=column["name"]):
                    node = columns[f"column:{table['name']}.{column['name']}"]
                    self.assertEqual(node["nullable"], column["nullable"])
                    self.assertEqual(node["primary_key"], column["primary_key"])

        # The paper compiler would tighten the base attribute's nullability
        # and replace this native composite primary key with the many-side ID.
        self.assertTrue(columns["column:c_Account.a_Account_number"]["nullable"])
        self.assertTrue(columns["column:l_account.r_account_source_id"]["primary_key"])
        self.assertTrue(columns["column:l_account.r_account_target_id"]["primary_key"])
        features = adapted["paper_features"]
        self.assertEqual(features["pk_column_count"], record["features"]["pk_column_count"])
        self.assertEqual(features["nullable_fraction"], record["features"]["nullable_column_count"] / len(columns))

    def test_rejects_tampered_ddl_before_deriving_prediction_views(self):
        record = _native_candidate()
        record["ddl"] = record["ddl"].replace('"a_Account_number" TEXT', '"a_Account_number" TEXT NOT NULL')
        with self.assertRaisesRegex(UnsupportedMappingError, "stale or modified"):
            native_prediction_records([record])
