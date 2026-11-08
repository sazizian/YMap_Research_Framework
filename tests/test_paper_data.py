from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from ymap_research.evaluation import load_dataset
from ymap_research.paper_data import file_sha256, resolve_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_inputs(candidates: Path, measurements: Path, rows: list[tuple]) -> None:
    records = [
        {"schema_id": schema_id, "object_model": group, "features": {"table_count": i + 1}}
        for i, (schema_id, group) in enumerate(dict.fromkeys((row[0], row[1]) for row in rows))
    ]
    candidates.parent.mkdir(parents=True, exist_ok=True)
    measurements.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with measurements.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["schema_id", "object_model", "label_source", "workload_id", "environment_id", "seed", "insert_ms", "query_ms", "storage_mb"])
        writer.writerows(rows)


class DatasetManifestTests(unittest.TestCase):
    def test_archive_filters_the_shared_catalog_without_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ("local", "native_system", "postgresql_measured", "workload_8872893b0f8e92c3", "environment_dae41b366b570b7b", 42, 1, 2, 3),
                ("old", "archive_system", "imported_measured", "reinforceorm_supplied", "reinforceorm_supplied_units_unconfirmed", 1, 100, 200, 300),
                ("old", "archive_system", "imported_measured", "reinforceorm_supplied", "reinforceorm_supplied_units_unconfirmed", 2, 102, 202, 302),
            ]
            write_inputs(root / "all_candidates.jsonl", root / "all_measurements.csv", rows)
            dataset = resolve_dataset(PROJECT_ROOT, "archive", dataset_root=root)
            _, data, _ = load_dataset(dataset.candidates_path, dataset.measurements_path, dataset.evaluation_config())
            self.assertEqual(data.schema_id.tolist(), ["old"])
            self.assertEqual(data.insert_ms.tolist(), [101.0])
            self.assertEqual(data.measurement_seed_count.tolist(), [2])
            self.assertFalse(dataset.metadata()["units_confirmed"])
            with self.assertRaisesRegex(ValueError, "expected 27488 measured_candidates"):
                dataset.check_alignment(data)
            with self.assertRaisesRegex(ValueError, "exactly one workload_id/environment_id"):
                load_dataset(dataset.candidates_path, dataset.measurements_path)

    def test_original_inputs_require_matching_hashes_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, measurements = root / "original.jsonl", root / "original.csv"
            write_inputs(candidates, measurements, [
                ("ref1", "system1", "postgresql_measured", "original_workload", "original_environment", 42, 1, 2, 3),
                ("ref2", "system2", "postgresql_measured", "original_workload", "original_environment", 42, 2, 4, 6),
            ])
            manifest = root / "reference.yaml"
            entry = {
                "scope": "Original reference",
                "candidates": candidates.name,
                "measurements": measurements.name,
                "workload_id": "original_workload",
                "environment_id": "original_environment",
                "label_source": "postgresql_measured",
                "units_confirmed": True,
                "counts": {"measured_candidates": 2, "source_systems": 2, "measurement_rows": 2},
                "provenance": {
                    "source": "Original artifact",
                    "citation": "Artifact citation",
                    "protocol": "Reference measurements on PostgreSQL",
                    "input_sha256": {"candidates": file_sha256(candidates), "measurements": file_sha256(measurements)},
                },
            }

            def write_manifest():
                manifest.write_text(yaml.safe_dump({"version": 1, "datasets": {"paper-reference": entry}}), encoding="utf-8")

            write_manifest()
            dataset = resolve_dataset(PROJECT_ROOT, "paper-reference", dataset_root=root, manifest_path=manifest)
            _, data, _ = load_dataset(dataset.candidates_path, dataset.measurements_path, dataset.evaluation_config())
            self.assertEqual(dataset.check_alignment(data), entry["counts"])
            with self.assertRaisesRegex(ValueError, "explicit dataset manifest"):
                resolve_dataset(PROJECT_ROOT, "paper-reference", dataset_root=root)

            entry["candidates"] = "../original.jsonl"
            write_manifest()
            with self.assertRaisesRegex(ValueError, "relative path within the dataset root"):
                resolve_dataset(PROJECT_ROOT, "paper-reference", dataset_root=root, manifest_path=manifest)

            entry["candidates"] = candidates.name
            citation = entry["provenance"].pop("citation")
            write_manifest()
            with self.assertRaisesRegex(ValueError, "provenance.citation"):
                resolve_dataset(PROJECT_ROOT, "paper-reference", dataset_root=root, manifest_path=manifest)

            entry["provenance"]["citation"] = citation
            write_manifest()
            with measurements.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "measurements SHA-256 does not match"):
                resolve_dataset(PROJECT_ROOT, "paper-reference", dataset_root=root, manifest_path=manifest)
