"""Build one verified folder for the 12,500 PostgreSQL schema files.

Requires only Python 3.9+. Reads the original Y-Map cohort folder (or ZIP)
and the output of convert_ymap_archive_to_postgresql.py. No input is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


EXPECTED = {"native": 728, "archive": 11_772}
SQL_SUBDIR = "Dataset/YMap/All_SQL_Solutions"
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
COHORT_FIELDS = {
    "cohort_index", "scenario", "object_model", "schema_id", "sql_path", "sql_sha256_lf",
    "mapping_status", "sql_dialect_status", "common_protocol_status",
}
CONVERSION_FIELDS = {
    "schema_id", "object_model", "source_sql_path", "source_sql_sha256_lf",
    "postgresql_sql_path", "postgresql_sql_sha256", "status",
}
NEW_FIELDS = ("source_sql_sha256_lf", "postgresql_sql_sha256", "postgresql_origin")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_lf(data):
    return sha(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def read_csv(data, name, required_fields):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = list(reader.fieldnames or ())
    check(len(fields) == len(set(fields)) and required_fields <= set(fields),
          f"{name} has missing or duplicated columns")
    rows = list(reader)
    check(all(None not in row and None not in row.values() for row in rows),
          f"{name} has rows with the wrong number of fields")
    return fields, rows


def by_id(rows, name):
    result = {}
    for row in rows:
        identifier = row["schema_id"]
        check(bool(identifier) and identifier not in result, f"{name} has an empty or duplicate schema ID: {identifier!r}")
        result[identifier] = row
    return result


def relative_path(row):
    raw = row["sql_path"]
    p = PurePosixPath(raw)
    parts = p.parts
    check(len(parts) == 6 and raw == p.as_posix() and "/".join(parts[:3]) == SQL_SUBDIR
          and parts[3] == row["object_model"] and parts[4] == "SQL_Solutions"
          and p.suffix == ".sql" and all(SAFE_COMPONENT.fullmatch(part) for part in parts),
          f"Unexpected SQL path: {raw!r}")
    return raw


def write_file(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def prepare(original, converted, destination):
    original, converted, destination = (path.resolve() for path in (original, converted, destination))
    check(original.is_dir() or (original.is_file() and original.suffix.lower() == ".zip"),
          f"Original cohort folder or ZIP not found: {original}")
    check(converted.is_dir(), f"Conversion output folder not found: {converted}")
    check(not destination.exists(), f"Output already exists; use a new folder: {destination}")

    with ExitStack() as stack:
        bundle = stack.enter_context(ZipFile(original)) if original.is_file() else None
        if bundle is not None:
            names = bundle.namelist()
            check(len(names) == len(set(names)), "Original cohort ZIP contains duplicated names")

        def read_original(path):
            return bundle.read(path) if bundle is not None else (original / path).read_bytes()

        source_manifest = read_original("candidates.csv")
        bundle_manifest = read_original("sql_bundle_manifest.json")
        info = json.loads(bundle_manifest)
        check(info["sql_file_count"] == sum(EXPECTED.values()) and info["candidates_csv_sha256"] == sha(source_manifest),
              "Original candidates.csv does not match its bundle manifest")
        fields, rows = read_csv(source_manifest, "Original candidates.csv", COHORT_FIELDS)
        check(not set(NEW_FIELDS).intersection(fields), "Pass the original cohort, not an already combined folder")
        by_id(rows, "Original candidates.csv")
        observed = Counter(row["scenario"] for row in rows)
        check(observed == EXPECTED, f"Unexpected original cohort counts: {dict(observed)}")
        paths = [relative_path(row) for row in rows]
        check(len(paths) == len({path.casefold() for path in paths}), "Original cohort has repeated SQL paths")

        report = (converted / "conversion_report.csv").read_bytes()
        summary_bytes = (converted / "conversion_summary.json").read_bytes()
        summary = json.loads(summary_bytes)
        check(summary.get("converted") == EXPECTED["archive"] and summary.get("failed") == 0,
              "The conversion summary does not report 11,772 successful conversions")
        _, converted_rows = read_csv(report, "Conversion report", CONVERSION_FIELDS)
        conversions = by_id(converted_rows, "Conversion report")
        archive_ids = {row["schema_id"] for row in rows if row["scenario"] == "archive"}
        check(conversions.keys() == archive_ids, "Conversion report IDs do not exactly match the archive cohort")
        for row in rows:
            if row["scenario"] != "archive":
                continue
            entry = conversions[row["schema_id"]]
            check(entry["status"] == "converted" and not entry.get("error")
                  and entry["object_model"] == row["object_model"]
                  and entry["source_sql_path"] == row["sql_path"]
                  and entry["postgresql_sql_path"] == row["sql_path"]
                  and entry["source_sql_sha256_lf"] == row["sql_sha256_lf"],
                  f"Conversion report does not match the original row: {row['schema_id']}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        # Build in a temporary sibling and publish only after every checksum matches.
        with TemporaryDirectory(prefix=".ymap-stage-", dir=destination.parent) as work:
            staging = Path(work) / "cohort"
            staging.mkdir()
            with (staging / "candidates.csv").open("x", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields + list(NEW_FIELDS))
                writer.writeheader()
                for row in rows:
                    path, sid = row["sql_path"], row["schema_id"]
                    source_data = read_original(path)
                    old_sha = row["sql_sha256_lf"]
                    check(sha_lf(source_data) == old_sha, f"Original SQL checksum mismatch: {sid}")
                    is_archive = row["scenario"] == "archive"
                    data = (converted / path).read_bytes() if is_archive else source_data
                    exact = sha(data)
                    if is_archive:
                        check(exact == conversions[sid]["postgresql_sql_sha256"],
                              f"Converted PostgreSQL SQL checksum mismatch: {sid}")
                    write_file(staging, path, data)
                    writer.writerow({
                        **row,
                        "sql_sha256_lf": sha_lf(data),
                        "source_sql_sha256_lf": old_sha,
                        "postgresql_sql_sha256": exact,
                        "postgresql_origin": "converted_from_archive" if is_archive else "native_unchanged",
                        "sql_dialect_status": "postgresql_converted" if is_archive else row["sql_dialect_status"],
                    })

            originals = {
                "candidates.csv": source_manifest,
                "sql_bundle_manifest.json": bundle_manifest,
                **{name: read_original(name) for name in ("source_systems.csv", "selection_manifest.json")},
            }
            for name, data in originals.items():
                write_file(staging, f"provenance/original/{name}", data)
            for name in ("LICENSE-YMap.md", "LICENSE-Archive-CC-BY-NC-4.0.md", "UPSTREAM_DATASET_README.md"):
                write_file(staging, f"provenance/{name}", read_original(f"provenance/{name}"))
            write_file(staging, "provenance/conversion/conversion_report.csv", report)
            write_file(staging, "provenance/conversion/conversion_summary.json", summary_bytes)
            renamed = converted / "renamed_identifiers.jsonl"
            if renamed.is_file():
                write_file(staging, "provenance/conversion/renamed_identifiers.jsonl", renamed.read_bytes())

            manifest = {
                "scope": "verified_postgresql_ddl_inventory",
                "sql_file_count": sum(observed.values()),
                "native_unchanged": observed["native"],
                "archive_converted": observed["archive"],
                "candidates_csv_sha256": sha((staging / "candidates.csv").read_bytes()),
                "original_candidates_csv_sha256": sha(source_manifest),
                "conversion_report_sha256": sha(report),
                "new_measurements": False,
                "common_protocol_verified": False,
                "archive_mapping_recovered": False,
            }
            write_file(staging, "postgresql_bundle_manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
            write_file(staging, "README.txt", (
                "Y-Map PostgreSQL schema inventory\n"
                "12,500 SQL files: 728 unchanged native and 11,772 converted archive.\n"
                "Use candidates.csv in this folder; sql_path values are relative to this folder.\n"
                "sql_sha256_lf and postgresql_sql_sha256 identify the current SQL.\n"
                "source_sql_sha256_lf identifies the original SQL. Original licenses and reports are in provenance/.\n"
                "Each file describes a separate schema design. This folder contains DDL only.\n"
                "Archive mapping and common-protocol performance measurements have not been produced.\n"
            ).encode())
            check(not destination.exists(), f"Output was created during preparation: {destination}")
            staging.rename(destination)

    print("PostgreSQL SQL files prepared: 12,500")
    print("Native PostgreSQL files copied: 728")
    print("Converted archive files copied: 11,772")
    print(f"Updated list: {destination / 'candidates.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="original 12,500-schema cohort folder or ZIP")
    parser.add_argument("converted", type=Path, help="folder from the archive conversion script")
    parser.add_argument("destination", type=Path, help="new folder for the complete PostgreSQL cohort")
    args = parser.parse_args()
    try:
        prepare(args.original, args.converted, args.destination)
    except (OSError, ValueError, KeyError, BadZipFile, csv.Error) as exc:
        print(f"Could not prepare PostgreSQL cohort: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
