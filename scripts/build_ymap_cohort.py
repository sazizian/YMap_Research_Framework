"""Select a reproducible Y-Map candidate inventory; do not profile or train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile


PINNED_COMMIT = "b236d61601e7ca7daed7c8f28e5b7c719bf54291"
SEED = "ymap-candidate-cohort-v1"
TOTAL = 12_500
EXPECTED = {"native": (728, 11, 728), "archive": (27_488, 15, 31_393)}
SOURCES = {"ymap": "native", "reinforceorm": "archive"}
SCENARIOS = {
    "native": ("postgresql_measured", "workload_8872893b0f8e92c3", "environment_dae41b366b570b7b"),
    "archive": ("imported_measured", "reinforceorm_supplied", "reinforceorm_supplied_units_unconfirmed"),
}
ARCHIVE_NAMES = {
    "rorm_bank": "Bank", "rorm_csos": "CSOS", "rorm_camping": "Camping", "rorm_canteen": "Canteen",
    "rorm_customer_order": "Customer - Order", "rorm_decider": "Decider", "rorm_ecommerce": "E-Commerce",
    "rorm_flagship": "Flagship", "rorm_hospital": "Hospital", "rorm_library": "Library",
    "rorm_onlinestore": "Onlinestore", "rorm_order_processing": "Order-Processing",
    "rorm_store": "Store", "rorm_traffic_control": "Traffic-Control",
    "rorm_university_payroll": "University-Payroll",
}
FIELDS = (
    "cohort_index", "scenario", "object_model", "schema_id", "sql_path", "sql_sha256_lf",
    "existing_measurement_rows", "existing_units_confirmed", "mapping_status",
    "sql_dialect_status", "common_protocol_status",
)
INPUT_FILES = (
    "Generated/candidates.jsonl", "Measurements/ymap_postgresql_measurements.csv",
    "all_candidates.jsonl", "all_measurements.csv",
)
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def lf(data):
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        yield from (json.loads(line) for line in f if line.strip())


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def write_csv(path, fields, rows):
    # UTF-8 BOM enables straightforward viewing in Windows Excel.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def candidate_catalog(root):
    groups = defaultdict(list)
    candidates = {}
    paths = set()
    for row in jsonl(root / "all_candidates.jsonl"):
        scenario = SOURCES[row["source"]]
        schema_id, system, relative = (row[k] for k in ("schema_id", "object_model", "sql_path"))
        require(ID.fullmatch(schema_id) and ID.fullmatch(system), "Unexpected schema or system identifier")
        path = PurePosixPath(relative)
        require(
            len(path.parts) == 4 and path.parts[:3] == ("All_SQL_Solutions", system, "SQL_Solutions")
            and path.name.endswith(".sql") and ID.fullmatch(path.name) and relative == path.as_posix(),
            f"Invalid repository SQL path for {schema_id}: {relative}",
        )
        require(schema_id not in candidates and relative not in paths, f"Duplicate schema ID or SQL path: {schema_id}")
        candidates[schema_id] = row
        paths.add(relative)
        groups[scenario, system].append(row)

    native = {}
    for row in jsonl(root / "Generated/candidates.jsonl"):
        key = row["schema_id"]
        require(key not in native, f"Duplicate native schema ID: {key}")
        require(
            key in candidates and candidates[key]["source"] == "ymap"
            and candidates[key]["object_model"] == row["object_model"]
            and isinstance(row.get("source_model"), dict) and bool(row["source_model"])
            and isinstance(row.get("mapping"), dict) and bool(row["mapping"])
            and isinstance(row.get("association_mapping"), dict) and bool(row.get("tables"))
            and isinstance(row.get("ddl"), str) and bool(row["ddl"]),
            f"Native schema or mapping does not match the merged catalog: {key}",
        )
        native[key] = row
    require(set(native) == {k for k, v in candidates.items() if v["source"] == "ymap"}, "Native catalog coverage mismatch")
    return candidates, groups, native


def measurement_counts(root, candidates):
    counts = Counter()
    scenario_rows = Counter()
    seeds = set()
    for row in csv_rows(root / "all_measurements.csv"):
        key = row["schema_id"]
        candidate = candidates.get(key)
        require(candidate is not None and all(row[field] == candidate[field] for field in ("source", "object_model")),
                f"Unmatched measurement identity: {key}")
        scenario = SOURCES[row["source"]]
        require(tuple(row[k] for k in ("label_source", "workload_id", "environment_id")) == SCENARIOS[scenario],
                f"Unexpected measurement scenario: {key}")
        require(row["units_confirmed"].strip().casefold() == ("true" if scenario == "native" else "false"),
                f"Unexpected unit status: {key}")
        seed_key = (key, row["seed"])
        require(bool(row["seed"]) and seed_key not in seeds, f"Duplicate or blank measurement seed: {key}")
        seeds.add(seed_key)
        counts[key] += 1
        scenario_rows[scenario] += 1

    native_labels = set()
    for row in csv_rows(root / "Measurements/ymap_postgresql_measurements.csv"):
        key = row["schema_id"]
        require(key not in native_labels and key in candidates and candidates[key]["source"] == "ymap"
                and row["object_model"] == candidates[key]["object_model"]
                and tuple(row[k] for k in ("label_source", "workload_id", "environment_id")) == SCENARIOS["native"],
                f"Native measurement identity mismatch: {key}")
        native_labels.add(key)
    require(native_labels == {k for k, row in candidates.items() if row["source"] == "ymap"},
            "Native measurement coverage mismatch")
    require(set(candidates) == set(counts), "Some catalog IDs lack an existing measurement record")
    return counts, scenario_rows


def rank(row):
    material = "\0".join((SEED, row["source"], row["object_model"], row["schema_id"]))
    return sha(material.encode("utf-8")), row["schema_id"]


def choose(groups):
    archive_groups = {system: rows for (scenario, system), rows in groups.items() if scenario == "archive"}
    population = sum(map(len, archive_groups.values()))
    take = TOTAL - EXPECTED["native"][0]
    slots = {system: divmod(take * len(rows), population) for system, rows in archive_groups.items()}
    quotas = {system: whole for system, (whole, _) in slots.items()}
    remaining = take - sum(quotas.values())
    for system in sorted(slots, key=lambda s: (-slots[s][1], s))[:remaining]:
        quotas[system] += 1
    chosen = {}
    for (scenario, system), rows in sorted(groups.items(), key=lambda p: (p[0][0] != "native", p[0][1])):
        quota = len(rows) if scenario == "native" else quotas[system]
        require(0 < quota <= len(rows), f"Invalid quota for {system}")
        cohort = sorted(rows, key=rank)[:quota]
        chosen[scenario, system] = sorted(cohort, key=lambda r: r["schema_id"])
    require(sum(map(len, chosen.values())) == TOTAL, "Selection did not reach 12,500")
    return chosen


def sql_member(row):
    path = row["sql_path"]
    if row["source"] == "ymap":
        return "YMap_Research_Framework/Dataset/YMap/" + path
    system = ARCHIVE_NAMES[row["object_model"]]
    return ("YMap_Research_Framework/ReinforceORM_Old_Artifacts_Initial_Submission/All_SQL_Solutions/"
            + system + "/SQL_Solutions/" + PurePosixPath(path).name)


def build(root, out, bundle=None):
    candidates, groups, native = candidate_catalog(root)
    counts, scenario_rows = measurement_counts(root, candidates)
    for scenario, expected in EXPECTED.items():
        available = [rows for (s, _), rows in groups.items() if s == scenario]
        require((sum(map(len, available)), len(available), scenario_rows[scenario]) == expected,
                f"Unexpected source inventory for {scenario}")
    chosen = choose(groups)

    if out.exists():
        require(out.is_dir() and not any(out.iterdir()), "Use a new or empty output directory.")
    zf = ZipFile(bundle) if bundle else None
    try:
        entries = {entry.filename.replace("\\", "/"): entry for entry in zf.infolist()} if zf else {}
        for row in candidates.values():
            exists = sql_member(row) in entries if zf else (root / row["sql_path"]).is_file()
            require(exists, f"Missing candidate SQL path: {row['sql_path']}")
        results = []
        sql_hashes = {}
        by_system = []
        for (scenario, system), rows in chosen.items():
            is_native = scenario == "native"
            for row in rows:
                key = row["schema_id"]
                raw = zf.read(entries[sql_member(row)]) if zf else (root / row["sql_path"]).read_bytes()
                text = lf(raw)
                require(b"CREATE TABLE" in text.upper(), f"No CREATE TABLE found in {row['sql_path']}")
                if is_native:
                    require(text.rstrip(b"\n") == lf(native[key]["ddl"].encode("utf-8")).rstrip(b"\n"),
                            f"Native SQL content does not match generator record: {key}")
                digest = sha(text)
                require(digest not in sql_hashes, f"Duplicate LF-normalized SQL: {key} and {sql_hashes.get(digest)}")
                sql_hashes[digest] = key
                results.append(dict(
                    cohort_index=len(results) + 1, scenario=scenario, object_model=system, schema_id=key,
                    sql_path="Dataset/YMap/" + row["sql_path"], sql_sha256_lf=digest,
                    existing_measurement_rows=counts[key], existing_units_confirmed="yes" if is_native else "no",
                    mapping_status="recorded" if is_native else "required",
                    sql_dialect_status="postgresql_recorded" if is_native else "mysql_style_port_validation_required",
                    common_protocol_status="not_verified",
                ))
            by_system.append(dict(
                scenario=scenario, object_model=system, available_candidates=len(groups[scenario, system]),
                selected_candidates=len(rows), existing_measurement_rows=sum(counts[row["schema_id"]] for row in rows),
            ))
    finally:
        if zf:
            zf.close()

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "candidates.csv", FIELDS, results)
    write_csv(out / "source_systems.csv", ("scenario", "object_model", "available_candidates",
              "selected_candidates", "existing_measurement_rows"), by_system)
    manifest = {
        "scope": "candidate_inventory_for_new_study",
        "repository": "https://github.com/sazizian/YMap_Research_Framework",
        "dataset_commit": PINNED_COMMIT,
        "candidate_count": TOTAL,
        "catalog_source_system_id_count": len(by_system),
        "available_by_scenario": {scenario: dict(zip(("candidates", "source_system_ids", "measurement_rows"), values))
                                  for scenario, values in EXPECTED.items()},
        "selected_by_scenario": dict(Counter(row["scenario"] for row in results)),
        "selected_by_source_system": by_system,
        "selection": {
            "native": "retain all 728 candidate IDs",
            "archive": "allocate 11,772 slots proportionally by original archive system; largest remainder; lexicographic system ID breaks ties",
            "rank": "lowest SHA-256 of UTF-8 seed, raw source, object_model, schema_id separated by NUL; lexical schema_id breaks hash ties",
            "seed": SEED, "performance_outcomes_used": False,
        },
        "input_sha256_lf": {name: sha(lf((root / name).read_bytes())) for name in INPUT_FILES},
        "output_sha256": {name: sha((out / name).read_bytes()) for name in ("candidates.csv", "source_systems.csv")},
        "checks": {
            "catalog_ids_and_paths_unique": len(candidates),
            "catalog_candidates_with_a_measurement_record": len(counts),
            "catalog_sql_paths_resolved": len(candidates),
            "selected_unique_schema_ids": len({row["schema_id"] for row in results}),
            "selected_unique_lf_normalized_sql_hashes": len(sql_hashes),
            "native_sql_matches_recorded_ddl": len(native),
        },
        "provenance": {
            "native": "Y-Map October 2026 measured PostgreSQL study; exact scenario IDs recorded in config/datasets.yaml.",
            "archive": {
                "source": "ReinforceORM artifact as recorded in config/datasets.yaml",
                "citation": "Md Rashedul Hasan and Hamid Bagheri, Formal Verification-Integrated Reinforcement Learning for Automated ORM Schema Synthesis (ASE 2026)",
                "paper_doi": "10.1145/3832783.3837501", "artifact_doi": "10.6084/m9.figshare.30071260",
                "license": "CC BY-NC 4.0",
            },
        },
        "limits": [
            "A new inventory; neither the manuscript cohort nor a profiled or training-ready data set.",
            "Existing archive measurement units are unconfirmed. Counts describe available rows, not comparable outcomes.",
            "Archive graph and descriptor mappings must be recovered and its SQL ported and validated for PostgreSQL.",
            "Every selected design needs a verified measurement under the chosen common protocol before joint training or evaluation.",
            "Catalog source-system IDs do not establish independent logical systems; audit lineage before cross-system splitting.",
            "Distinct normalized SQL hashes establish distinct files only, not logical schema inequivalence.",
        ],
    }
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset/YMap or an equivalent verified catalog")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sql-bundle", type=Path, help="Optional v0.4 bundle of source SQL when the checked-out SQL tree is unavailable")
    args = parser.parse_args()
    report = build(args.dataset_root, args.out_dir, args.sql_bundle)
    print(json.dumps({key: report[key] for key in ("candidate_count", "selected_by_scenario", "checks")}, indent=2))
