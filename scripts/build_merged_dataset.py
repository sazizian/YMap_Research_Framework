r"""Build the merged Y-Map dataset (Y-Map candidates + ReinforceORM benchmark).

Run from the project root:
    .\.venv\Scripts\python.exe scripts/build_merged_dataset.py

Inputs (all under Dataset/YMap/, never modified):
  Generated/candidates.jsonl                      Y-Map generated candidates
  Measurements/ymap_postgresql_measurements.csv   Y-Map PostgreSQL measurements
  Measurements/rorm_supplied_measurements.csv     ReinforceORM supplied records
  All_SQL_Solutions/<system>/SQL_Solutions/*.sql  DDL of every schema

Outputs (Dataset/YMap/):
  catalog.csv          one row per SQL file (and per unmatched supplied record), with source and status
  all_candidates.jsonl one record per distinct measured schema, common DDL-derived features
  all_measurements.csv one row per measurement, Y-Map loader columns plus source

Common features are counted from the DDL text for both sources. link_table_count is
a heuristic (tables with at least two foreign keys whose primary-key columns are
all foreign-key columns); the other counts match the Y-Map generator exactly.
A ReinforceORM schema is one distinct DDL body within a system; SQL files with an
identical body are copies, and every supplied record that maps to that body is a
repeated measurement (seed 1..k). Original values are kept unchanged.

Every row keeps its source. Sources are separate scenarios (workload_id and
environment_id differ), so the evaluation code never pools them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RORM_SYSTEMS = {  # original ReinforceORM system name -> merged object_model id
    "Bank": "rorm_bank", "CSOS": "rorm_csos", "Camping": "rorm_camping", "Canteen": "rorm_canteen",
    "Customer - Order": "rorm_customer_order", "Decider": "rorm_decider", "E-Commerce": "rorm_ecommerce",
    "Flagship": "rorm_flagship", "Hospital": "rorm_hospital", "Library": "rorm_library",
    "Onlinestore": "rorm_onlinestore", "Order-Processing": "rorm_order_processing", "Store": "rorm_store",
    "Traffic-Control": "rorm_traffic_control", "University-Payroll": "rorm_university_payroll",
}
RORM_SCENARIO = {"label_source": "imported_measured", "workload_id": "reinforceorm_supplied", "environment_id": "reinforceorm_supplied_units_unconfirmed"}
COMMON_FEATURES = ["table_count", "column_count", "foreign_key_count", "pk_column_count", "nullable_column_count",
                   "unique_count", "check_count", "index_count", "trigger_count", "link_table_count",
                   "max_columns_per_table"]


# ---------------------------------------------------------------- DDL scanning
def _strip_comments(sql: str) -> str:
    out, i, n = [], 0, len(sql)
    while i < n:
        if sql.startswith("--", i):
            j = sql.find("\n", i); i = n if j < 0 else j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2); i = n if j < 0 else j + 2
        elif sql[i] in "'\"`":
            q = sql[i]; j = i + 1
            while j < n and not (sql[j] == q and sql[j - 1] != "\\"):
                j += 1
            out.append(sql[i:j + 1]); i = j + 1
        elif sql[i] == "$":
            m = re.match(r"\$[A-Za-z0-9_]*\$", sql[i:])
            if m:
                tag = m.group(0); j = sql.find(tag, i + len(tag)); j = n if j < 0 else j + len(tag)
                out.append(sql[i:j]); i = j
            else:
                out.append(sql[i]); i += 1
        else:
            out.append(sql[i]); i += 1
    return "".join(out)


def _statements(sql: str) -> list[str]:
    parts, depth, start, i, n = [], 0, 0, 0, len(sql)
    while i < n:
        c = sql[i]
        if c in "'\"`":
            j = i + 1
            while j < n and not (sql[j] == c and sql[j - 1] != "\\"):
                j += 1
            i = j + 1; continue
        if c == "$":
            m = re.match(r"\$[A-Za-z0-9_]*\$", sql[i:])
            if m:
                j = sql.find(m.group(0), i + len(m.group(0))); i = n if j < 0 else j + len(m.group(0)); continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ";" and depth == 0:
            parts.append(sql[start:i].strip()); start = i + 1
        i += 1
    if sql[start:].strip():
        parts.append(sql[start:].strip())
    return [p for p in parts if p]


def _split_top(body: str) -> list[str]:
    items, depth, start = [], 0, 0
    for i, c in enumerate(body):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            items.append(body[start:i].strip()); start = i + 1
    items.append(body[start:].strip())
    return [x for x in items if x]


def _ident(text: str) -> str:
    return text.strip().strip('`"').split(".")[-1].strip('`"').lower()


def _cols(text: str) -> list[str]:
    return [_ident(x) for x in text.split(",") if x.strip()]


def ddl_features(sql: str) -> dict:
    sql = _strip_comments(sql)
    tables: dict[str, dict] = {}
    fks, uniques, checks, indexes, triggers = [], 0, 0, 0, 0
    for stmt in _statements(sql):
        head = re.sub(r"\s+", " ", stmt[:200]).upper()
        if head.startswith("CREATE TABLE"):
            m = re.match(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)\s*\((.*)\)[^)]*$", stmt, re.S | re.I)
            if not m:
                raise ValueError(f"Unparsed CREATE TABLE: {stmt[:120]}")
            name = _ident(m.group(1)); t = {"columns": {}, "pk": set()}
            for item in _split_top(m.group(2)):
                up = item.upper()
                if up.startswith(("PRIMARY KEY", "CONSTRAINT")) or re.match(r"^(UNIQUE|FOREIGN|CHECK|KEY|INDEX)\b", up):
                    body = re.sub(r"^CONSTRAINT\s+\S+\s+", "", item, flags=re.I); bu = body.upper()
                    if bu.startswith("PRIMARY KEY"):
                        t["pk"].update(_cols(re.search(r"\((.*?)\)", body).group(1)))
                    elif bu.startswith("FOREIGN KEY"):
                        mm = re.search(r"FOREIGN\s+KEY\s*\((.*?)\)\s*REFERENCES\s+([^\s(]+)", body, re.I)
                        fks.append((name, tuple(_cols(mm.group(1))), _ident(mm.group(2))))
                    elif bu.startswith("UNIQUE"):
                        uniques += 1
                    elif bu.startswith("CHECK"):
                        checks += 1
                    elif re.match(r"^(KEY|INDEX)\b", bu):
                        indexes += 1
                    continue
                cm = re.match(r"([`\"]?[^\s`\"]+[`\"]?)\s+(.*)$", item, re.S)
                col, rest = _ident(cm.group(1)), cm.group(2).upper()
                t["columns"][col] = "NOT NULL" not in rest and "PRIMARY KEY" not in rest
                if "PRIMARY KEY" in rest:
                    t["pk"].add(col)
                if re.search(r"\bUNIQUE\b", rest):
                    uniques += 1
                checks += len(re.findall(r"\bCHECK\s*\(", rest))
                rm = re.search(r"\bREFERENCES\s+([^\s(]+)", rest)
                if rm:
                    fks.append((name, (col,), _ident(rm.group(1))))
            tables[name] = t
        elif head.startswith("ALTER TABLE"):
            name = _ident(re.match(r"ALTER\s+TABLE\s+(?:ONLY\s+)?([^\s]+)", stmt, re.I).group(1))
            for mm in re.finditer(r"FOREIGN\s+KEY\s*\((.*?)\)\s*REFERENCES\s+([^\s(]+)", stmt, re.I):
                fks.append((name, tuple(_cols(mm.group(1))), _ident(mm.group(2))))
            for mm in re.finditer(r"PRIMARY\s+KEY\s*\((.*?)\)", stmt, re.I):
                tables.setdefault(name, {"columns": {}, "pk": set()})["pk"].update(_cols(mm.group(1)))
            uniques += len(re.findall(r"ADD\s+(?:CONSTRAINT\s+\S+\s+)?UNIQUE\b", stmt, re.I))
            checks += len(re.findall(r"ADD\s+(?:CONSTRAINT\s+\S+\s+)?CHECK\b", stmt, re.I))
        elif re.match(r"CREATE (UNIQUE )?INDEX", head):
            if head.startswith("CREATE UNIQUE"):
                uniques += 1
            else:
                indexes += 1
        elif re.match(r"CREATE (OR REPLACE )?(CONSTRAINT )?TRIGGER", head):
            triggers += 1
    for name, t in tables.items():  # primary-key columns are never nullable
        for c in t["pk"]:
            if c in t["columns"]:
                t["columns"][c] = False
    fk_cols = defaultdict(set)
    for name, cols, _ in fks:
        fk_cols[name].update(cols)
    association = sum(1 for name, t in tables.items()
                      if t["pk"] and len([f for f in fks if f[0] == name]) >= 2 and t["pk"] <= fk_cols[name])
    return {"table_count": len(tables), "column_count": sum(len(t["columns"]) for t in tables.values()),
            "foreign_key_count": len(fks), "pk_column_count": sum(len(t["pk"]) for t in tables.values()),
            "nullable_column_count": sum(v for t in tables.values() for v in t["columns"].values()),
            "unique_count": uniques, "check_count": checks, "index_count": indexes, "trigger_count": triggers,
            "link_table_count": association,
            "max_columns_per_table": max((len(t["columns"]) for t in tables.values()), default=0)}


# ---------------------------------------------------------------- build
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ddl_body_sha256(text: str) -> str:
    """Hash of the schema itself: header (CREATE DATABASE FOR / USE), comments and whitespace removed."""
    text = re.sub(r"(?im)^\s*(--\s*)?CREATE DATABASE FOR[^\n]*\n", "", text)
    text = re.sub(r"(?im)^\s*USE\s+[^;]+;\s*\n", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", _strip_comments(text)).strip().encode()).hexdigest()


def _natural(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def build(root: Path) -> dict:
    data = root / "Dataset" / "YMap"
    meas_dir, sql_dir = data / "Measurements", data / "All_SQL_Solutions"
    catalog, candidates, measurements = [], [], []

    # Y-Map: generated candidates + PostgreSQL measurements
    ymap_meas = {r["schema_id"]: r for r in csv.DictReader(open(meas_dir / "ymap_postgresql_measurements.csv", encoding="utf-8"))}
    for line in open(data / "Generated" / "candidates.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        sid, model = rec["schema_id"], rec["object_model"]
        rel = f"All_SQL_Solutions/{model}/SQL_Solutions/{sid}.sql"
        feats = ddl_features(rec["ddl"])
        m = ymap_meas.get(sid)
        status = "included" if m else "no_measurement"
        catalog.append({"source": "ymap", "object_model": model, "schema_id": sid, "original_system": model,
                        "original_solution": sid, "sql_path": rel, "measurement_rows": int(m is not None),
                        "label_source": m["label_source"] if m else "", "environment_id": m["environment_id"] if m else "",
                        "units": "insert_ms, query_ms, storage_mb", "units_confirmed": True, "status": status})
        if m:
            candidates.append({"schema_id": sid, "object_model": model, "source": "ymap", "sql_path": rel,
                               "features": feats, "generator_features": rec["features"]})
            measurements.append({"schema_id": sid, "object_model": model, "source": "ymap",
                                 "insert_ms": m["insert_ms"], "query_ms": m["query_ms"], "storage_mb": m["storage_mb"],
                                 "label_source": m["label_source"], "workload_id": m["workload_id"],
                                 "environment_id": m["environment_id"], "seed": m["seed"], "units_confirmed": True,
                                 "original_query_time": "", "original_insert_time": "", "original_storage": ""})

    # ReinforceORM: one schema = one distinct DDL body within a system.
    # A supplied record names a database; it is matched to SQL files by the file's
    # own USE identifier, else by file name. All records that land on the same DDL
    # are repeated measurements of that schema (seeds 1..k).
    raw = list(csv.DictReader(open(meas_dir / "rorm_supplied_measurements.csv", encoding="utf-8-sig")))
    for system, model in RORM_SYSTEMS.items():
        files = []
        for path in sorted((sql_dir / model / "SQL_Solutions").glob("*.sql"), key=lambda p: _natural(p.stem)):
            text = path.read_text(encoding="utf-8", errors="replace")
            use = re.search(r"^\s*USE\s+`?([^`;\s]+)`?\s*;", text, re.M | re.I)
            files.append({"path": path, "text": text, "use": use.group(1).strip().lower() if use else "",
                          "stem": path.stem.strip().lower(), "ddl": ddl_body_sha256(text)})
        by_use, by_stem = defaultdict(set), defaultdict(set)
        for f in files:
            if f["use"]:
                by_use[f["use"]].add(f["ddl"])
            by_stem[f["stem"]].add(f["ddl"])
        rows_by_ddl, unmatched = defaultdict(list), []
        for r in (r for r in raw if r["subject_system"] == system):
            name = r["solution"].strip().lower()
            ddls = by_use.get(name) or by_stem.get(name) or set()
            if len(ddls) == 1:
                rows_by_ddl[next(iter(ddls))].append(r)
            else:
                unmatched.append((r, "measurement_without_sql" if not ddls else "measurement_matches_several_schemas"))
        first_file = {}
        for f in files:
            first_file.setdefault(f["ddl"], f)
        for f in files:
            rep = first_file[f["ddl"]]
            sid = f"{model}__{rep['path'].stem.strip()}"
            rows = rows_by_ddl.get(f["ddl"], [])
            if f is not rep:
                status = "duplicate_ddl_of_" + rep["path"].stem.strip()
            else:
                status = "included" if rows else "sql_without_measurement"
            catalog.append({"source": "reinforceorm", "object_model": model, "schema_id": sid, "original_system": system,
                            "original_solution": f["path"].stem, "sql_path": f["path"].relative_to(data).as_posix(),
                            "measurement_rows": len(rows), "label_source": RORM_SCENARIO["label_source"] if rows else "",
                            "environment_id": RORM_SCENARIO["environment_id"] if rows else "",
                            "units": "as supplied (query_time, insert_time, storage)", "units_confirmed": False, "status": status})
            if f is rep and rows:
                candidates.append({"schema_id": sid, "object_model": model, "source": "reinforceorm",
                                   "sql_path": f["path"].relative_to(data).as_posix(), "features": ddl_features(f["text"]),
                                   "supplied_solution_names": sorted({r["solution"].strip() for r in rows}, key=_natural)})
                for seed, r in enumerate(rows, 1):
                    measurements.append({"schema_id": sid, "object_model": model, "source": "reinforceorm",
                                         "insert_ms": r["insert_time"], "query_ms": r["query_time"], "storage_mb": r["storage"],
                                         **RORM_SCENARIO, "seed": seed, "units_confirmed": False,
                                         "original_query_time": r["query_time"], "original_insert_time": r["insert_time"],
                                         "original_storage": r["storage"]})
        for r, reason in unmatched:
            catalog.append({"source": "reinforceorm", "object_model": model, "schema_id": "", "original_system": system,
                            "original_solution": r["solution"], "sql_path": "", "measurement_rows": 1, "label_source": "",
                            "environment_id": "", "units": "", "units_confirmed": False, "status": reason})

    with open(data / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(catalog[0])); w.writeheader(); w.writerows(catalog)
    with open(data / "all_candidates.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for c in candidates:
            fh.write(json.dumps(c, sort_keys=True) + "\n")
    with open(data / "all_measurements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(measurements[0])); w.writeheader(); w.writerows(measurements)

    summary = {"inputs_sha256": {p: sha256(data / p) for p in ("Generated/candidates.jsonl",
               "Measurements/ymap_postgresql_measurements.csv", "Measurements/rorm_supplied_measurements.csv")},
               "catalog_status": {f"{s}:{st}": n for (s, st), n in sorted(Counter((c["source"], "duplicate_ddl" if c["status"].startswith("duplicate_ddl_of_") else c["status"]) for c in catalog).items())},
               "measurement_rows": dict(Counter(m["source"] for m in measurements)),
               "usable_schemas": dict(Counter(c["source"] for c in candidates)),
               "usable_schemas_per_object_model": dict(sorted(Counter(c["object_model"] for c in candidates).items())),
               "common_features": COMMON_FEATURES,
               "scenarios": {"ymap": sorted({(m["workload_id"], m["environment_id"]) for m in measurements if m["source"] == "ymap"}),
                             "reinforceorm": [[RORM_SCENARIO["workload_id"], RORM_SCENARIO["environment_id"]]]}}
    (data / "merged_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    print(json.dumps(build(parser.parse_args().project_root.resolve()), indent=2))
