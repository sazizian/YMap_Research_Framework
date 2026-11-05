"""Real PostgreSQL reference profiling, with no formula-derived timing labels."""
from __future__ import annotations

import csv
import datetime as dt
from decimal import Decimal
import hashlib
import json
import os
import platform
import random
import statistics
import time
import uuid
from pathlib import Path

from .synthesis import UnsupportedMappingError, _literal, _name, canonical_candidate, model_index, quote, validate_model


class ProfilingError(RuntimeError):
    pass


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_logical_fixture(model: dict, rows_per_class=100, seed=42) -> dict:
    """Create one explicit logical reference dataset, independent of mapping.

    These are generated input records, not previously measured research data.
    Cardinalities are exact-class counts; abstract classes never get own objects.
    """
    validate_model(model)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProfilingError("seed must be an integer.")
    idx = model_index(model)
    if isinstance(rows_per_class, bool) or not isinstance(rows_per_class, (int, dict)):
        raise ProfilingError("rows_per_class must be a positive integer or exact-concrete-class count dictionary.")
    if isinstance(rows_per_class, dict) and set(rows_per_class) != set(idx["concrete"]):
        raise ProfilingError("rows_per_class dictionary must specify every concrete class exactly once.")
    counts = {c: rows_per_class[c] if isinstance(rows_per_class, dict) else rows_per_class for c in idx["concrete"]}
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ProfilingError("Every concrete class count must be a positive integer.")
    rng = random.Random(seed)
    objects, next_id = [], 1
    for cls in idx["concrete"]:
        for _ in range(counts[cls]):
            attrs = {}
            for owner in idx["paths"][cls]:
                for attr in idx["classes"][owner].get("attributes", []):
                    col = _name("a_", owner + "_" + attr["name"])
                    kind = attr.get("type", "string").lower()
                    if attr.get("nullable", False) and rng.randrange(10) == 0:
                        value = None
                    elif kind in ("int", "integer"):
                        value = rng.randrange(1, 1_000_001)
                    elif kind in ("float", "double"):
                        value = round(rng.uniform(0.01, 10_000), 4)
                    elif kind == "decimal":
                        # String adaptation to NUMERIC preserves exact four-place
                        # decimal input, avoiding float-to-numeric roundoff.
                        value = f"{rng.uniform(0.01, 10_000):.4f}"
                    elif kind in ("bool", "boolean"):
                        value = bool(rng.randrange(2))
                    elif kind == "date":
                        value = (dt.date(2020, 1, 1) + dt.timedelta(days=rng.randrange(2000))).isoformat()
                    elif kind == "datetime":
                        value = (dt.datetime(2020, 1, 1) + dt.timedelta(seconds=rng.randrange(100_000_000))).isoformat()
                    else:
                        value = f"{owner}_{attr['name']}_{next_id}_{rng.randrange(1_000_000):06d}"
                    attrs[col] = value
            objects.append({"id": next_id, "class": cls, "attributes": attrs})
            next_id += 1
    links = {}
    for assoc in sorted(model.get("associations", []), key=lambda a: a["name"]):
        sources = [o["id"] for o in objects if assoc["source"] in idx["paths"][o["class"]]]
        targets = [o["id"] for o in objects if assoc["target"] in idx["paths"][o["class"]]]
        arng = random.Random(int(_digest([seed, assoc["name"]])[:16], 16))
        arng.shuffle(sources)
        mult = assoc.get("multiplicity", "one-to-many")
        if mult == "one-to-one" and not assoc.get("nullable", False) and len(targets) > len(sources):
            raise ProfilingError(f"Required one-to-one association {assoc['name']} has more target than source objects. Supply feasible per-class counts.")
        pairs = []
        for i, target in enumerate(targets):
            if mult != "many-to-many" and assoc.get("nullable", False) and i % 5 == 0:
                continue
            if mult == "one-to-one":
                if i < len(sources):
                    pairs.append([sources[i], target])
            elif mult == "many-to-many":
                for offset in range(min(2, len(sources))):
                    pairs.append([sources[(i + offset) % len(sources)], target])
            else:
                pairs.append([sources[i % len(sources)], target])
        links[assoc["name"]] = pairs
    return {"protocol": "ymap_reference_fixture_v1", "seed": seed, "rows_per_class": counts, "objects": objects, "links": links, "realization_id": _digest({"model": model, "objects": objects, "links": links})}


def validate_logical_fixture(model: dict, fixture: dict) -> None:
    """Validate loader invariants before any physical rows can overwrite one another."""
    validate_model(model)
    idx = model_index(model)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("objects"), list) or not isinstance(fixture.get("links"), dict):
        raise ProfilingError("Logical fixture requires objects and association links.")
    by_id = {}
    for obj in fixture["objects"]:
        if not isinstance(obj, dict) or isinstance(obj.get("id"), bool) or not isinstance(obj.get("id"), int) or obj["id"] <= 0:
            raise ProfilingError("Logical object IDs must be positive integers.")
        if obj["id"] in by_id:
            raise ProfilingError("Logical object IDs must be globally unique across concrete classes.")
        if not isinstance(obj.get("class"), str) or obj["class"] not in idx["concrete"]:
            raise ProfilingError("Logical fixture contains an unknown or abstract exact class.")
        expected = {_name("a_", owner + "_" + attr["name"]): attr for owner in idx["paths"][obj["class"]] for attr in idx["classes"][owner].get("attributes", [])}
        if not isinstance(obj.get("attributes"), dict) or set(obj["attributes"]) != set(expected):
            raise ProfilingError("Logical object attributes do not match its complete inheritance path.")
        for col, attr in expected.items():
            if obj["attributes"][col] is None and not attr.get("nullable", False):
                raise ProfilingError("Logical fixture omits a required attribute value.")
        by_id[obj["id"]] = obj
    assoc_names = {a["name"] for a in model.get("associations", [])}
    if set(fixture["links"]) != assoc_names:
        raise ProfilingError("Logical fixture links must match all source-model associations.")
    for assoc in model.get("associations", []):
        pairs = fixture["links"][assoc["name"]]
        if not isinstance(pairs, list):
            raise ProfilingError("Association links must be a list of source/target ID pairs.")
        seen, targets, sources = set(), {}, {}
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(isinstance(oid, bool) or not isinstance(oid, int) or oid not in by_id for oid in pair):
                raise ProfilingError("Association link references an invalid logical object ID.")
            source, target = pair
            if tuple(pair) in seen:
                raise ProfilingError("Duplicate logical association pair.")
            seen.add(tuple(pair))
            if assoc["source"] not in idx["paths"][by_id[source]["class"]] or assoc["target"] not in idx["paths"][by_id[target]["class"]]:
                raise ProfilingError("Association link violates its declared polymorphic endpoint type.")
            sources[source] = sources.get(source, 0) + 1
            targets[target] = targets.get(target, 0) + 1
        mult = assoc.get("multiplicity", "one-to-many")
        if mult != "many-to-many":
            if any(count > 1 for count in targets.values()):
                raise ProfilingError("Association links violate at-most-one source per target.")
            if mult == "one-to-one" and any(count > 1 for count in sources.values()):
                raise ProfilingError("One-to-one links violate source uniqueness.")
            if not assoc.get("nullable", False):
                required = {oid for oid, obj in by_id.items() if assoc["target"] in idx["paths"][obj["class"]]}
                if set(targets) != required:
                    raise ProfilingError("Association links violate required target participation.")
    realization = _digest({"model": model, "objects": fixture["objects"], "links": fixture["links"]})
    if fixture.get("realization_id") != realization:
        raise ProfilingError("Logical fixture realization_id does not match its source data.")


def build_load_plan(candidate: dict, fixture: dict) -> dict:
    """Map exactly the same logical objects/links to each supported schema."""
    model = candidate["source_model"]
    validate_logical_fixture(model, fixture)
    idx = model_index(model)
    rows = {t["name"]: {} for t in candidate["tables"]}
    by_id = {o["id"]: o for o in fixture["objects"]}
    for obj in fixture["objects"]:
        cls, oid = obj["class"], obj["id"]
        m = candidate["mapping"][cls]
        if m["strategy"] == "single_table":
            rows[m["table"]][oid] = {"id": oid, "dtype": cls, **obj["attributes"]}
        elif m["strategy"] == "joined":
            for owner in idx["paths"][cls]:
                attrs = {_name("a_", owner + "_" + a["name"]): obj["attributes"][_name("a_", owner + "_" + a["name"])] for a in idx["classes"][owner].get("attributes", [])}
                rows[candidate["mapping"][owner]["table"]][oid] = {"id": oid, "dtype": cls, **attrs}
        else:
            rows[m["table"]][oid] = {"id": oid, **obj["attributes"]}
    for assoc in model.get("associations", []):
        am = candidate["association_mapping"][assoc["name"]]
        for source_id, target_id in fixture["links"][assoc["name"]]:
            source_values = [source_id, by_id[source_id]["class"]][:len(am["source_columns"])]
            if am["strategy"] == "direct_fk":
                rows[am["table"]][target_id].update(zip(am["source_columns"], source_values))
            else:
                target_values = [target_id, by_id[target_id]["class"]][:len(am["target_columns"])]
                rows[am["table"]][(source_id, target_id)] = {**dict(zip(am["source_columns"], source_values)), **dict(zip(am["target_columns"], target_values))}
    plan = {}
    for table in candidate["tables"]:
        columns = [c["name"] for c in table["columns"]]
        data = [tuple(row.get(c) for c in columns) for _, row in sorted(rows[table["name"]].items())]
        plan[table["name"]] = {"columns": columns, "rows": data, "sql": f"INSERT INTO {quote(table['name'])} (" + ", ".join(map(quote, columns)) + ") VALUES (" + ", ".join("%s" for _ in columns) + ")"}
    return plan


def _class_select(candidate: dict, cls: str) -> str:
    """Polymorphic logical class projection, including inherited attributes."""
    idx = model_index(candidate["source_model"])
    columns = [(owner, attr) for owner in idx["paths"][cls] for attr in idx["classes"][owner].get("attributes", [])]
    branches = []
    for concrete in idx["descendants"][cls]:
        if concrete not in idx["concrete"]:
            continue
        m = candidate["mapping"][concrete]
        if m["strategy"] == "joined":
            path = idx["paths"][concrete]
            aliases = {owner: "p" + str(i) for i, owner in enumerate(path)}
            root_alias = aliases[path[0]]
            selected = [root_alias + '."id" AS "id"', root_alias + '."dtype" AS "dtype"']
            selected += [aliases[owner] + "." + quote(_name("a_", owner + "_" + attr["name"])) + " AS " + quote(_name("a_", owner + "_" + attr["name"])) for owner, attr in columns]
            from_sql = quote(candidate["mapping"][path[0]]["table"]) + " " + root_alias
            for owner in path[1:]:
                alias = aliases[owner]
                from_sql += " JOIN " + quote(candidate["mapping"][owner]["table"]) + " " + alias + " ON " + alias + '."id"=' + root_alias + '."id"'
            branch = "SELECT " + ", ".join(selected) + " FROM " + from_sql + " WHERE " + root_alias + '."dtype"=' + _literal(concrete)
        else:
            selected = ['p."id" AS "id"', ("p.\"dtype\"" if m["strategy"] == "single_table" else _literal(concrete)) + ' AS "dtype"']
            selected += ["p." + quote(_name("a_", owner + "_" + attr["name"])) + " AS " + quote(_name("a_", owner + "_" + attr["name"])) for owner, attr in columns]
            branch = "SELECT " + ", ".join(selected) + " FROM " + quote(m["table"]) + " p"
            if m["strategy"] == "single_table":
                branch += ' WHERE p."dtype"=' + _literal(concrete)
        branches.append(branch)
    return " UNION ALL ".join("(" + branch + ")" for branch in branches)


def build_query_plan(candidate: dict, query_limit=100, fixture: dict | None = None) -> list[dict]:
    """Shared logical workload template: class lookup/scan + association joins.

    Every query consumes all result rows. IDs and source data are identical for
    every mapping of a model; generated physical SQL legitimately differs.
    """
    if isinstance(query_limit, bool) or not isinstance(query_limit, int) or query_limit <= 0:
        raise ProfilingError("query_limit must be a positive integer.")
    idx = model_index(candidate["source_model"])
    queries = []
    for cls in sorted(idx["classes"]):
        relation = _class_select(candidate, cls)
        ids = [o["id"] for o in fixture["objects"] if cls in idx["paths"][o["class"]]] if fixture else [1]
        queries.append({"name": "lookup:" + cls, "sql": "SELECT * FROM (" + relation + ') q WHERE q."id"=%s', "params": [min(ids)]})
        queries.append({"name": "scan:" + cls, "sql": "SELECT * FROM (" + relation + ') q ORDER BY q."id" LIMIT %s', "params": [query_limit]})
    for assoc in sorted(candidate["source_model"].get("associations", []), key=lambda a: a["name"]):
        am = candidate["association_mapping"][assoc["name"]]
        target_id = am["target_columns"][0]
        sql = "SELECT s.*, t.* FROM (" + _class_select(candidate, assoc["source"]) + ") s JOIN " + quote(am["table"]) + " l ON l." + quote(am["source_columns"][0]) + '=s."id" JOIN (' + _class_select(candidate, assoc["target"]) + ") t ON l." + quote(target_id) + '=t."id" ORDER BY s."id", t."id" LIMIT %s'
        queries.append({"name": "association:" + assoc["name"], "sql": sql, "params": [query_limit]})
    return queries


def expected_query_results(candidate: dict, fixture: dict, query_limit=100) -> dict:
    """Independent logical oracle for the exact reference-query template."""
    model = candidate["source_model"]
    validate_logical_fixture(model, fixture)
    idx = model_index(model)
    by_id = {o["id"]: o for o in fixture["objects"]}

    def project(obj, static_class):
        values = [obj["id"], obj["class"]]
        for owner in idx["paths"][static_class]:
            for attr in idx["classes"][owner].get("attributes", []):
                value = obj["attributes"][_name("a_", owner + "_" + attr["name"])]
                kind = attr.get("type", "string").lower()
                if value is not None:
                    if kind == "date":
                        value = dt.date.fromisoformat(value)
                    elif kind == "datetime":
                        value = dt.datetime.fromisoformat(value)
                    elif kind == "decimal":
                        value = Decimal(value)
                values.append(value)
        return tuple(values)

    results = {}
    for cls in sorted(idx["classes"]):
        objects = sorted((o for o in fixture["objects"] if cls in idx["paths"][o["class"]]), key=lambda o: o["id"])
        if not objects:
            raise ProfilingError("The reference workload requires at least one object for every static class.")
        lookup = [project(objects[0], cls)]
        scan = [project(o, cls) for o in objects[:query_limit]]
        results["lookup:" + cls] = {"rows": 1, "sha256": _digest(lookup)}
        results["scan:" + cls] = {"rows": len(scan), "sha256": _digest(scan)}
    for assoc in model.get("associations", []):
        pairs = sorted(fixture["links"][assoc["name"]])[:query_limit]
        rows = [project(by_id[s], assoc["source"]) + project(by_id[t], assoc["target"]) for s, t in pairs]
        results["association:" + assoc["name"]] = {"rows": len(rows), "sha256": _digest(rows)}
    return results


def validate_postgresql_syntax(candidate: dict, fixture: dict | None = None, query_limit=100) -> dict:
    """Optionally parse generated SQL/PLpgSQL without claiming live DB validation."""
    try:
        import pglast
    except ImportError as exc:
        raise ProfilingError("Install optional pglast for static PostgreSQL syntax validation; it does not execute a database.") from exc
    candidate = canonical_candidate(candidate)
    statements = pglast.split(candidate["ddl"])
    for statement in statements:
        pglast.parse_sql(statement)
        if statement.lstrip().startswith("CREATE FUNCTION"):
            pglast.parse_plpgsql(statement)
    queries = build_query_plan(candidate, query_limit, fixture)
    for query in queries:
        # Only compiler-generated positional placeholders are replaced. No
        # input values are interpolated into SQL during profiling itself.
        sql = query["sql"]
        for position in range(1, len(query["params"]) + 1):
            sql = sql.replace("%s", "$" + str(position), 1)
        pglast.parse_sql(sql)
    if fixture is not None:
        for plan in build_load_plan(candidate, fixture).values():
            sql = plan["sql"]
            for position in range(1, len(plan["columns"]) + 1):
                sql = sql.replace("%s", "$" + str(position), 1)
            pglast.parse_sql(sql)
    return {"validation": "static_postgresql_and_plpgsql_parse_only", "parser_version": pglast.__version__, "ddl_statements": len(statements), "queries": len(queries), "database_execution": False}


def _environment(conn, config: dict) -> dict:
    import psycopg
    settings = ["server_version", "shared_buffers", "work_mem", "maintenance_work_mem", "effective_cache_size", "random_page_cost", "seq_page_cost", "default_statistics_target", "jit", "TimeZone", "server_encoding", "max_parallel_workers_per_gather", "synchronous_commit"]
    with conn.cursor() as cur:
        cur.execute("SELECT version(), inet_server_addr()::text, inet_server_port()")
        version, address, port = cur.fetchone()
        current = {}
        for setting in settings:
            cur.execute("SELECT current_setting(%s)", (setting,))
            current[setting] = cur.fetchone()[0]
    runtime_class = "postgresql_wasm" if "pglite" in version.lower() or "wasm" in version.lower() else "postgresql_server"
    return {"postgresql_version": version, "runtime_class": runtime_class, "server_address": address, "server_port": port, "settings": current, "client": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version(), "cpu_count": os.cpu_count()}, "environment_notes": config.get("environment_notes", "Server hardware/storage/concurrent load are not automatically characterized."), "client_protocol": {"driver": "psycopg3", "driver_version": psycopg.__version__, "prepared_statements": "disabled", "batch_size": config.get("batch_size", 1000), "statement_timeout_ms": config.get("statement_timeout_ms", 30000)}}


def profile_candidates(candidates, config: dict, out_dir: Path) -> Path:
    """Measure actual PostgreSQL operations in fresh UUID-named schemas.

    insert_ms: transaction wall time for the full logical fixture, including
    client batching, network, indexes, constraints, and COMMIT; excludes DDL,
    fixture creation, TRUNCATE, ANALYZE, and candidate generation.
    query_ms: mean wall time per logical SQL query, including result fetch,
    measured against the freshly loaded/analyzed warm-cache fixture.
    storage_mb: sum pg_total_relation_size of candidate base tables, including
    their indexes/TOAST, divided by 1,000,000 (decimal MB). Raw bytes retained.
    Summary rows are medians across measured repetitions. Raw times retained.
    """
    if config.get("fixture_mode") != "deterministic_generated":
        raise ProfilingError("Set fixture_mode: deterministic_generated to explicitly authorize generated reference input data. No old experimental data or timing labels are generated implicitly.")
    dsn_env = config.get("dsn_env", "YMAP_POSTGRES_DSN")
    if not isinstance(dsn_env, str) or not dsn_env:
        raise ProfilingError("dsn_env must name an environment variable.")
    dsn = os.environ.get(dsn_env)
    if not dsn:
        raise ProfilingError(f"PostgreSQL connection variable {dsn_env} is unset; no synthetic measurements produced.")
    for key, default, minimum in (("repetitions", 5, 1), ("warmup", 1, 0), ("batch_size", 1000, 1), ("statement_timeout_ms", 30000, 1)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ProfilingError(f"{key} must be an integer >= {minimum}.")
    try:
        import psycopg
    except ImportError as exc:
        raise ProfilingError("Install the optional psycopg[binary]>=3.1 dependency for real PostgreSQL profiling.") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "measurements.csv"
    if summary.exists() or (out / "raw_measurements.jsonl").exists():
        raise ProfilingError("Profiling output already exists; use a fresh output directory to preserve prior evidence.")
    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
        conn.prepare_threshold = None
    except Exception:
        raise ProfilingError(f"Could not connect to PostgreSQL using {dsn_env}. Check the server and credentials. No synthetic measurements produced.") from None
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        conn.close()
        raise ProfilingError("seed must be an integer.")
    protocol = {"version": "ymap_logical_workload_v1", "fixture": "ymap_reference_fixture_v1", "rows_per_class": config.get("rows_per_class", 100), "query_limit": config.get("query_limit", 100), "queries": "one point lookup and one ordered limited scan per logical class; one ordered limited join per association", "null_probability": 0.1, "link_policy": "optional target skips every fifth object; many-to-many two sources per target", "cache_policy": "no forced cache eviction; repeated freshly loaded data; ANALYZE before queries", "prepared_statements": "disabled"}
    workload_id = "workload_" + _digest(protocol)[:16]
    seen, fixture_cache, expected = set(), {}, {}
    repetitions, warmup = config.get("repetitions", 5), config.get("warmup", 1)
    fields = ["schema_id", "object_model", "insert_ms", "query_ms", "storage_mb", "label_source", "workload_id", "environment_id", "seed", "realization_id", "repetitions", "query_count"]
    count = 0
    try:
        env = _environment(conn, config)
        environment_id = "environment_" + _digest(env)[:16]
        (out / "environment.json").write_text(json.dumps({"environment_id": environment_id, **env}, indent=2), encoding="utf-8")
        (out / "workload.json").write_text(json.dumps({"workload_id": workload_id, "seed": seed, **protocol, "measurement_boundaries": profile_candidates.__doc__}, indent=2), encoding="utf-8")
        with (out / "measurements.csv.partial").open("w", newline="", encoding="utf-8") as summary_file, (out / "raw_measurements.jsonl").open("w", encoding="utf-8") as raw:
            writer = csv.DictWriter(summary_file, fieldnames=fields)
            writer.writeheader()
            for candidate in candidates:
                try:
                    candidate = canonical_candidate(candidate)
                except UnsupportedMappingError as exc:
                    raise ProfilingError(str(exc)) from exc
                sid = candidate["schema_id"]
                if sid in seen:
                    raise ProfilingError(f"Duplicate schema_id in profiling input: {sid}.")
                seen.add(sid)
                model_key = _digest(candidate["source_model"])
                if model_key not in fixture_cache:
                    fixture_cache[model_key] = build_logical_fixture(candidate["source_model"], config.get("rows_per_class", 100), seed)
                fixture = fixture_cache[model_key]
                load_plan = build_load_plan(candidate, fixture)
                queries = build_query_plan(candidate, config.get("query_limit", 100), fixture)
                expected[model_key] = expected_query_results(candidate, fixture, config.get("query_limit", 100))
                candidate_dir = out / "candidates" / sid
                candidate_dir.mkdir(parents=True, exist_ok=False)
                (candidate_dir / "candidate.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")
                (candidate_dir / "queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
                (candidate_dir / "expected_query_results.json").write_text(json.dumps(expected[model_key], indent=2), encoding="utf-8")
                (candidate_dir / "fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
                (candidate_dir / "schema.sql").write_text(candidate["ddl"] + "\n", encoding="utf-8")
                schema_name = "ymap_" + uuid.uuid4().hex
                created = False
                results = []
                lifecycle = {"owned_schema": schema_name, "created_by_this_run": False, "cleanup_status": "not_created"}
                lifecycle_file = candidate_dir / "schema_lifecycle.json"
                lifecycle_file.write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
                try:
                    with conn.cursor() as cur:
                        cur.execute("CREATE SCHEMA " + quote(schema_name))
                        created = True
                        lifecycle.update({"created_by_this_run": True, "cleanup_status": "pending"})
                        lifecycle_file.write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
                        cur.execute("SET search_path TO " + quote(schema_name) + ", pg_catalog")
                        cur.execute("SET statement_timeout TO " + str(config.get("statement_timeout_ms", 30000)))
                        cur.execute(candidate["ddl"], prepare=False)
                    for repeat in range(-warmup, repetitions):
                        with conn.cursor() as cur:
                            cur.execute("TRUNCATE " + ", ".join(quote(t["name"]) for t in candidate["tables"]))
                        start = time.perf_counter_ns()
                        with conn.transaction():
                            with conn.cursor() as cur:
                                for table_plan in load_plan.values():
                                    data = table_plan["rows"]
                                    batch = config.get("batch_size", 1000)
                                    for offset in range(0, len(data), batch):
                                        # executemany may prepare statements in psycopg3;
                                        # disable auto-prepare on the connection globally.
                                        cur.executemany(table_plan["sql"], data[offset:offset + batch])
                        insert_ms = (time.perf_counter_ns() - start) / 1_000_000
                        with conn.cursor() as cur:
                            for table in candidate["tables"]:
                                cur.execute("ANALYZE " + quote(table["name"]))
                            query_times, signatures = [], {}
                            for query in queries:
                                start = time.perf_counter_ns()
                                cur.execute(query["sql"], query["params"], prepare=False)
                                fetched = cur.fetchall()
                                query_times.append((time.perf_counter_ns() - start) / 1_000_000)
                                signatures[query["name"]] = {"rows": len(fetched), "sha256": _digest(fetched)}
                            cur.execute("SELECT COALESCE(SUM(pg_total_relation_size(c.oid)),0) FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid WHERE n.nspname=%s AND c.relkind='r'", (schema_name,), prepare=False)
                            storage_bytes = int(cur.fetchone()[0])
                            storage_mb = storage_bytes / 1_000_000
                        if signatures != expected[model_key]:
                            (candidate_dir / "query_result_mismatch.json").write_text(json.dumps({"expected": expected[model_key], "observed": signatures}, indent=2), encoding="utf-8")
                            raise ProfilingError(f"Query result mismatch against the independent logical fixture oracle for {sid}; candidate measurements are rejected.")
                        record = {"schema_id": sid, "object_model": candidate["object_model"], "seed": seed, "repeat": repeat, "warmup": repeat < 0, "insert_ms": insert_ms, "query_ms": statistics.mean(query_times), "storage_mb": storage_mb, "storage_bytes": storage_bytes, "query_times_ms": dict(zip((q["name"] for q in queries), query_times)), "query_results": signatures, "realization_id": fixture["realization_id"], "workload_id": workload_id, "environment_id": environment_id, "label_source": "postgresql_measured"}
                        raw.write(json.dumps(record, sort_keys=True) + "\n")
                        raw.flush()
                        if repeat >= 0:
                            results.append(record)
                    row = {field: results[0].get(field) for field in fields}
                    row.update({metric: statistics.median(r[metric] for r in results) for metric in ("insert_ms", "query_ms", "storage_mb")})
                    row.update({"repetitions": repetitions, "query_count": len(queries)})
                    writer.writerow(row)
                    summary_file.flush()
                    count += 1
                finally:
                    if created:
                        try:
                            with conn.cursor() as cur:
                                # Only the UUID schema successfully created by this call
                                # is dropped. Existing/user schemas are never accepted.
                                cur.execute("SET search_path TO pg_catalog")
                                cur.execute("DROP SCHEMA " + quote(schema_name) + " CASCADE")
                            lifecycle["cleanup_status"] = "dropped"
                        except Exception as cleanup_error:
                            lifecycle.update({"cleanup_status": "failed", "cleanup_error_type": type(cleanup_error).__name__})
                            raise ProfilingError(f"Could not clean up the dedicated schema {schema_name}; inspect its recorded lifecycle before manually removing only that owned schema.") from None
                        finally:
                            lifecycle_file.write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        if count == 0:
            raise ProfilingError("No candidates supplied; no measurements produced.")
        (out / "measurements.csv.partial").replace(summary)
        return summary
    except ProfilingError:
        raise
    except Exception as exc:
        # Do not print connection strings, credentials, or raw driver errors.
        diagnostic = getattr(getattr(exc, "diag", None), "message_primary", None)
        (out / "failure.json").write_text(json.dumps({"error_type": type(exc).__name__, "sqlstate": getattr(exc, "sqlstate", None), "database_message": diagnostic, "completed_candidates": count, "message": "PostgreSQL profiling did not complete; partial evidence retained."}, indent=2), encoding="utf-8")
        raise ProfilingError(f"PostgreSQL profiling failed ({type(exc).__name__}); partial evidence is in {out}. No synthetic fallback was used.") from None
    finally:
        conn.close()
