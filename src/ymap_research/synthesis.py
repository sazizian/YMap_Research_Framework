"""Finite, streaming schema synthesis with explicit supported mapping semantics.

The Python backend enumerates mapping decisions. The optional Alloy backend
solves the same finite decision constraints with the real Alloy Java API. Neither
backend is a proof of full object/relational semantic equivalence.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator


class ModelValidationError(ValueError):
    pass


class UnsupportedMappingError(ValueError):
    pass


class AlloyUnavailableError(RuntimeError):
    pass


SQL_TYPES = {"int": "BIGINT", "integer": "BIGINT", "string": "TEXT", "text": "TEXT", "float": "DOUBLE PRECISION", "double": "DOUBLE PRECISION", "decimal": "NUMERIC", "date": "DATE", "datetime": "TIMESTAMP", "bool": "BOOLEAN", "boolean": "BOOLEAN"}
STRATEGIES = ("single_table", "joined", "table_per_class")
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,47}$")


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _name(prefix: str, value: str) -> str:
    name = prefix + value
    return name if len(name) <= 60 else name[:47] + "_" + hashlib.sha256(name.encode()).hexdigest()[:12]


def model_index(model: dict) -> dict:
    classes = {c["name"]: c for c in model["classes"]}
    paths = {}
    for name in classes:
        path, current = [], name
        while current:
            path.append(current)
            current = classes[current].get("extends")
        paths[name] = list(reversed(path))
    descendants = {name: sorted(c for c in classes if name in paths[c]) for name in classes}
    return {"classes": classes, "paths": paths, "descendants": descendants, "roots": sorted(c for c in classes if not classes[c].get("extends")), "concrete": sorted(c for c in classes if not classes[c].get("abstract", False))}


def validate_model(model: dict) -> None:
    """Reject malformed/unsupported input instead of silently inventing semantics."""
    if not isinstance(model, dict):
        raise ModelValidationError("Object model must be a JSON object.")
    if set(model) - {"name", "classes", "associations", "domain", "metadata"}:
        raise ModelValidationError("Unsupported model fields; express annotations under metadata instead of silently ignored semantics.")
    if not isinstance(model.get("name"), str) or not IDENTIFIER.fullmatch(model["name"]):
        raise ModelValidationError("Model name must match [A-Za-z][A-Za-z0-9_]{0,47}.")
    if not isinstance(model.get("classes"), list) or not model["classes"]:
        raise ModelValidationError("classes must be a nonempty list.")
    names = []
    for cls in model["classes"]:
        if not isinstance(cls, dict) or not isinstance(cls.get("name"), str) or not IDENTIFIER.fullmatch(cls["name"]):
            raise ModelValidationError("Every class requires a valid identifier name.")
        if set(cls) - {"name", "extends", "abstract", "attributes", "metadata"}:
            raise ModelValidationError(f"Unsupported class fields in {cls['name']}: {sorted(set(cls) - {'name', 'extends', 'abstract', 'attributes', 'metadata'})}.")
        names.append(cls["name"])
        if "abstract" in cls and not isinstance(cls["abstract"], bool):
            raise ModelValidationError("Class abstract must be boolean.")
        if not isinstance(cls.get("attributes", []), list):
            raise ModelValidationError("Class attributes must be a list.")
        attrs = []
        for attr in cls.get("attributes", []):
            if not isinstance(attr, dict) or not isinstance(attr.get("name"), str) or not IDENTIFIER.fullmatch(attr["name"]):
                raise ModelValidationError("Every attribute requires a valid identifier name.")
            if set(attr) - {"name", "type", "nullable", "metadata"}:
                raise ModelValidationError(f"Unsupported attribute fields in {cls['name']}.{attr['name']}.")
            attrs.append(attr["name"])
            if not isinstance(attr.get("type", "string"), str) or attr.get("type", "string").lower() not in SQL_TYPES:
                raise ModelValidationError(f"Unsupported attribute type: {attr.get('type')!r}.")
            if "nullable" in attr and not isinstance(attr["nullable"], bool):
                raise ModelValidationError("Attribute nullable must be boolean.")
        if len(attrs) != len(set(attrs)):
            raise ModelValidationError(f"Duplicate attributes in class {cls['name']}.")
    if len(names) != len(set(names)):
        raise ModelValidationError("Duplicate class names.")
    classes = {c["name"]: c for c in model["classes"]}
    for cls in model["classes"]:
        parent = cls.get("extends")
        if parent is not None and not isinstance(parent, str):
            raise ModelValidationError("extends must name one parent class as a string.")
        if parent is not None and parent not in classes:
            raise ModelValidationError(f"Unknown parent {parent!r}.")
        seen, current = set(), cls["name"]
        while current:
            if current in seen:
                raise ModelValidationError("Inheritance must be acyclic and single-parent.")
            seen.add(current)
            current = classes[current].get("extends")
    idx = model_index(model)
    if not idx["concrete"]:
        raise ModelValidationError("At least one concrete class is required.")
    for root in idx["roots"]:
        physical = [_name("a_", c + "_" + a["name"]) for c in idx["descendants"][root] for a in classes[c].get("attributes", [])]
        if len(physical) != len(set(physical)):
            raise ModelValidationError(f"Attribute names in hierarchy {root} collide after physical-name encoding; rename an attribute or class.")
    for name, path in idx["paths"].items():
        inherited = [a["name"] for c in path for a in classes[c].get("attributes", [])]
        if len(inherited) != len(set(inherited)):
            raise ModelValidationError(f"Attribute overriding/shadowing is unsupported in {name}.")
        if not any(c in idx["concrete"] for c in idx["descendants"][name]):
            raise ModelValidationError(f"Class {name} has no concrete realization.")
    associations = model.get("associations", [])
    if not isinstance(associations, list):
        raise ModelValidationError("associations must be a list.")
    assoc_names = []
    for assoc in associations:
        if not isinstance(assoc, dict) or not isinstance(assoc.get("name"), str) or not IDENTIFIER.fullmatch(assoc["name"]):
            raise ModelValidationError("Every association requires a valid identifier name.")
        if set(assoc) - {"name", "source", "target", "multiplicity", "nullable", "metadata"}:
            raise ModelValidationError(f"Unsupported association fields in {assoc['name']}.")
        assoc_names.append(assoc["name"])
        if not isinstance(assoc.get("source"), str) or not isinstance(assoc.get("target"), str) or assoc["source"] not in classes or assoc["target"] not in classes:
            raise ModelValidationError(f"Association {assoc['name']} references an unknown class.")
        if assoc.get("multiplicity", "one-to-many") not in ("one-to-one", "one-to-many", "many-to-many"):
            raise ModelValidationError("Supported multiplicities: one-to-one, one-to-many, many-to-many.")
        if "nullable" in assoc and not isinstance(assoc["nullable"], bool):
            raise ModelValidationError("Association nullable must be boolean.")
        if assoc.get("multiplicity") == "many-to-many" and "nullable" in assoc:
            raise ModelValidationError("many-to-many has zero-to-many participation at both ends; omit nullable.")
    if len(assoc_names) != len(set(assoc_names)):
        raise ModelValidationError("Duplicate association names.")


def decision_space(model: dict) -> tuple[list[tuple[str, list[str]]], list[str]]:
    idx = model_index(model)
    dimensions, exclusions = [], []
    for root in idx["roots"]:
        if len(idx["descendants"][root]) == 1:
            continue
        choices = list(STRATEGIES)
        poly = [a["name"] for a in model.get("associations", []) if any(idx["paths"][a[end]][0] == root and len(idx["descendants"][a[end]]) > 1 for end in ("source", "target"))]
        if poly:
            choices.remove("table_per_class")
            exclusions.append(f"table_per_class excluded for hierarchy {root}: polymorphic association endpoints in {', '.join(poly)} require cross-table referential constraints not implemented here.")
        dimensions.append((f"inheritance:{root}", choices))
    for assoc in sorted(model.get("associations", []), key=lambda a: a["name"]):
        choices = ["association_table"] if assoc.get("multiplicity", "one-to-many") == "many-to-many" else ["direct_fk", "association_table"]
        dimensions.append((f"association:{assoc['name']}", choices))
    return dimensions, exclusions


def _column(name: str, data_type: str, nullable=False, primary_key=False) -> dict:
    return {"name": name, "type": data_type, "nullable": bool(nullable), "primary_key": bool(primary_key)}


def _build_candidate(model: dict, decisions: dict, backend: str, exclusions: list[str]) -> dict:
    idx = model_index(model)
    tables, mapping, association_mapping = {}, {}, {}

    def table(name, kind):
        t = {"name": name, "kind": kind, "columns": [], "foreign_keys": [], "unique": [], "checks": []}
        tables[name] = t
        return t

    def attrs(cls):
        return [(c, a) for c in idx["paths"][cls] for a in idx["classes"][c].get("attributes", [])]

    def add_attrs(t, pairs, force_nullable=False):
        for owner, attr in pairs:
            t["columns"].append(_column(_name("a_", owner + "_" + attr["name"]), SQL_TYPES[attr.get("type", "string").lower()], force_nullable or attr.get("nullable", False)))

    def fk(t, columns, ref_table, ref_columns):
        t["foreign_keys"].append({"columns": columns, "references": {"table": ref_table, "columns": ref_columns}})
        for col, ref in zip(columns, ref_columns):
            next(c for c in t["columns"] if c["name"] == col)["foreign_key"] = ref_table + "." + ref

    for root in idx["roots"]:
        family = idx["descendants"][root]
        strategy = decisions.get("inheritance:" + root, "table_per_class")
        if strategy == "single_table":
            t = table(_name("c_", root), "entity")
            t["columns"] = [_column("id", "BIGINT", primary_key=True), _column("dtype", "TEXT")]
            t["unique"].append(["id", "dtype"])
            concrete = [c for c in family if c in idx["concrete"]]
            t["checks"].append("dtype IN (" + ",".join(map(_literal, concrete)) + ")")
            for cls in family:
                add_attrs(t, [(cls, a) for a in idx["classes"][cls].get("attributes", [])], True)
                applicable = [c for c in idx["descendants"][cls] if c in idx["concrete"]]
                condition = "dtype IN (" + ",".join(map(_literal, applicable)) + ")"
                for attr in idx["classes"][cls].get("attributes", []):
                    col = quote(_name("a_", cls + "_" + attr["name"]))
                    t["checks"].append(f"({condition}) OR {col} IS NULL")
                    if not attr.get("nullable", False):
                        t["checks"].append(f"NOT ({condition}) OR {col} IS NOT NULL")
                mapping[cls] = {"table": t["name"], "strategy": strategy, "root": root, "path": idx["paths"][cls]}
        elif strategy == "joined":
            for cls in sorted(family, key=lambda c: (len(idx["paths"][c]), c)):
                t = table(_name("c_", cls), "entity" if cls == root else "inheritance")
                t["columns"] = [_column("id", "BIGINT", primary_key=True), _column("dtype", "TEXT")]
                add_attrs(t, [(cls, a) for a in idx["classes"][cls].get("attributes", [])])
                t["unique"].append(["id", "dtype"])
                concrete = [c for c in idx["descendants"][cls] if c in idx["concrete"]]
                t["checks"].append("dtype IN (" + ",".join(map(_literal, concrete)) + ")")
                parent = idx["classes"][cls].get("extends")
                if parent:
                    fk(t, ["id", "dtype"], _name("c_", parent), ["id", "dtype"])
                mapping[cls] = {"table": t["name"], "strategy": strategy, "root": root, "path": idx["paths"][cls]}
        elif strategy == "table_per_class":
            for cls in family:
                mapping[cls] = {"table": _name("c_", cls) if cls in idx["concrete"] else None, "strategy": strategy, "root": root, "path": idx["paths"][cls]}
                if cls in idx["concrete"]:
                    t = table(_name("c_", cls), "entity" if cls == root else "inheritance")
                    t["columns"] = [_column("id", "BIGINT", primary_key=True)]
                    add_attrs(t, attrs(cls))
        else:
            raise UnsupportedMappingError(f"Unsupported inheritance strategy: {strategy}.")

    triggers = []

    def endpoint_fk(t, endpoint, prefix, nullable):
        m = mapping[endpoint]
        if m["strategy"] == "table_per_class" and len(idx["descendants"][endpoint]) > 1:
            raise UnsupportedMappingError("TPC polymorphic association endpoints are unsupported.")
        if m["table"] is None:
            raise UnsupportedMappingError("Cannot reference an abstract TPC endpoint.")
        idcol = _name("r_", prefix + "_id")
        t["columns"].append(_column(idcol, "BIGINT", nullable))
        cols, refs = [idcol], ["id"]
        if m["strategy"] in ("single_table", "joined"):
            typecol = _name("r_", prefix + "_dtype")
            t["columns"].append(_column(typecol, "TEXT", nullable))
            cols.append(typecol)
            refs.append("dtype")
            allowed = [c for c in idx["descendants"][endpoint] if c in idx["concrete"]]
            t["checks"].append(quote(typecol) + " IS NULL OR " + quote(typecol) + " IN (" + ",".join(map(_literal, allowed)) + ")")
        fk(t, cols, m["table"], refs)
        return cols

    for assoc in sorted(model.get("associations", []), key=lambda a: a["name"]):
        name, source, target = assoc["name"], assoc["source"], assoc["target"]
        mult = assoc.get("multiplicity", "one-to-many")
        strategy = decisions["association:" + name]
        target_map = mapping[target]
        allowed_target = [c for c in idx["descendants"][target] if c in idx["concrete"]]
        if strategy == "direct_fk":
            if mult == "many-to-many":
                raise UnsupportedMappingError("many-to-many requires an association table.")
            t = tables[target_map["table"]]
            cols = endpoint_fk(t, source, name + "_source", True)
            applicable = "TRUE" if target_map["strategy"] == "table_per_class" else "dtype IN (" + ",".join(map(_literal, allowed_target)) + ")"
            t["checks"].append(f"({applicable}) OR {quote(cols[0])} IS NULL")
            if not assoc.get("nullable", False):
                t["checks"].append(f"NOT ({applicable}) OR {quote(cols[0])} IS NOT NULL")
            if mult == "one-to-one":
                t["unique"].append([cols[0]])
            association_mapping[name] = {"strategy": strategy, "table": t["name"], "source_columns": cols, "target_columns": ["id"], "target_class": target}
        else:
            t = table(_name("l_", name), "association")
            source_cols = endpoint_fk(t, source, name + "_source", False)
            target_cols = endpoint_fk(t, target, name + "_target", False)
            next(c for c in t["columns"] if c["name"] == source_cols[0])["primary_key"] = True
            next(c for c in t["columns"] if c["name"] == target_cols[0])["primary_key"] = True
            if mult in ("one-to-one", "one-to-many"):
                t["unique"].append([target_cols[0]])
            if mult == "one-to-one":
                t["unique"].append([source_cols[0]])
            association_mapping[name] = {"strategy": strategy, "table": t["name"], "source_columns": source_cols, "target_columns": target_cols, "target_class": target}
            if mult != "many-to-many" and not assoc.get("nullable", False):
                # Deferred target participation is checked at transaction end,
                # so entity/link insert order and cyclic associations remain legal.
                target_table = target_map["table"]
                condition = "TRUE" if target_map["strategy"] == "table_per_class" else "t.dtype IN (" + ",".join(map(_literal, allowed_target)) + ")"
                fn = _name("require_", name)
                body = f"CREATE FUNCTION {quote(fn)}() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $ymap$ BEGIN IF EXISTS (SELECT 1 FROM {quote(target_table)} t WHERE ({condition}) AND NOT EXISTS (SELECT 1 FROM {quote(t['name'])} l WHERE l.{quote(target_cols[0])}=t.id)) THEN RAISE EXCEPTION 'Required target participation violated: {name}'; END IF; RETURN NULL; END; $ymap$;"
                triggers.append(body)
                for trigger_table, suffix in ((target_table, "target"), (t["name"], "link")):
                    triggers.append(f"CREATE CONSTRAINT TRIGGER {quote(_name('ct_', name + '_' + suffix))} AFTER INSERT OR UPDATE OR DELETE ON {quote(trigger_table)} DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION {quote(fn)}();")

    table_list = [tables[k] for k in sorted(tables)]
    ddl = []
    for t in table_list:
        definitions = [f"{quote(c['name'])} {c['type']}" + (" NOT NULL" if not c["nullable"] else "") for c in t["columns"]]
        pk = [quote(c["name"]) for c in t["columns"] if c["primary_key"]]
        definitions.append("PRIMARY KEY (" + ", ".join(pk) + ")")
        definitions += ["UNIQUE (" + ", ".join(map(quote, cols)) + ")" for cols in t["unique"]]
        definitions += ["CHECK (" + check + ")" for check in t["checks"]]
        ddl.append(f"CREATE TABLE {quote(t['name'])} (" + ", ".join(definitions) + ");")
    for t in table_list:
        for i, f in enumerate(t["foreign_keys"]):
            ddl.append(f"ALTER TABLE {quote(t['name'])} ADD CONSTRAINT {quote(_name('fk_', t['name'] + '_' + str(i)))} FOREIGN KEY (" + ", ".join(map(quote, f["columns"])) + ") REFERENCES " + quote(f["references"]["table"]) + " (" + ", ".join(map(quote, f["references"]["columns"])) + ") MATCH FULL DEFERRABLE INITIALLY DEFERRED;")
    ddl.extend(triggers)
    columns = [c for t in table_list for c in t["columns"]]
    features = {"table_count": len(table_list), "column_count": len(columns), "foreign_key_count": sum(len(t["foreign_keys"]) for t in table_list), "pk_column_count": sum(c["primary_key"] for c in columns), "nullable_column_count": sum(c["nullable"] for c in columns), "association_table_count": sum(t["kind"] == "association" for t in table_list), "inheritance_table_count": sum(t["kind"] == "inheritance" for t in table_list), "class_count": len(idx["classes"]), "association_count": len(model.get("associations", [])), "inheritance_depth": max(len(p) - 1 for p in idx["paths"].values()), "check_count": sum(len(t["checks"]) for t in table_list), "unique_count": sum(len(t["unique"]) for t in table_list), "trigger_count": sum(s.startswith("CREATE CONSTRAINT TRIGGER") for s in triggers)}
    for strategy in STRATEGIES:
        features[strategy + "_count"] = sum(v == strategy for k, v in decisions.items() if k.startswith("inheritance:"))
    nodes, edges = [], []
    for t in table_list:
        tid = "table:" + t["name"]
        nodes.append({"id": tid, "type": "table", "name": t["name"], "kind": t["kind"]})
        for col in t["columns"]:
            cid = "column:" + t["name"] + "." + col["name"]
            nodes.append({"id": cid, "type": "column", "name": col["name"], "data_type": col["type"], "nullable": col["nullable"], "primary_key": col["primary_key"]})
            edges.append({"source": tid, "target": cid, "type": "contains"})
            if col.get("foreign_key"):
                edges.append({"source": cid, "target": "column:" + col["foreign_key"], "type": "foreign_key"})
    identity = json.dumps({"model": model, "decisions": decisions, "mapping_version": 1}, sort_keys=True, separators=(",", ":"))
    schema_id = model["name"] + "_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    candidate = {"schema_id": schema_id, "object_model": model["name"], "decisions": decisions, "tables": table_list, "ddl": "\n".join(ddl), "features": features, "graph": {"nodes": nodes, "edges": edges}, "source_model": copy.deepcopy(model), "mapping": mapping, "association_mapping": association_mapping, "generation": {"backend": backend, "mapping_version": 1, "excluded_choices": exclusions, "semantic_guarantee": "Encoded structural SQL constraints and supported loader invariants; not a full semantic-equivalence proof."}}
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: dict) -> None:
    """Check generated structural references; this is not semantic verification."""
    tables = {table["name"]: table for table in candidate["tables"]}
    if len(tables) != len(candidate["tables"]):
        raise UnsupportedMappingError("Generated table-name collision.")
    for table in tables.values():
        columns = {column["name"]: column for column in table["columns"]}
        if len(columns) != len(table["columns"]):
            raise UnsupportedMappingError("Generated column-name collision.")
        primary = [c["name"] for c in table["columns"] if c["primary_key"]]
        if not primary or any(columns[c]["nullable"] for c in primary):
            raise UnsupportedMappingError("Generated table requires a nonnullable primary key.")
        for fk in table["foreign_keys"]:
            target = tables.get(fk["references"]["table"])
            if target is None:
                raise UnsupportedMappingError("Generated foreign key references missing table.")
            target_cols = {c["name"]: c for c in target["columns"]}
            referenced = fk["references"]["columns"]
            target_pk = [c["name"] for c in target["columns"] if c["primary_key"]]
            if referenced not in [target_pk, *target["unique"]]:
                raise UnsupportedMappingError("Foreign key target must be primary or unique.")
            if len(fk["columns"]) != len(referenced):
                raise UnsupportedMappingError("Foreign key arity mismatch.")
            for source, dest in zip(fk["columns"], referenced):
                if source not in columns or dest not in target_cols or columns[source]["type"] != target_cols[dest]["type"]:
                    raise UnsupportedMappingError("Foreign key column/type mismatch.")
    graph_ids = {n["id"] for n in candidate["graph"]["nodes"]}
    if any(e["source"] not in graph_ids or e["target"] not in graph_ids for e in candidate["graph"]["edges"]):
        raise UnsupportedMappingError("Graph edge references missing node.")


def canonical_candidate(candidate: dict) -> dict:
    """Reject stale/tampered serialized SQL and reconstruct trusted generated SQL.

    Profiling never executes arbitrary DDL supplied in candidate JSON. All
    executable mapping/query metadata must match the current supported compiler.
    """
    if not isinstance(candidate, dict) or not isinstance(candidate.get("source_model"), dict):
        raise UnsupportedMappingError("Candidate must include its source_model object.")
    model = candidate["source_model"]
    validate_model(model)
    dimensions, exclusions = decision_space(model)
    decisions = candidate.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != {key for key, _ in dimensions}:
        raise UnsupportedMappingError("Candidate decisions do not match the supported source-model decision space.")
    for key, allowed in dimensions:
        if decisions[key] not in allowed:
            raise UnsupportedMappingError(f"Unsupported decision for {key}: {decisions[key]!r}.")
    rebuilt = _build_candidate(model, decisions, candidate.get("generation", {}).get("backend", "python"), exclusions)
    keys = ("schema_id", "object_model", "tables", "ddl", "features", "graph", "mapping", "association_mapping")
    if any(candidate.get(key) != rebuilt[key] for key in keys):
        raise UnsupportedMappingError("Candidate metadata/SQL is stale or modified. Regenerate candidates with the current supported compiler before profiling.")
    return rebuilt


def _alloy_decisions(dimensions: list, jar: str | Path | None, artifact_dir: str | Path | None, limit: int | None) -> Iterator[dict]:
    if jar is None or not Path(jar).is_file():
        raise AlloyUnavailableError("backend='alloy' requires an existing local Alloy distribution JAR via alloy_jar.")
    if not shutil.which("javac") or not shutil.which("java"):
        raise AlloyUnavailableError("Alloy backend requires a JDK with java and javac on PATH.")
    if artifact_dir is None:
        raise AlloyUnavailableError("Alloy backend requires artifact_dir to preserve specification and solver output.")
    out = Path(artifact_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = out / "java_build"
    build.mkdir(exist_ok=True)
    jar = Path(jar).resolve()
    src = Path(__file__).parent / "java" / "AlloyBridge.java"
    java_version = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=15)
    (out / "alloy_runtime.json").write_text(json.dumps({"alloy_jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest(), "bridge_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "java_version": (java_version.stdout + java_version.stderr).strip(), "backend_scope": "finite_strategy_choice_encoding_only"}, indent=2), encoding="utf-8")
    compile_result = subprocess.run(["javac", "-cp", str(jar), "-d", str(build), str(src)], capture_output=True, text=True)
    (out / "compile.stderr.txt").write_text(compile_result.stderr, encoding="utf-8")
    if compile_result.returncode:
        raise AlloyUnavailableError(f"Alloy Java API bridge did not compile; see {out / 'compile.stderr.txt'}.")
    atoms = {choice: "S" + str(i) for i, choice in enumerate(sorted({v for _, values in dimensions for v in values}))}
    spec = ["module mapping_choices", "abstract sig Strategy {}"]
    spec += ["one sig " + atom + " extends Strategy {}" for atom in atoms.values()]
    for i, (key, values) in enumerate(dimensions):
        spec += [f"one sig D{i} {{ choice: one Strategy }}", f"fact Allowed{i} {{ D{i}.choice in " + " + ".join(atoms[v] for v in values) + " }"]
    if not dimensions:
        spec.append("one sig EmptyChoice {}")
    scope = max(1, len(atoms))
    spec.append(f"run {{}} for {scope} but exactly {len(atoms)} Strategy")
    als = out / "mapping_choices.als"
    als.write_text("\n".join(spec) + "\n", encoding="utf-8")
    (out / "decision_encoding.json").write_text(json.dumps({"dimensions": dimensions, "strategy_atoms": atoms, "scope": scope, "meaning": "Finite strategy-choice admissibility only; generated DDL is constructed and checked in Python."}, indent=2), encoding="utf-8")
    stderr = (out / "solver.stderr.txt").open("w", encoding="utf-8")
    raw = (out / "solutions.jsonl").open("w", encoding="utf-8")
    command = ["java", "-cp", str(jar) + os.pathsep + str(build), "AlloyBridge", str(als), str(len(dimensions)), str(limit or 0)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True)
    reverse = {atom: choice for choice, atom in atoms.items()}
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if line.rstrip("\r\n") != "YMAP_CHOICES" and not line.startswith("YMAP_CHOICES\t"):
                raw.write(json.dumps({"solver_output": line.rstrip()}) + "\n")
                continue
            selected = line.rstrip("\r\n").split("\t")[1:]
            if len(selected) != len(dimensions):
                raise AlloyUnavailableError("Unexpected Alloy bridge decision output.")
            if any(atom not in reverse for atom in selected):
                raise AlloyUnavailableError("Unexpected Alloy bridge strategy atom.")
            decision = {key: reverse[atom] for (key, _), atom in zip(dimensions, selected)}
            raw.write(json.dumps(decision, sort_keys=True) + "\n")
            raw.flush()
            yield decision
        if process.wait() != 0:
            raise AlloyUnavailableError(f"Alloy failed; see {out / 'solver.stderr.txt'}.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()
        stderr.close()
        raw.close()


def generate_candidates(model: dict, max_candidates: int | None = None, backend="python", alloy_jar=None, artifact_dir=None) -> Iterator[dict]:
    """Yield candidates without materializing the Cartesian decision product.

    max_candidates truncates a deterministic lexicographic Python prefix, NOT a
    uniform sample and NOT an optimized top-k set. Exclusions are explicit in
    every candidate's generation metadata.
    """
    validate_model(model)
    if max_candidates is not None and (isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0):
        raise ValueError("max_candidates must be a positive integer or None.")
    dimensions, exclusions = decision_space(model)
    if backend == "python":
        keys = [key for key, _ in dimensions]
        choices = (dict(zip(keys, values)) for values in itertools.product(*(values for _, values in dimensions)))
    elif backend == "alloy":
        choices = _alloy_decisions(dimensions, alloy_jar, artifact_dir, max_candidates)
    else:
        raise ValueError("backend must be 'python' or 'alloy'; no automatic fallback is performed.")
    count = 0
    try:
        for decisions in choices:
            yield _build_candidate(model, decisions, backend, exclusions)
            count += 1
            if max_candidates is not None and count >= max_candidates:
                break
    finally:
        choices.close()
