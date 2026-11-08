"""Stages A and B for the paper workflow, over the supported ORM compiler.

The available Alloy bridge solves finite strategy choices. It does not encode
relational atoms or claim to recover the original study's candidate corpus.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from hashlib import sha256
from heapq import heapify, heappop, heappush
from itertools import product
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any

from .synthesis import (
    UnsupportedMappingError,
    _alloy_decisions,
    _build_candidate,
    _name,
    decision_space,
    model_index,
    quote,
    validate_model,
)


PAPER_FEATURES = (
    "table_count", "column_count", "pk_column_count", "foreign_key_count",
    "association_table_count", "max_inheritance_depth", "mean_inheritance_depth",
    "mean_table_width", "max_table_width", "foreign_key_density",
    "nullable_fraction", "mean_join_pressure",
)
PAPER_RELATIONS = ("hasColumn", "pkOf", "fkOf", "references", "inheritsFrom")
PAPER_SCHEMA_VERSION = 1


def _paper_tables(record: Mapping[str, Any]) -> list[dict]:
    """Apply paper mapping rules whose semantics are supported by the loader."""
    tables = {table["name"]: deepcopy(table) for table in record["tables"]}
    model = record["source_model"]
    idx = model_index(model)
    concrete = set(idx["concrete"])

    # In a single-table hierarchy, only attributes inapplicable to some of the
    # represented types need conditional nullability. Existing CHECKs retain
    # per-type constraints on the remaining columns.
    for cls, mapping in record["mapping"].items():
        if mapping["strategy"] != "single_table":
            continue
        if concrete.intersection(idx["descendants"][cls]) != concrete.intersection(idx["descendants"][mapping["root"]]):
            continue
        mandatory = {_name("a_", cls + "_" + attr["name"]) for attr in idx["classes"][cls].get("attributes", ()) if not attr.get("nullable", False)}
        for column in tables[mapping["table"]]["columns"]:
            if column["name"] in mandatory:
                column["nullable"] = False

    for association in model.get("associations", ()):
        mapping = record["association_mapping"][association["name"]]
        table = tables[mapping["table"]]
        if mapping["strategy"] == "association_table" and association.get("multiplicity", "one-to-many") == "one-to-many":
            # The FK to the many side is the association table key; the
            # generated source/target ID and discriminator columns remain intact.
            many_id = mapping["target_columns"][0]
            for column in table["columns"]:
                column["primary_key"] = column["name"] == many_id
            table["unique"] = [columns for columns in table["unique"] if columns != [many_id]]
        elif mapping["strategy"] == "direct_fk" and not association.get("nullable", False):
            target = association["target"]
            mapped_target = record["mapping"][target]
            conditional = mapped_target["strategy"] == "single_table" and concrete.intersection(idx["descendants"][target]) != concrete.intersection(idx["descendants"][mapped_target["root"]])
            if not conditional:
                for column in table["columns"]:
                    if column["name"] in mapping["source_columns"]:
                        column["nullable"] = False

    ordered = []
    for name in _table_order(tables):
        table = tables[name]
        fks = {column for fk in table["foreign_keys"] for column in fk["columns"]}
        # A composite key's internal order can affect its index. Only reorder
        # declarations here; retain the original constraint order explicitly.
        table["primary_key"] = [c["name"] for c in table["columns"] if c["primary_key"]]
        table["columns"].sort(key=lambda c: (0 if c["primary_key"] else 1 if c["name"] in fks else 2, c["name"]))
        table["foreign_keys"].sort(key=lambda fk: (tuple(fk["columns"]), fk["references"]["table"], tuple(fk["references"]["columns"])))
        table["unique"].sort()
        table["checks"].sort()
        ordered.append(table)
    return ordered


def _table_order(tables: Mapping[str, dict]) -> list[str]:
    """Topologically order FK components, with lexicographic ties and cycles."""
    deps = {name: {fk["references"]["table"] for fk in table["foreign_keys"]} - {name} for name, table in tables.items()}
    users = {name: set() for name in tables}
    for name, targets in deps.items():
        for target in targets:
            users[target].add(name)

    # Kosaraju with an iterative DFS: FK cycles can be arbitrarily long.
    seen: set[str] = set()
    finished: list[str] = []
    for root in sorted(tables):
        if root in seen:
            continue
        stack = [(root, iter(sorted(deps[root])))]
        seen.add(root)
        while stack:
            name, adjacent = stack[-1]
            next_name = next(adjacent, None)
            if next_name is None:
                finished.append(name)
                stack.pop()
            elif next_name not in seen:
                seen.add(next_name)
                stack.append((next_name, iter(sorted(deps[next_name]))))

    group: dict[str, str] = {}
    components: dict[str, list[str]] = {}
    for root in reversed(finished):
        if root in group:
            continue
        pending, members = [root], []
        group[root] = root
        while pending:
            name = pending.pop()
            members.append(name)
            for successor in users[name]:
                if successor not in group:
                    group[successor] = root
                    pending.append(successor)
        components[root] = sorted(members)

    required = {root: set() for root in components}
    dependent = {root: set() for root in components}
    for name, targets in deps.items():
        for target in targets:
            if group[name] != group[target]:
                required[group[name]].add(group[target])
                dependent[group[target]].add(group[name])
    ready = [(components[root][0], root) for root in components if not required[root]]
    heapify(ready)
    result = []
    while ready:
        _, root = heappop(ready)
        result.extend(components[root])
        for successor in dependent[root]:
            required[successor].remove(root)
            if not required[successor]:
                heappush(ready, (components[successor][0], successor))
    return result


def _canonical_sql(record: Mapping[str, Any], tables: list[dict]) -> str:
    def identifiers(names):
        return ", ".join(map(quote, names))

    statements, delayed, emitted = [], [], set()
    for table in tables:
        name = table["name"]
        emitted.add(name)  # PostgreSQL permits an inline self-reference.
        definitions = [f"{quote(c['name'])} {c['type']}" + ("" if c["nullable"] else " NOT NULL") for c in table["columns"]]
        definitions.append("PRIMARY KEY (" + identifiers(table["primary_key"]) + ")")
        definitions.extend("UNIQUE (" + identifiers(cols) + ")" for cols in table["unique"])
        definitions.extend("CHECK (" + check + ")" for check in table["checks"])
        for i, fk in enumerate(table["foreign_keys"]):
            ref = fk["references"]
            clause = f"CONSTRAINT {quote(_name('fk_', name + '_' + str(i)))} FOREIGN KEY ({identifiers(fk['columns'])}) REFERENCES {quote(ref['table'])} ({identifiers(ref['columns'])}) MATCH FULL DEFERRABLE INITIALLY DEFERRED"
            if ref["table"] in emitted:
                definitions.append(clause)
            else:
                delayed.append(f"ALTER TABLE {quote(name)} ADD {clause};")
        statements.append(f"CREATE TABLE {quote(name)} (" + ", ".join(definitions) + ");")
    # A deferred participation check is needed for mandatory relationships in
    # link tables. Keep trusted compiler functions and triggers verbatim.
    triggers = [line for line in record["ddl"].splitlines() if line.startswith(("CREATE FUNCTION ", "CREATE CONSTRAINT TRIGGER "))]
    return "\n".join(statements + delayed + triggers)


def _paper_graph(record: Mapping[str, Any], tables: list[dict]) -> dict:
    nodes, edges = [], []

    def edge(source, target, relation):
        edges.append({"source": source, "target": target, "type": relation})

    for table in tables:
        table_id = "table:" + table["name"]
        nodes.append({"id": table_id, "type": "table", "column_count": len(table["columns"])})
        targets: dict[str, set[str]] = {}
        for fk in table["foreign_keys"]:
            for column in fk["columns"]:
                targets.setdefault(column, set()).add(fk["references"]["table"])
        unique = {columns[0] for columns in [table["primary_key"], *table["unique"]] if len(columns) == 1}
        for column in table["columns"]:
            name = column["name"]
            column_id = "column:" + table["name"] + "." + name
            nodes.append({
                "id": column_id, "type": "column", "data_type": column["type"],
                "primary_key": column["primary_key"], "foreign_key": name in targets,
                "nullable": column["nullable"], "unique": name in unique,
                "discriminator": name == "dtype",
            })
            edge(table_id, column_id, "hasColumn")
            if column["primary_key"]:
                edge(column_id, table_id, "pkOf")
            if name in targets:
                edge(column_id, table_id, "fkOf")
                for target in sorted(targets[name]):
                    edge(column_id, "table:" + target, "references")

    mapping = record["mapping"]
    for name in sorted(mapping):
        entry = mapping[name]
        path = entry["path"]
        if entry["strategy"] == "joined" and len(path) > 1:
            edge("table:" + entry["table"], "table:" + mapping[path[-2]]["table"], "inheritsFrom")
    return {"nodes": nodes, "edges": edges}


def paper_graph(record: Mapping[str, Any]) -> dict:
    """Return the table/column graph with the five directed paper relations."""
    graph = record.get("paper_graph", record.get("graph"))
    if isinstance(graph, Mapping) and isinstance(graph.get("nodes"), list) and graph["nodes"] and isinstance(graph.get("edges"), list) and all(isinstance(edge, Mapping) and edge.get("type") in PAPER_RELATIONS for edge in graph["edges"]):
        return dict(graph)
    if "paper_graph" not in record and {"tables", "source_model", "mapping", "association_mapping"} <= record.keys():
        return _paper_graph(record, _paper_tables(record))
    raise ValueError("The paper schema graph requires generated mapping metadata or an explicit paper_graph with the five paper relation types.")


def _paper_features(record: Mapping[str, Any], tables: list[dict]) -> dict[str, int | float]:
    widths = [len(table["columns"]) for table in tables]
    num_tables, num_columns = len(tables), sum(widths)
    num_fks = sum(len(table["foreign_keys"]) for table in tables)
    idx = model_index(record["source_model"])
    mapping = record["mapping"]
    # Depth is measured in inheritsFrom edges, not the object depth of a
    # hierarchy stored in one table or with repeated inherited attributes.
    depths = [max(len(idx["paths"][cls]) - 1 for cls in idx["descendants"][root]) if mapping[root]["strategy"] == "joined" else 0 for root in idx["roots"] if len(idx["descendants"][root]) > 1]
    # An instance can be reconstructed only for a concrete class. Associations
    # are not fetched when reconstructing one instance's own attributes.
    joins = [len(idx["paths"][cls]) - 1 if mapping[cls]["strategy"] == "joined" else 0 for cls in idx["concrete"]]
    values = (
        num_tables, num_columns,
        sum(c["primary_key"] for table in tables for c in table["columns"]),
        num_fks, sum(table["kind"] == "association" for table in tables),
        max(depths, default=0), sum(depths) / len(depths) if depths else 0.0,
        num_columns / num_tables, max(widths), num_fks / num_tables,
        sum(c["nullable"] for table in tables for c in table["columns"]) / num_columns,
        sum(joins) / len(joins),
    )
    return dict(zip(PAPER_FEATURES, values))


def paper_features(record: Mapping[str, Any]) -> dict[str, int | float]:
    """Return the twelve Appendix B descriptors in their specified order."""
    features = record.get("paper_features", record.get("features"))
    if isinstance(features, Mapping) and all(name in features for name in PAPER_FEATURES):
        result = {name: features[name] for name in PAPER_FEATURES}
        if all(isinstance(value, (float, int)) and not isinstance(value, bool) and isfinite(value) for value in result.values()):
            return result
    if "paper_features" not in record and {"tables", "source_model", "mapping", "association_mapping"} <= record.keys():
        return _paper_features(record, _paper_tables(record))
    raise ValueError("Paper descriptors require generated mapping metadata or all twelve Appendix B descriptors in paper_features.")


def _paper_views(record: dict) -> dict:
    tables = _paper_tables(record)
    ddl = _canonical_sql(record, tables)
    fingerprint = sha256(ddl.encode("utf-8")).hexdigest()
    return {
        "schema_id": record["object_model"] + "_paper_" + fingerprint[:24],
        "compiler_schema_id": record["schema_id"],
        "paper_ddl": ddl,
        "paper_ddl_sha256": fingerprint,
        "paper_features": _paper_features(record, tables),
        "paper_graph": _paper_graph(record, tables),
        "paper_tables": tables,
    }


def paper_executable_candidate(record: dict) -> dict:
    """Rebuild trusted executable SQL and verify a serialized paper candidate."""
    if not isinstance(record, dict) or not isinstance(record.get("source_model"), dict):
        raise UnsupportedMappingError("Paper candidate must include its source_model object.")
    model = record["source_model"]
    validate_model(model)
    dimensions, exclusions = _paper_decision_space(model)
    decisions = record.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != {key for key, _ in dimensions} or any(decisions[key] not in choices for key, choices in dimensions):
        raise UnsupportedMappingError("Candidate uses a mapping excluded from the paper workflow.")
    generation = record.get("generation")
    if not isinstance(generation, Mapping) or generation.get("backend") not in ("alloy", "python"):
        raise UnsupportedMappingError("Paper candidate must identify its enumeration backend.")
    rebuilt = _build_candidate(model, decisions, generation["backend"], exclusions)
    views = _paper_views(rebuilt)
    metadata = ("object_model", "features", "graph", "mapping", "association_mapping")
    executable = (record.get("ddl"), record.get("tables"))
    if (any(record.get(key) != value for key, value in views.items())
        or any(record.get(key) != rebuilt[key] for key in metadata)
        or executable not in ((rebuilt["ddl"], rebuilt["tables"]), (views["paper_ddl"], views["paper_tables"]))):
        raise UnsupportedMappingError("Paper candidate representations or DDL are stale or modified. Regenerate with the paper workflow before profiling.")
    return {**record, **views, "ddl": views["paper_ddl"], "tables": views["paper_tables"]}


def _paper_decision_space(model: dict) -> tuple[list[tuple[str, list[str]]], list[str]]:
    dimensions, exclusions = decision_space(model)
    one_to_one = {"association:" + association["name"] for association in model.get("associations", ()) if association.get("multiplicity") == "one-to-one"}
    for key in sorted(one_to_one):
        exclusions.append(f"association_table excluded for {key}: the paper specifies a unique foreign key for a one-to-one association.")
    return [(key, ["direct_fk"] if key in one_to_one else choices) for key, choices in dimensions], exclusions


def paper_candidates(
    model: dict, *, max_tables: int, backend: str = "alloy", alloy_jar: str | Path | None = None,
    artifact_dir: str | Path | None = None, seed: int | None = None,
    counts: MutableMapping[str, int] | None = None, deduplicate: bool = True,
) -> Iterator[dict]:
    """Stream bounded candidates and optionally update per-model audit counts.

    Alloy selects finite admissible strategy vectors. A Python table-count
    filter enforces the supplied upper bound before compilation; it is not an
    Alloy table-atom scope. A seed reorders declarations and choices, not the
    solver RNG, and does not imply random or independent samples.
    """
    validate_model(model)
    if isinstance(max_tables, bool) or not isinstance(max_tables, int) or max_tables < 1:
        raise ValueError("max_tables must be a positive integer.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None.")
    if backend not in ("python", "alloy"):
        raise ValueError("backend must be 'python' or 'alloy'; no automatic fallback is performed.")
    dimensions, exclusions = _paper_decision_space(model)
    idx = model_index(model)
    concrete = set(idx["concrete"])
    table_cost = {}
    fixed_tables = 0
    for root in idx["roots"]:
        family = idx["descendants"][root]
        if len(family) == 1:
            fixed_tables += 1
        else:
            table_cost["inheritance:" + root] = {"single_table": 1, "joined": len(family), "table_per_class": len(concrete.intersection(family))}
    for assoc in model.get("associations", ()):
        table_cost["association:" + assoc["name"]] = {"direct_fk": 0, "association_table": 1}

    if seed is not None:
        rng = Random(seed)
        dimensions = [(key, choices[:]) for key, choices in dimensions]
        rng.shuffle(dimensions)
        for _, choices in dimensions:
            rng.shuffle(choices)
    keys = tuple(key for key, _ in dimensions)
    if backend == "alloy":
        decisions = _alloy_decisions(dimensions, alloy_jar, artifact_dir, None)
    else:
        decisions = (dict(zip(keys, choices)) for choices in product(*(choices for _, choices in dimensions)))

    audit = counts if counts is not None else {}
    audit.update(decision_vectors=0, out_of_scope=0, enumerated=0, duplicates=0, unique=0)
    seen = set()
    try:
        for decision in decisions:
            audit["decision_vectors"] += 1
            size = fixed_tables + sum(table_cost[key][value] for key, value in decision.items())
            if size > max_tables:
                audit["out_of_scope"] += 1
                continue
            audit["enumerated"] += 1
            candidate = _build_candidate(model, decision, backend, exclusions)
            if len(candidate["tables"]) != size:
                raise RuntimeError("Paper table budget disagrees with the ORM compiler output.")
            views = _paper_views(candidate)
            fingerprint = views["paper_ddl_sha256"]
            if fingerprint in seen:
                audit["duplicates"] += 1
                if deduplicate:
                    continue
            else:
                audit["unique"] += 1
                seen.add(fingerprint)
            candidate["generation"].update({
                "paper_schema_version": PAPER_SCHEMA_VERSION, "max_tables": max_tables,
                "scope_mechanism": "table_count_filter_before_compilation",
                "choice_order_seed": seed,
                "solver_scope": "finite_strategy_choice_encoding_only" if backend == "alloy" else None,
                "compiler_note": "Polymorphic integrity uses deferred foreign keys to unique (id, dtype) superkeys and conditional checks; the original relational Alloy translator is not supplied.",
            })
            yield {**candidate, **views}
    finally:
        decisions.close()
