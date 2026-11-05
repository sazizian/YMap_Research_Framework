"""Deterministic representations of *actual* schema declarations.

Hashing here identifies cache entries; it never generates an embedding.  This
module has no dependency on PyTorch, Transformers, a database, or a network.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

REPRESENTATION_VERSION = "schema-lexical-graph-v2"
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9$]*")
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_DOLLAR = re.compile(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$")
_OPERATOR_CHARACTERS = frozenset("+-*/<>=~!@#%^&|`?")


def ddl_tokens(ddl: str) -> list[str]:
    """Lex PostgreSQL DDL, removing comments and normalizing unquoted words.

    Quoted identifiers, SQL strings, E-strings, and dollar-quoted strings retain
    their exact spelling. Nested block comments are supported. This is lexical
    normalization, not a SQL parser or a test of semantic schema equivalence.
    Invalid unterminated comments/quotes raise instead of sharing a cache key.
    """
    if not isinstance(ddl, str) or not ddl.strip():
        raise ValueError("ddl must be a nonempty string")
    result: list[str] = []
    i, n = 0, len(ddl)
    while i < n:
        if ddl[i].isspace():
            i += 1
            continue
        if ddl.startswith("--", i):
            end = ddl.find("\n", i + 2)
            i = n if end < 0 else end + 1
            continue
        if ddl.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if ddl.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif ddl.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                raise ValueError("unterminated SQL block comment")
            continue
        dollar = _DOLLAR.match(ddl, i)
        if dollar:
            delimiter = dollar.group()
            end = ddl.find(delimiter, dollar.end())
            if end < 0:
                raise ValueError("unterminated dollar-quoted SQL string")
            result.append(ddl[i:end + len(delimiter)])
            i = end + len(delimiter)
            continue
        # E'...' is treated as one token; backslash escapes only in E-strings.
        escaped = ddl[i:i + 1].lower() == "e" and i + 1 < n and ddl[i + 1] == "'"
        if ddl[i] in "\"'" or escaped:
            start = i
            if escaped:
                i += 1
            quote, i = ddl[i], i + 1
            closed = False
            while i < n:
                if escaped and ddl[i] == "\\":
                    i += 2
                elif ddl[i] == quote:
                    if i + 1 < n and ddl[i + 1] == quote:
                        i += 2
                    else:
                        i, closed = i + 1, True
                        break
                else:
                    i += 1
            if not closed:
                raise ValueError("unterminated SQL quote")
            token = ddl[start:i]
            result.append("e" + token[1:] if escaped else token)
            continue
        word = _WORD.match(ddl, i)
        number = _NUMBER.match(ddl, i)
        if word:
            result.append(word.group().lower())
            i = word.end()
        elif number:
            result.append(number.group())
            i = number.end()
        else:
            # PostgreSQL permits extensible operator names. Preserve complete
            # operator runs; '@>' must not collide with separated '@' and '>'.
            if ddl.startswith(("::", ":="), i):
                result.append(ddl[i:i + 2])
                i += 2
            elif ddl[i] in _OPERATOR_CHARACTERS:
                start, i = i, i + 1
                while i < n and ddl[i] in _OPERATOR_CHARACTERS and not ddl.startswith(("--", "/*"), i):
                    i += 1
                result.append(ddl[start:i])
            else:
                result.append(ddl[i])
                i += 1
    if not result:
        raise ValueError("ddl contains no SQL tokens")
    return result


def canonical_ddl(ddl: str) -> str:
    return " ".join(ddl_tokens(ddl))


def representation_key(ddl: str, parameters: Mapping[str, Any]) -> str:
    """SHA256 identity for preprocessing/cache parameters, not an embedding."""
    payload = {"version": REPRESENTATION_VERSION, "ddl": canonical_ddl(ddl), "parameters": parameters}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


GRAPH_NODE_FEATURES = (
    "is_table", "is_column", "entity_table", "inheritance_table", "association_table",
    "nullable", "primary_key", "unique", "numeric_type", "text_type", "temporal_type",
    "boolean_type", "binary_type", "json_type", "other_type", "log_contains_count",
    "log_incoming_fk_count", "log_outgoing_fk_count",
)
GRAPH_RELATIONS = ("contains", "contained_by", "foreign_key", "referenced_by")


@dataclass(frozen=True)
class SchemaGraph:
    node_ids: tuple[str, ...]
    node_features: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    edge_types: np.ndarray


def schema_graph(graph: Mapping[str, Any]) -> SchemaGraph:
    """Encode table/column declarations and typed edges, including reverse edges.

    Input follows the generated candidate's ``graph`` field. Unknown node/edge
    types and dangling endpoints fail explicitly; no graph is invented from text.
    """
    if not isinstance(graph, Mapping):
        raise ValueError("candidate graph must be an object with nodes and edges")
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", [])
    if not isinstance(raw_nodes, (list, tuple)) or not raw_nodes:
        raise ValueError("candidate graph must contain at least one table and its columns")
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError("candidate graph edges must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for node in raw_nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str) or not node["id"]:
            raise ValueError("every graph node requires a string id")
        if node["id"] in by_id:
            raise ValueError(f"duplicate graph node id: {node['id']}")
        if node.get("type") not in ("table", "column"):
            raise ValueError(f"unsupported graph node type: {node.get('type')}")
        if node["type"] == "table" and node.get("kind", "entity") not in ("entity", "inheritance", "association"):
            raise ValueError(f"unsupported table kind: {node.get('kind')!r}")
        for flag in ("nullable", "primary_key", "unique"):
            if flag in node and type(node[flag]) is not bool:
                raise ValueError(f"graph node {flag} must be a boolean")
        by_id[node["id"]] = node
    node_ids = tuple(sorted(by_id))
    lookup = {node_id: i for i, node_id in enumerate(node_ids)}
    if not any(node["type"] == "table" for node in by_id.values()):
        raise ValueError("candidate graph has no table nodes")
    contains = {node_id: 0 for node_id in node_ids}
    fk_in = dict(contains)
    fk_out = dict(contains)
    edges: set[tuple[int, int, int]] = set()
    for edge in raw_edges:
        if not isinstance(edge, Mapping):
            raise ValueError("each graph edge must be an object")
        source, target, kind = edge.get("source"), edge.get("target"), edge.get("type")
        if source not in lookup or target not in lookup:
            raise ValueError(f"dangling graph edge: {source!r} -> {target!r}")
        if kind == "contains":
            if by_id[source]["type"] != "table" or by_id[target]["type"] != "column":
                raise ValueError("contains edges must run from a table to a column")
            relation, reverse = 0, 1
        elif kind == "foreign_key":
            if by_id[source]["type"] != "column" or by_id[target]["type"] != "column":
                raise ValueError("foreign_key edges must connect columns")
            relation, reverse = 2, 3
        else:
            raise ValueError(f"unsupported graph edge type: {kind!r}")
        edges.add((lookup[source], lookup[target], relation))
        edges.add((lookup[target], lookup[source], reverse))
    # Derive counts after deduplication, so repeated JSON edges change no features.
    parents = {node_id: 0 for node_id in node_ids}
    for source, target, relation in edges:
        if relation == 0:
            contains[node_ids[source]] += 1
            parents[node_ids[target]] += 1
        elif relation == 2:
            fk_out[node_ids[source]] += 1
            fk_in[node_ids[target]] += 1
    if any(parents[node_id] != 1 for node_id in node_ids if by_id[node_id]["type"] == "column"):
        raise ValueError("every column must belong to exactly one table through a contains edge")
    features = []
    for node_id in node_ids:
        node = by_id[node_id]
        table = node["type"] == "table"
        kind = node.get("kind", "entity")
        dtype = str(node.get("data_type", "")).upper().split("(")[0].split("[")[0].strip()
        base_type = dtype.split()[0] if dtype else ""
        categories = [False] * 7
        if not table:
            if base_type in ("INT", "INT2", "INT4", "INT8", "SMALLINT", "INTEGER", "BIGINT", "NUMERIC", "DECIMAL", "REAL", "DOUBLE", "FLOAT", "FLOAT4", "FLOAT8", "SERIAL", "SMALLSERIAL", "BIGSERIAL", "MONEY"):
                categories[0] = True
            elif base_type in ("CHAR", "CHARACTER", "VARCHAR", "TEXT", "UUID", "NAME", "CITEXT", "BPCHAR"):
                categories[1] = True
            elif base_type in ("DATE", "TIME", "TIMETZ", "TIMESTAMP", "TIMESTAMPTZ", "INTERVAL"):
                categories[2] = True
            elif base_type in ("BOOL", "BOOLEAN"):
                categories[3] = True
            elif base_type in ("BYTEA", "BINARY", "BLOB", "BIT", "VARBIT"):
                categories[4] = True
            elif base_type in ("JSON", "JSONB"):
                categories[5] = True
            else:
                categories[6] = True
        row = [table, not table, table and kind == "entity", table and kind == "inheritance", table and kind == "association", not table and bool(node.get("nullable", False)), not table and bool(node.get("primary_key", False)), not table and bool(node.get("unique", False)), *categories, math.log1p(contains[node_id]), math.log1p(fk_in[node_id]), math.log1p(fk_out[node_id])]
        features.append(row)
    ordered_edges = sorted(edges)
    return SchemaGraph(node_ids, np.asarray(features, dtype=np.float32), np.asarray([e[0] for e in ordered_edges], dtype=np.int64), np.asarray([e[1] for e in ordered_edges], dtype=np.int64), np.asarray([e[2] for e in ordered_edges], dtype=np.int64))


def symbolic_feature_names(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    vocabulary = set()
    for record in records:
        features = record.get("features")
        if not isinstance(features, Mapping) or any(not isinstance(name, str) or not name for name in features):
            raise ValueError("candidate features require nonempty string names in a numeric object")
        vocabulary.update(features)
    names = tuple(sorted(vocabulary))
    if not names:
        raise ValueError("symbolic view requires nonempty numerical candidate features")
    return names


def symbolic_matrix(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    rows = []
    for record in records:
        declared = record.get("features")
        if not isinstance(declared, Mapping):
            raise ValueError("candidate features must be a numeric object")
        row = []
        for name in names:
            value = declared.get(name, 0.0)
            if isinstance(value, (str, bytes)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or abs(float(value)) > float(np.finfo(np.float32).max):
                raise ValueError(f"nonfinite or nonnumeric feature {name!r}")
            row.append(float(value))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32).reshape(len(records), len(names))
