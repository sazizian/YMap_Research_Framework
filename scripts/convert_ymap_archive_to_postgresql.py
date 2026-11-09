"""Convert the 11,772 archived Y-Map MySQL schema files to PostgreSQL DDL.

Requires only Python 3.9+. Reads the extracted cohort folder or the original
cohort ZIP, selects archive rows from candidates.csv, and leaves inputs alone.
The converter supports the source cohort's DDL grammar, not general MySQL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import BadZipFile, ZipFile


EXPECTED = {"native": 728, "archive": 11_772}
SQL_PREFIX = "Dataset/YMap/All_SQL_Solutions/"
VERSION = "ymap-archive-ddl-pg-v1"
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
PATH_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
PSEUDO_HEADER = re.compile(
    r"(?im)^\s*CREATE DATABASE FOR [^\r\n;]+\.sql[ \t]*(?:\r?\n|$)"
)
LEX = re.compile(
    r"(?P<space>\s+)|(?P<comment>--(?=\s|$)[^\r\n]*)"
    r"|(?P<identifier>`(?:[^`\r\n]|``)+`)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9]*)|(?P<number>[0-9]+)"
    r"|(?P<punct>[(),;])|(?P<unsupported>.)"
)
SIMPLE_TYPES = {"INT": "integer", "BOOLEAN": "boolean", "LONGBLOB": "bytea"}
ACTIONS = {
    ("CASCADE",): "CASCADE",
    ("RESTRICT",): "RESTRICT",
    ("SET", "NULL"): "SET NULL",
    # InnoDB treats NO ACTION as RESTRICT; preserve immediate checks.
    ("NO", "ACTION"): "RESTRICT",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def lf(data):
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def columns_sql(names):
    return "(" + ", ".join(map(quote, names)) + ")"


@dataclass
class Table:
    name: str
    columns: dict = field(default_factory=dict)
    primary: tuple = ()
    indexes: list = field(default_factory=list)


@dataclass(frozen=True)
class ForeignKey:
    table: str
    name: str
    columns: tuple
    target: str
    target_columns: tuple
    actions: dict


class Parser:
    def __init__(self, text):
        text = PSEUDO_HEADER.sub("", text)
        self.tokens = []
        for match in LEX.finditer(text):
            kind = match.lastgroup
            if kind == "unsupported":
                line = text.count("\n", 0, match.start()) + 1
                raise ValueError(f"Unsupported MySQL character at line {line}: {match.group()!r}")
            if kind not in {"space", "comment"}:
                self.tokens.append(match.group())
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def take(self):
        token = self.peek()
        check(bool(token), "Unexpected end of DDL")
        self.pos += 1
        return token

    def accept(self, *tokens):
        stop = self.pos + len(tokens)
        if tuple(t.upper() for t in self.tokens[self.pos:stop]) == tokens:
            self.pos = stop
            return True
        return False

    def expect(self, *tokens):
        check(self.accept(*tokens), f"Expected {' '.join(tokens)}; found {self.peek() or 'end of DDL'}")

    def ident(self):
        token = self.take()
        if token.startswith("`"):
            value = token[1:-1].replace("``", "`")
        else:
            check(WORD.fullmatch(token), f"Expected an identifier, got {token!r}")
            value = token
        check(bool(value) and "\x00" not in value, "Empty or invalid identifier")
        return value

    def unsigned_number(self):
        token = self.take()
        check(token.isascii() and token.isdecimal(), f"Expected a type size, got {token!r}")
        return int(token)

    def keys(self):
        self.expect("(")
        names = [self.ident()]
        while self.accept(","):
            names.append(self.ident())
        self.expect(")")
        check(len(names) == len(set(names)), "Repeated column in a key")
        return tuple(names)

    def column_type(self):
        kind = self.take().upper()
        if kind in SIMPLE_TYPES:
            return SIMPLE_TYPES[kind]
        if kind == "VARCHAR":
            self.expect("(")
            size = self.unsigned_number()
            self.expect(")")
            check(1 <= size <= 10_485_760, f"Unsupported VARCHAR width: {size}")
            return f"varchar({size})"
        if kind in {"DECIMAL", "NUMERIC"}:
            self.expect("(")
            precision = self.unsigned_number()
            self.expect(",")
            scale = self.unsigned_number()
            self.expect(")")
            check(1 <= precision <= 65 and 0 <= scale <= precision, "Unsupported DECIMAL precision/scale")
            return f"numeric({precision},{scale})"
        raise ValueError(f"Unsupported column type: {kind}")

    def table(self):
        table = Table(self.ident())
        self.expect("(")
        while True:
            if self.accept("PRIMARY", "KEY"):
                check(not table.primary, f"Multiple primary keys in {table.name}")
                table.primary = self.keys()
            elif self.peek().upper() in {"KEY", "INDEX", "UNIQUE"}:
                unique = self.accept("UNIQUE")
                check(self.accept("KEY") or self.accept("INDEX"), "Expected KEY or INDEX after UNIQUE")
                table.indexes.append((self.ident(), self.keys(), unique))
            else:
                column = self.ident()
                check(column not in table.columns, f"Duplicate column {table.name}.{column}")
                sql_type = self.column_type()
                not_null = self.accept("NOT", "NULL")
                if not not_null:
                    self.accept("NULL")
                table.columns[column] = (sql_type, not_null)
            if not self.accept(","):
                break
        self.expect(")", ";")
        check(bool(table.columns), f"Empty table: {table.name}")
        return table

    def foreign_key(self, table):
        self.expect("ADD", "CONSTRAINT")
        name = self.ident()
        self.expect("FOREIGN", "KEY")
        keys = self.keys()
        self.expect("REFERENCES")
        target, target_columns = self.ident(), self.keys()
        actions = {"DELETE": "RESTRICT", "UPDATE": "RESTRICT"}
        seen = set()
        while self.accept("ON"):
            event = self.take().upper()
            check(event in actions and event not in seen, f"Unexpected or repeated FK event: {event}")
            seen.add(event)
            action = (self.take().upper(),)
            if action[0] in {"NO", "SET"}:
                action += (self.take().upper(),)
            check(action in ACTIONS, f"Unsupported FK action: {' '.join(action)}")
            actions[event] = ACTIONS[action]
        return ForeignKey(table, name, keys, target, target_columns, actions)

    def parse(self):
        tables, foreign_keys = {}, []
        if self.accept("USE"):
            self.ident()
            self.expect(";")
        while self.peek():
            if self.accept("CREATE", "TABLE"):
                table = self.table()
                check(table.name not in tables, f"Duplicate table: {table.name}")
                tables[table.name] = table
            elif self.accept("ALTER", "TABLE"):
                name = self.ident()
                foreign_keys.append(self.foreign_key(name))
                while self.accept(","):
                    foreign_keys.append(self.foreign_key(name))
                self.expect(";")
            else:
                raise ValueError(f"Unsupported SQL statement beginning with {self.peek()!r}")
        check(bool(tables), "No CREATE TABLE statements found")
        return tables, foreign_keys


def new_name(name, scope, used):
    """Keep valid names when possible; PostgreSQL index names share a namespace."""
    if len(name.encode("utf-8")) <= 63 and name not in used:
        used.add(name)
        return name
    prefix = name.encode("utf-8")[:45].decode("utf-8", "ignore")
    counter = 0
    while True:
        suffix = hash_bytes(f"{scope}\0{name}\0{counter}".encode("utf-8"))[:16]
        candidate = prefix + "_" + suffix
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def convert(text):
    tables, foreign_keys = Parser(text).parse()
    renamed, statements, index_sql = [], [], []
    relation_names = set(tables)
    constraint_names = {name: set() for name in tables}
    for table in tables.values():
        for name in (table.name, *table.columns):
            check(len(name.encode("utf-8")) <= 63, f"Table or column name exceeds PostgreSQL's 63-byte limit: {name}")
        check(set(table.primary) <= table.columns.keys(), f"Unknown primary-key column in {table.name}")
        specs = [
            f"  {quote(column)} {kind}" + (" NOT NULL" if not_null else "")
            for column, (kind, not_null) in table.columns.items()
        ]
        if table.primary:
            name = new_name(table.name + "_pkey", table.name + ":primary", relation_names)
            constraint_names[table.name].add(name)
            specs.append(f"  CONSTRAINT {quote(name)} PRIMARY KEY {columns_sql(table.primary)}")
        statements.append(f"CREATE TABLE {quote(table.name)} (\n" + ",\n".join(specs) + "\n);")
        seen = set()
        for name, cols, unique in table.indexes:
            check(name not in seen and set(cols) <= table.columns.keys(), f"Duplicate or invalid index {table.name}.{name}")
            seen.add(name)
            assigned = new_name(name, table.name + ":index", relation_names)
            if assigned != name:
                renamed.append({"kind": "index", "table": table.name, "mysql": name, "postgresql": assigned})
            index_sql.append(f"CREATE {'UNIQUE ' if unique else ''}INDEX {quote(assigned)} ON {quote(table.name)} {columns_sql(cols)};")

    fk_sql = []
    seen_keys = set()
    for fk in foreign_keys:
        local, target = tables.get(fk.table), tables.get(fk.target)
        check(local is not None and target is not None, f"FK {fk.name} uses a missing table: {fk.table} or {fk.target}")
        check(len(fk.columns) == len(fk.target_columns) and set(fk.columns) <= local.columns.keys()
              and set(fk.target_columns) <= target.columns.keys(), f"Invalid columns on FK {fk.table}.{fk.name}")
        # PostgreSQL accepts the target only when it has an appropriate unique key.
        # Never add uniqueness silently to make a formerly nonunique reference work.
        unique_keys = [target.primary, *(cols for _, cols, unique in target.indexes if unique)]
        check(any(set(key) == set(fk.target_columns) for key in unique_keys),
              f"FK {fk.name} references nonunique columns on {fk.target}")
        for left, right in zip(fk.columns, fk.target_columns):
            left_type, right_type = local.columns[left][0], target.columns[right][0]
            check(left_type == right_type or (left_type.startswith("varchar(") and right_type.startswith("varchar(")),
                  f"FK {fk.name} has incompatible types: {left_type} -> {right_type}")
        check((fk.table, fk.name) not in seen_keys, f"Duplicate FK name {fk.table}.{fk.name}")
        seen_keys.add((fk.table, fk.name))
        assigned = new_name(fk.name, fk.table + ":foreign", constraint_names[fk.table])
        if assigned != fk.name:
            renamed.append({"kind": "foreign_key", "table": fk.table, "mysql": fk.name, "postgresql": assigned})
        fk_sql.append(
            f"ALTER TABLE {quote(fk.table)} ADD CONSTRAINT {quote(assigned)} "
            f"FOREIGN KEY {columns_sql(fk.columns)} REFERENCES {quote(fk.target)} "
            f"{columns_sql(fk.target_columns)} ON DELETE {fk.actions['DELETE']} ON UPDATE {fk.actions['UPDATE']};"
        )

    sql = (
        "-- PostgreSQL DDL converted from an archived MySQL schema.\n"
        "-- Run one schema file at a time in its own empty schema or database.\n\n"
        + "\n\n".join(statements + index_sql + fk_sql) + "\n"
    )
    stats = {"tables": len(tables), "primary_keys": sum(bool(t.primary) for t in tables.values()),
             "indexes": len(index_sql), "foreign_keys": len(fk_sql), "renamed_identifiers": renamed}
    return sql.encode("utf-8"), stats


def safe_sql_path(row):
    path = row["sql_path"]
    parts = PurePosixPath(path).parts
    check(path.startswith(SQL_PREFIX) and len(parts) == 6
          and parts[3] == row["object_model"] and parts[4] == "SQL_Solutions"
          and parts[5].endswith(".sql") and all(PATH_PART.fullmatch(part) for part in parts)
          and "/".join(parts) == path, f"Invalid cohort SQL path: {path!r}")
    return path


def run(source, destination):
    source, destination = source.resolve(), destination.resolve()
    check(not destination.exists(), f"Output folder already exists; choose a new one: {destination}")
    check(source.is_dir() or (source.is_file() and source.suffix.lower() == ".zip"),
          f"Input must be an extracted cohort folder or the original cohort ZIP: {source}")
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(source)) if source.is_file() else None
        if archive is not None:
            archive_names = archive.namelist()
            check(len(archive_names) == len(set(archive_names)), "Input ZIP has duplicate paths")
        def read(path):
            return archive.read(path) if archive is not None else (source / path).read_bytes()

        rows = list(csv.DictReader(io.StringIO(read("candidates.csv").decode("utf-8-sig"))))
        check(Counter(row["scenario"] for row in rows) == EXPECTED, "The cohort must contain 728 native and 11,772 archive rows")
        check(len({r["schema_id"] for r in rows}) == len(rows), "Repeated cohort schema ID")
        check(len({safe_sql_path(r).casefold() for r in rows}) == len(rows), "Repeated cohort SQL path")

        destination.mkdir(parents=True)
        successes, failures, totals = 0, 0, Counter()
        renames_path = destination / "renamed_identifiers.jsonl"
        fields = ("schema_id", "object_model", "source_sql_path", "source_sql_sha256_lf",
                  "postgresql_sql_path", "postgresql_sql_sha256", "status", "error",
                  "tables", "primary_keys", "indexes", "foreign_keys", "renamed_identifiers")
        with (destination / "conversion_report.csv").open("x", encoding="utf-8-sig", newline="") as report, \
             renames_path.open("x", encoding="utf-8") as renames:
            writer = csv.DictWriter(report, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if row["scenario"] != "archive":
                    continue
                relative = row["sql_path"]
                item = {"schema_id": row["schema_id"], "object_model": row["object_model"],
                        "source_sql_path": relative, "source_sql_sha256_lf": row["sql_sha256_lf"]}
                try:
                    raw = read(relative)
                    check(hash_bytes(lf(raw)) == row["sql_sha256_lf"], "Input SQL checksum does not match candidates.csv")
                    pg, stats = convert(raw.decode("utf-8-sig"))
                    out = destination / relative
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with out.open("xb") as f:
                        f.write(pg)
                    renamed = stats.pop("renamed_identifiers")
                    for rename in renamed:
                        renames.write(json.dumps({"schema_id": row["schema_id"], **rename}, ensure_ascii=False) + "\n")
                    item.update(postgresql_sql_path=relative, postgresql_sql_sha256=hash_bytes(pg),
                                status="converted", renamed_identifiers=len(renamed), **stats)
                    successes += 1
                    totals.update(stats)
                    totals["renamed_identifiers"] += len(renamed)
                except (OSError, ValueError, KeyError, BadZipFile) as exc:
                    item.update(status="failed", error=str(exc))
                    failures += 1
                writer.writerow(item)

        for name in ("selection_manifest.json", "provenance/LICENSE-Archive-CC-BY-NC-4.0.md",
                     "provenance/UPSTREAM_DATASET_README.md"):
            try:
                notice = read(name)
            except (OSError, KeyError):
                continue
            out = destination / "source_notices" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(notice)

    summary = {"converter": VERSION, "source": str(source), "output": str(destination),
               "archive_expected": EXPECTED["archive"], "converted": successes, "failed": failures,
               "native_skipped": EXPECTED["native"], "objects": dict(totals),
               "report": "conversion_report.csv", "identifier_changes": "renamed_identifiers.jsonl",
               "scope": "DDL conversion only; this script does not import or benchmark schemas."}
    (destination / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Archive SQL converted: {successes:,} / {EXPECTED['archive']:,}")
    print(f"Failed: {failures:,}")
    print(f"Native PostgreSQL files skipped: {EXPECTED['native']:,}")
    print(f"Report: {destination / 'conversion_report.csv'}")
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted cohort folder or original cohort ZIP")
    parser.add_argument("destination", type=Path, help="new directory for converted SQL and reports")
    args = parser.parse_args()
    try:
        return run(args.source, args.destination)
    except (OSError, ValueError, KeyError, BadZipFile, csv.Error) as exc:
        print(f"Cannot convert cohort: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
