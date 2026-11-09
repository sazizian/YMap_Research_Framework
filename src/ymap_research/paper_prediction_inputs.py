"""Prediction views of the measured native schemas, without resynthesis."""
from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from .paper_schema import _canonical_sql, _paper_features, _paper_graph, _table_order
from .synthesis import canonical_candidate


NATIVE_TEXT_SOURCE = "native_measured_constraints_canonical_order_v1"


def _native_text_view(record: dict, tables: list[dict]) -> str:
    """Canonically order the measured constraints for text encoders only."""
    by_name = {table["name"]: table for table in tables}
    ordered = []
    for name in _table_order(by_name):
        table = by_name[name]
        fk_columns = {column for fk in table["foreign_keys"] for column in fk["columns"]}
        columns = sorted(table["columns"], key=lambda col: (
            0 if col["primary_key"] else 1 if col["name"] in fk_columns else 2,
            col["name"],
        ))
        # Keep the primary key list in its original order, even when the
        # declarations are sorted. Preserve FK list order (constraint names),
        # FK member order, checks, and unique keys exactly as measured.
        ordered.append({**table, "columns": columns})
    return _canonical_sql(record, ordered)


def native_prediction_records(records: Iterable[dict], *, include_text: bool = False) -> list[dict]:
    """Attach structural views and optional text for measured native candidates.

    Validate the existing DDL and mapping against the native compiler, then use
    its physical tables as measured. Applying the paper synthesis rules here
    could change a primary key or nullability and pair measurements with the
    wrong physical schema. With include_text, a separate DDL representation
    uses canonical ordering without changing those measured constraints or
    replacing the original executable DDL. This describes the native study.
    """
    result = []
    for record in records:
        original = canonical_candidate(record)
        tables = [
            {
                **table,
                "primary_key": [
                    column["name"] for column in table["columns"]
                    if column["primary_key"]
                ],
            }
            for table in original["tables"]
        ]
        adapted = {
            **record,
            "paper_graph": _paper_graph(original, tables),
            "paper_features": _paper_features(original, tables),
        }
        if include_text:
            ddl = _native_text_view(original, tables)
            adapted.update(
                paper_ddl=ddl,
                paper_ddl_sha256=sha256(ddl.encode("utf-8")).hexdigest(),
                paper_ddl_source=NATIVE_TEXT_SOURCE,
            )
        result.append(adapted)
    return result
