"""Prediction views of the measured native schemas, without resynthesis."""
from __future__ import annotations

from collections.abc import Iterable

from .paper_schema import _paper_features, _paper_graph
from .synthesis import canonical_candidate


def native_prediction_records(records: Iterable[dict]) -> list[dict]:
    """Attach the two structural paper views to verified native candidates.

    Validate the existing DDL and mapping against the native compiler, then use
    its physical tables as measured. Applying the paper synthesis rules here
    could change a primary key or nullability and pair measurements with the
    wrong physical schema. This adapter does not produce canonical paper DDL
    for the text encoders.
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
        result.append({
            **record,
            "paper_graph": _paper_graph(original, tables),
            "paper_features": _paper_features(original, tables),
        })
    return result
