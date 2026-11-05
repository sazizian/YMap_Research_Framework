from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["insert_ms", "query_ms", "storage_mb"]
RESEARCH_SOURCES = {"postgresql_measured", "imported_measured"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_output_schema_ids(records):
    """Return IDs whose DDL filenames are safe and unique on Linux and Windows."""
    if not records:
        raise ValueError("No candidates")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    identifiers = []
    for row in records:
        identifier = row.get("schema_id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) or identifier in {".", ".."}:
            raise ValueError("Candidate schema_id must be a safe nonempty filename identifier")
        if identifier.split(".", 1)[0].upper() in reserved:
            raise ValueError("Candidate schema_id cannot use a reserved Windows filename")
        identifiers.append(identifier)
    if len({identifier.casefold() for identifier in identifiers}) != len(identifiers):
        raise ValueError("Candidate schema_id values must produce unique filenames, including case-insensitive filesystems")
    return identifiers


def json_ready(value):
    """Represent explicitly undefined aggregate metrics as JSON null."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "sqlglot", "torch", "transformers", "psycopg", "xgboost"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def validate_dataset(candidates_path, measurements_path, mode="research"):
    if mode not in {"research", "demo"}:
        raise ValueError("mode must be research or demo")
    records = read_jsonl(candidates_path)
    if not records:
        raise ValueError("Candidate file is empty")
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Every candidate must be a JSON object")
        for field in ("schema_id", "object_model"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Candidate {field} must be a nonempty string")
        if not isinstance(row.get("features"), dict):
            raise ValueError("Candidate features must be a JSON object")
    ids = [row["schema_id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate schema_id values must be unique")
    by_id = {row["schema_id"]: row for row in records}
    feature_names = sorted(records[0].get("features", {}))
    if not feature_names:
        raise ValueError("Candidates need numeric features")
    forbidden = [name for name in feature_names if name.lower() in {*TARGETS, "schema_id", "object_model", "seed", "insert_latency_ms", "query_latency_ms", "memory_mb", "storage_bytes", "label", "target", "label_source", "workload_id", "environment_id"} or name.lower().startswith(("true_", "measured_", "target_"))]
    if forbidden:
        raise ValueError(f"Measured outcomes cannot be prediction features: {forbidden}")
    for row in records:
        if sorted(row.get("features", {})) != feature_names:
            raise ValueError("Feature definitions must be identical for every schema")
        values = np.asarray([row["features"][key] for key in feature_names], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Features must be finite")
    string_fields = {key: str for key in ("schema_id", "object_model", "label_source", "workload_id", "environment_id")}
    measurements = pd.read_csv(measurements_path, dtype=string_fields, keep_default_na=False)
    required = {"schema_id", "object_model", "label_source", "workload_id", "environment_id", "seed", *TARGETS}
    missing = required - set(measurements)
    if missing:
        raise ValueError(f"Measurements are missing columns: {sorted(missing)}")
    if measurements.empty:
        raise ValueError("Measurements file is empty")
    if measurements[list(required)].isna().any().any():
        raise ValueError("Required measurement fields cannot be missing")
    for field in string_fields:
        if not measurements[field].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Measurement {field} must contain nonempty strings")
    values = measurements[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Measured objectives must be finite and non-negative")
    if mode == "research" and not set(measurements.label_source).issubset(RESEARCH_SOURCES):
        raise ValueError("Research mode rejects demo/synthetic/unknown label sources. Supply actual measurements and provenance.")
    if measurements.label_source.nunique() != 1:
        raise ValueError("Use a single documented label source in each training or evaluation run")
    if measurements.duplicated(["schema_id", "workload_id", "environment_id", "seed"]).any():
        raise ValueError("Duplicate schema/scenario/seed measurements")
    unknown = set(measurements.schema_id) - set(ids)
    if unknown:
        raise ValueError(f"Measurements refer to unknown schemas: {sorted(unknown)[:5]}")
    for row in measurements.itertuples(index=False):
        if by_id[row.schema_id]["object_model"] != row.object_model:
            raise ValueError(f"object_model mismatch for {row.schema_id}")
    if measurements.workload_id.nunique() != 1 or measurements.environment_id.nunique() != 1:
        raise ValueError("Evaluate each workload/environment scenario separately; filter the CSV explicitly.")
    return {
        "mode": mode,
        "candidate_count": len(records),
        "measured_unique_schemas": int(measurements.schema_id.nunique()),
        "measurement_rows": len(measurements),
        "source_systems": int(measurements.object_model.nunique()),
        "per_system": measurements.groupby("object_model").schema_id.nunique().to_dict(),
        "label_sources": sorted(measurements.label_source.unique().tolist()),
        "workload_id": str(measurements.workload_id.iloc[0]),
        "environment_id": str(measurements.environment_id.iloc[0]),
        "feature_names": feature_names,
        "candidate_sha256": sha256_file(candidates_path),
        "measurements_sha256": sha256_file(measurements_path),
        "provenance_note": "Source labels are declarations; verify raw logs and origin. This validator cannot establish measurement authenticity.",
    }


def aligned_training_data(candidates_path, measurements_path, mode="research"):
    report = validate_dataset(candidates_path, measurements_path, mode)
    records = read_jsonl(candidates_path)
    rows = pd.read_csv(measurements_path, dtype={"schema_id": str, "object_model": str}, keep_default_na=False)
    aggregate = rows.groupby(["schema_id", "object_model"], sort=True)[TARGETS].mean().reset_index()
    by_id = {row["schema_id"]: row for row in records}
    selected = [by_id[sid] for sid in aggregate.schema_id]
    names = report["feature_names"]
    X = np.asarray([[row["features"][name] for name in names] for row in selected], dtype=float)
    return selected, X, aggregate[TARGETS].to_numpy(dtype=float), report
