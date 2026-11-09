"""Score prepared candidate records using a saved, calibrated paper ensemble.

This path uses no measurement file or schema generator. Explicit Stage B views
are validated as model inputs; their correspondence to external source DDL and
mapping decisions remains the responsibility of the candidate exporter.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import tempfile

import numpy as np

from .common import TARGETS, environment, sha256_file, write_json
from .paper_model import TEXT_VIEWS, PaperEnsemble
from .paper_schema import paper_features, paper_graph


_GRAPH_FLAGS = ("primary_key", "foreign_key", "nullable", "unique", "discriminator")
_GRAPH_DIRECTIONS = {
    "hasColumn": ("table", "column"),
    "pkOf": ("column", "table"),
    "fkOf": ("column", "table"),
    "references": ("column", "table"),
    "inheritsFrom": ("table", "table"),
}


def _checked_graph(record):
    graph = paper_graph(record)
    node_types = {}
    for node in graph["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"].strip():
            raise ValueError("paper_graph nodes must have nonempty string IDs")
        key, kind = node["id"], node.get("type")
        if key in node_types or kind not in ("table", "column"):
            raise ValueError("paper_graph needs unique node IDs and table/column types")
        node_types[key] = kind
        if kind == "table":
            count = node.get("column_count")
            if type(count) is not int or count < 0:
                raise ValueError("paper_graph table nodes need a nonnegative integer column_count")
        elif (not isinstance(node.get("data_type"), str) or not node["data_type"].strip()
              or any(type(node.get(flag)) is not bool for flag in _GRAPH_FLAGS)):
            raise ValueError("paper_graph column nodes need a data_type and all five Boolean flags")
    if "table" not in node_types.values():
        raise ValueError("paper_graph needs at least one table")
    seen = set()
    for edge in graph["edges"]:
        source, target = edge.get("source"), edge.get("target")
        relation = edge["type"]
        if (not isinstance(source, str) or not isinstance(target, str)
                or (node_types.get(source), node_types.get(target)) != _GRAPH_DIRECTIONS[relation]):
            raise ValueError(f"paper_graph has missing or wrongly typed endpoints for {relation}")
        key = (relation, source, target)
        if key in seen:
            raise ValueError("paper_graph has a duplicate typed edge")
        seen.add(key)
    return graph


def _records(stream, views, digest):
    seen = set()
    text_used = any(view in TEXT_VIEWS for view in views)
    for line_no, raw in enumerate(stream, 1):
        digest.update(raw)
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in ("schema_id", "object_model")
            ):
                raise ValueError("schema_id and object_model must be nonempty strings")
            identifier = record["schema_id"]
            if identifier in seen:
                raise ValueError(f"duplicate schema_id {identifier!r}")
            seen.add(identifier)
            selected = {key: record[key] for key in ("schema_id", "object_model")}
            if text_used:
                ddl = record.get("paper_ddl")
                if not isinstance(ddl, str) or not ddl.strip() or "\x00" in ddl:
                    raise ValueError("text views require explicit nonempty canonical Stage B paper_ddl")
                fingerprint = hashlib.sha256(ddl.encode("utf-8")).hexdigest()
                if record.get("paper_ddl_sha256", fingerprint) != fingerprint:
                    raise ValueError("paper_ddl_sha256 does not match the supplied DDL")
                selected["paper_ddl"] = ddl
            if "graph" in views:
                if "paper_graph" not in record:
                    raise ValueError("the graph view requires explicit Stage B paper_graph")
                selected["paper_graph"] = _checked_graph(record)
            if "symbolic" in views:
                if "paper_features" not in record:
                    raise ValueError("the symbolic view requires all twelve explicit paper_features")
                selected["paper_features"] = paper_features(record)
        except (ValueError, UnicodeError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid candidate JSONL line {line_no}: {exc}") from exc
        yield selected


def _calibration_factors(manifest):
    raw = manifest.get("calibration_factors")
    if not isinstance(raw, list) or len(raw) != len(TARGETS):
        raise ValueError("fit_manifest.json must contain one calibration factor per objective")
    message = "Calibration factors must be finite nonnegative numbers or null"
    if any(value is not None and type(value) not in (float, int) for value in raw):
        raise ValueError(message)
    try:
        numeric = [float(value) for value in raw if value is not None]
        if any(not np.isfinite(value) or value < 0 for value in numeric):
            raise ValueError(message)
        return np.asarray([np.nan if value is None else value for value in raw], dtype=np.float64)
    except OverflowError as exc:
        raise ValueError(message) from exc


def _estimate(members, member_count, row_count, factors):
    members = np.asarray(members, dtype=np.float64)
    if (members.shape != (member_count, row_count, len(TARGETS))
            or not np.isfinite(members).all() or np.any(members <= 0)):
        raise ValueError("Paper ensemble must return one finite positive prediction per member, candidate and objective")
    # Scale before reducing to avoid overflow in the sum or squared differences.
    peak = members.max(axis=0)
    scaled = members / peak
    means = scaled.mean(axis=0) * peak
    if member_count == 1:
        spread = np.full_like(means, np.nan)
    else:
        spread = scaled.std(axis=0, ddof=1) * peak
    with np.errstate(over="ignore", invalid="ignore"):
        calibrated = spread * factors
    if np.isinf(calibrated).any():
        raise ValueError("Calibrated spread exceeds the numeric range; check the fitted model and calibration")
    return members, means, spread, calibrated


def _config_overrides(config):
    if config is None:
        return {}
    if not isinstance(config, Mapping) or not isinstance(config.get("neural", {}), Mapping):
        raise ValueError("Prediction config must contain a neural settings mapping")
    neural = config.get("neural", {})
    return {key: neural[key] for key in ("device", "checkpoints", "cache_dir") if key in neural}


def predict_paper(candidates_path, model_dir, output_dir, *, config=None, batch_size=None):
    """Score every JSONL record in order; return a provenance-bearing run manifest.

    A saved `paper fit` directory is required. Inference never estimates new
    calibration factors or fits scalers. Optional batching reduces working
    memory but can repeat cold encoder loading when the embedding cache misses.
    """
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("Prediction batch_size must be a positive integer")
    candidates_path, model_dir, out = map(Path, (candidates_path, model_dir, output_dir))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {out}")
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Prepared candidate JSONL was not found: {candidates_path}")

    fit_path = model_dir / "fit_manifest.json"
    fit_bytes = fit_path.read_bytes()
    fit = json.loads(fit_bytes)
    if not isinstance(fit, dict) or fit.get("status") != "fitted":
        raise ValueError("model_dir must point to a fitted paper ensemble and its fit_manifest.json")
    factors = _calibration_factors(fit)
    ensemble_path = model_dir / "ensemble"
    ensemble_hash = sha256_file(ensemble_path / "manifest.json")
    bound_hash = fit.get("ensemble_manifest_sha256")
    if bound_hash is not None and bound_hash != ensemble_hash:
        raise ValueError("Saved ensemble manifest does not match the fit manifest calibration identity")
    ensemble = PaperEnsemble.load(ensemble_path, **_config_overrides(config))
    views, seeds = tuple(ensemble.views), tuple(ensemble.seeds)
    member_hashes = {
        f"member-{index}.npz": sha256_file(ensemble_path / f"member-{index}.npz")
        for index in range(len(seeds))
    }
    bound_members = fit.get("ensemble_member_sha256")
    if bound_members is not None and bound_members != member_hashes:
        raise ValueError("Saved ensemble members do not match the fit manifest calibration identity")

    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    groups, rows = set(), 0
    with tempfile.TemporaryDirectory(prefix=".prediction-", dir=out) as temporary:
        staging = Path(temporary)
        csv_path = staging / "predictions.csv"
        manifest_path = staging / "prediction_manifest.json"
        with candidates_path.open("rb") as source, csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            prefixes = ["mean", "sample_std", "calibrated_std", *(f"member_seed_{seed}" for seed in seeds)]
            writer.writerow(["schema_id", "object_model", *(f"{prefix}_{target}" for prefix in prefixes for target in TARGETS)])
            records = _records(source, views, digest)
            while True:
                batch = list(records if batch_size is None else islice(records, batch_size))
                if not batch:
                    break
                members, means, spread, calibrated = _estimate(
                    ensemble.predict_members(batch), len(seeds), len(batch), factors,
                )
                for i, record in enumerate(batch):
                    groups.add(record["object_model"])
                    values = np.concatenate((means[i], spread[i], calibrated[i], members[:, i, :].ravel()))
                    writer.writerow([
                        record["schema_id"], record["object_model"],
                        *(float(value) if np.isfinite(value) else None for value in values),
                    ])
                rows += len(batch)
        if not rows:
            raise ValueError("Prepared candidate JSONL contains no records")

        report = {
            "status": "predicted",
            "scope": "prediction_only",
            "output": str(out),
            "candidate_count": rows,
            "object_models": sorted(groups),
            "objectives": TARGETS,
            "views": views,
            "seeds": seeds,
            "full_paper_configuration": ensemble.metadata.get("full_paper_configuration") is True,
            "prediction_batch_size": batch_size,
            "predictions": csv_path.name,
            "predictions_sha256": sha256_file(csv_path),
            "input": {"path": str(candidates_path), "sha256": digest.hexdigest()},
            "model": {
                "path": str(model_dir), "fit_manifest_sha256": hashlib.sha256(fit_bytes).hexdigest(),
                "ensemble_manifest_sha256": ensemble_hash, "member_sha256": member_hashes,
                "calibration_bound_to_ensemble_manifest": bound_hash is not None,
                "calibration_bound_to_member_files": bound_members is not None,
                "model_version": ensemble.metadata.get("model_version"),
                "text_encoders": ensemble.metadata.get("text_encoders", {}),
                "training_dataset": fit.get("dataset"), "validation_system": fit.get("validation_system"),
            },
            "software": environment(),
            "calibration_factors": factors,
            "uncertainty": "Sample standard deviation of original-unit predictions and its validation-calibrated counterpart. A blank field is unavailable.",
            "representation_scope": "Caller-supplied Stage B representations; shape and supplied DDL checksum checked, without certifying canonicalization, mapping provenance, or correspondence to source SQL.",
        }
        write_json(manifest_path, report)
        for path in (csv_path, manifest_path):
            if (out / path.name).exists():
                raise FileExistsError(f"Prediction output was created during inference: {out / path.name}")
        for path in (csv_path, manifest_path):
            path.replace(out / path.name)
    # Normalize unavailable factors here as well as in the persisted JSON.
    report["calibration_factors"] = [float(value) if np.isfinite(value) else None for value in factors]
    return report
