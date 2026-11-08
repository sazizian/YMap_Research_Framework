"""Resolve one measured study at a time from the repository's dataset root."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DATASET_NAMES = ("native", "archive", "paper-reference")
_COUNT_KEYS = ("measured_candidates", "source_systems", "measurement_rows")
_MEASURED_LABEL_SOURCES = {"postgresql_measured", "imported_measured"}
_HASH = re.compile(r"[0-9a-fA-F]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetInputs:
    name: str
    dataset_root: Path
    manifest_path: Path
    candidates_path: Path
    measurements_path: Path
    scope: str
    workload_id: str
    environment_id: str
    label_source: str
    units_confirmed: bool
    expected_counts: dict[str, int]
    provenance: dict

    def evaluation_config(self) -> dict:
        """Exact filters to apply after loading any base evaluation settings."""
        return {
            "mode": "research",
            "workload_id": self.workload_id,
            "environment_id": self.environment_id,
            "allowed_label_sources": [self.label_source],
        }

    def input_hashes(self) -> dict[str, str]:
        return {
            "candidates": file_sha256(self.candidates_path),
            "measurements": file_sha256(self.measurements_path),
            "dataset_manifest": file_sha256(self.manifest_path),
        }

    def metadata(self) -> dict:
        """Record scope and provenance alongside outputs and observed hashes."""
        return {
            "dataset": self.name,
            "scope": self.scope,
            "candidates": self.candidates_path.relative_to(self.dataset_root).as_posix(),
            "measurements": self.measurements_path.relative_to(self.dataset_root).as_posix(),
            "workload_id": self.workload_id,
            "environment_id": self.environment_id,
            "label_source": self.label_source,
            "units_confirmed": self.units_confirmed,
            "expected_counts": dict(self.expected_counts),
            "provenance": dict(self.provenance),
        }

    def check_alignment(self, data) -> dict[str, int]:
        """Check the declared scenario and counts against load_dataset's output."""
        for column, expected in (("workload_id", self.workload_id), ("environment_id", self.environment_id),
                                 ("label_source", self.label_source)):
            if set(data[column]) != {expected}:
                raise ValueError(f"Dataset {self.name!r} has an unexpected {column}")
        counts = {
            "measured_candidates": len(data),
            "source_systems": int(data["object_model"].nunique()),
            "measurement_rows": int(data["measurement_seed_count"].sum()),
        }
        for key, expected in self.expected_counts.items():
            if counts[key] != expected:
                raise ValueError(f"Dataset {self.name!r} expected {expected} {key}, found {counts[key]}")
        return counts


def _mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Dataset manifest field {field!r} must be a mapping")
    return value


def _string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset manifest field {field!r} must be a nonempty string")
    return value.strip()


def _input_path(root: Path, entry: dict, key: str) -> Path:
    path = Path(_string(entry.get(key), key))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Dataset manifest field {key!r} must be a relative path within the dataset root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Dataset manifest field {key!r} resolves outside the dataset root")
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset {key} file not found: {resolved}. Check the dataset checkout or --dataset-root.")
    return resolved


def _read_manifest(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid dataset manifest: {path}") from error
    document = _mapping(value, "document")
    if type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("Dataset manifest must specify version: 1")
    return document


def resolve_dataset(project_root: Path | str, name: str = "native", *,
                    dataset_root: Path | str | None = None,
                    manifest_path: Path | str | None = None) -> DatasetInputs:
    """Resolve a registered dataset; manuscript inputs require an explicit manifest.

    Built-in datasets use config/datasets.yaml and Dataset/YMap by default.
    Override paths are relative to the working directory; manifest paths inside
    an entry always resolve within the chosen dataset root.
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASET_NAMES)}")
    reference = name == "paper-reference"
    if reference and manifest_path is None:
        raise ValueError("paper-reference requires an explicit dataset manifest and documented measured inputs")
    if not reference and manifest_path is not None:
        raise ValueError("A custom dataset manifest is only supported for paper-reference")

    project = Path(project_root).expanduser().resolve()
    manifest = (Path(manifest_path).expanduser() if manifest_path is not None else project / "config" / "datasets.yaml").resolve()
    document = _read_manifest(manifest)
    root = (Path(dataset_root).expanduser() if dataset_root is not None else
            project / _string(document.get("dataset_root", "Dataset/YMap"), "dataset_root")).resolve()
    entry = _mapping(_mapping(document.get("datasets"), "datasets").get(name), f"datasets.{name}")

    label_source = _string(entry.get("label_source"), "label_source")
    if label_source not in _MEASURED_LABEL_SOURCES:
        raise ValueError(f"Dataset {name!r} requires a measured label source")
    units_confirmed = entry.get("units_confirmed")
    if type(units_confirmed) is not bool or (reference and not units_confirmed):
        raise ValueError(f"Dataset {name!r} must declare units_confirmed" + (": true" if reference else " as a boolean"))

    counts = _mapping(entry.get("counts"), "counts")
    if set(counts) != set(_COUNT_KEYS) or any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError(f"Dataset manifest counts must declare positive integers for {', '.join(_COUNT_KEYS)}")
    if not counts["source_systems"] <= counts["measured_candidates"] <= counts["measurement_rows"]:
        raise ValueError("Dataset counts must satisfy source_systems <= measured_candidates <= measurement_rows")

    provenance = _mapping(entry.get("provenance"), "provenance")
    required_provenance = ("source", "protocol", "citation") if reference else ("source", "protocol")
    for key in required_provenance:
        _string(provenance.get(key), f"provenance.{key}")
    candidates = _input_path(root, entry, "candidates")
    measurements = _input_path(root, entry, "measurements")
    if reference:
        hashes = _mapping(provenance.get("input_sha256"), "provenance.input_sha256")
        for key, path in (("candidates", candidates), ("measurements", measurements)):
            value = hashes.get(key)
            if not isinstance(value, str) or not _HASH.fullmatch(value):
                raise ValueError(f"paper-reference requires a SHA-256 for {key}")
            if file_sha256(path) != value.lower():
                raise ValueError(f"paper-reference {key} SHA-256 does not match the manifest")

    return DatasetInputs(
        name, root, manifest, candidates, measurements, _string(entry.get("scope"), "scope"),
        _string(entry.get("workload_id"), "workload_id"), _string(entry.get("environment_id"), "environment_id"),
        label_source, units_confirmed, dict(counts), dict(provenance),
    )
