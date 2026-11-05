#!/usr/bin/env python3
"""Read-only audit of an exported Y-Map PostgreSQL measurement bundle.

Reads exported artifacts only; does not connect to PostgreSQL, modify inputs,
discard observations, or substitute labels. Requires numpy, pandas, scipy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations, groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OBJECTIVES = ["insert_ms", "query_ms", "storage_mb"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def nondominated(values: np.ndarray) -> np.ndarray:
    """All objectives minimized; retain duplicate nondominated observations."""
    dominates = np.all(values[:, None, :] <= values[None, :, :], axis=2)
    dominates &= np.any(values[:, None, :] < values[None, :, :], axis=2)
    return ~np.any(dominates, axis=0)


def distribution(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {"min": float(x.min()), "p05": float(np.quantile(x, .05)),
            "p25": float(np.quantile(x, .25)), "median": float(np.median(x)),
            "p75": float(np.quantile(x, .75)), "p95": float(np.quantile(x, .95)),
            "p99": float(np.quantile(x, .99)), "max": float(x.max()),
            "mean": float(x.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Root containing profile.yaml and measurements/")
    parser.add_argument("--output", type=Path, required=True, help="New output directory outside the extracted input")
    args = parser.parse_args()
    source = args.input.resolve()
    measurements = source / "measurements"
    output = args.output.resolve()
    if output == source or output.is_relative_to(source):
        parser.error("Output must be outside the complete source input directory.")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (measurements / "raw_measurements.jsonl").open()]
    raw = pd.DataFrame([{k: v for k, v in row.items()
                         if k not in ("query_times_ms", "query_results")} for row in rows])
    summary = pd.read_csv(measurements / "measurements.csv")
    measured = raw.loc[~raw.warmup].copy()
    candidate_order = list(dict.fromkeys(raw.schema_id))
    source_hashes = {str(path.relative_to(source)): digest(path) for path in
                     [source / "profile.yaml", measurements / "raw_measurements.jsonl",
                      measurements / "measurements.csv", measurements / "environment.json",
                      measurements / "workload.json"]}
    grouped = raw.groupby("schema_id", sort=False)
    structure_errors = []
    for sid, group in grouped:
        if (len(group) != 6 or int(group.warmup.sum()) != 1 or
                sorted(group.loc[~group.warmup, "repeat"].tolist()) != list(range(5)) or
                group.loc[group.warmup, "repeat"].tolist() != [-1]):
            structure_errors.append(sid)
    medians = measured.groupby("schema_id")[OBJECTIVES].median()
    aligned = summary.set_index("schema_id").join(medians, rsuffix="_raw_median")
    median_errors = {metric: float(np.max(np.abs(aligned[metric] - aligned[metric + "_raw_median"])))
                     for metric in OBJECTIVES}
    per_candidate = []
    for sid, group in measured.groupby("schema_id", sort=False):
        record = {"schema_id": sid, "object_model": group.object_model.iloc[0]}
        for metric in OBJECTIVES:
            x = group[metric].to_numpy()
            record[metric + "_cv"] = float(np.std(x, ddof=1) / np.mean(x))
            record[metric + "_max_min_ratio"] = float(x.max() / x.min())
            record[metric + "_iqr_over_median"] = float((np.quantile(x, .75) - np.quantile(x, .25)) / np.median(x))
        per_candidate.append(record)
    variability = pd.DataFrame(per_candidate)
    variability.to_csv(output / "candidate_repeat_variability.csv", index=False)

    artifact_errors, fixture_errors, missing_query_errors = [], [], []
    expected_by_id, query_names_by_id, systems, lifecycles = {}, {}, {}, []
    artifacts = ["fixture.json", "queries.json", "expected_query_results.json",
                 "schema_lifecycle.json", "candidate.json", "schema.sql"]
    candidate_dirs = list((measurements / "candidates").iterdir())
    for folder in candidate_dirs:
        missing = [name for name in artifacts if not (folder / name).is_file()]
        if missing:
            artifact_errors.append({"schema_id": folder.name, "missing": missing})
            continue
        candidate = json.loads((folder / "candidate.json").read_text())
        fixture = json.loads((folder / "fixture.json").read_text())
        queries = json.loads((folder / "queries.json").read_text())
        expected = json.loads((folder / "expected_query_results.json").read_text())
        expected_by_id[folder.name] = expected
        query_names_by_id[folder.name] = {q["name"] for q in queries}
        actual_counts = Counter(obj["class"] for obj in fixture["objects"])
        if dict(actual_counts) != fixture["rows_per_class"]:
            fixture_errors.append(folder.name)
        model = candidate["object_model"]
        data = systems.setdefault(model, {"candidates": 0, "fixture_hashes": set(),
            "oracle_hashes": set(), "query_name_hashes": set(), "realization_ids": set(),
            "logical_classes": len(actual_counts), "fixture_objects": len(fixture["objects"]),
            "logical_queries": len(queries), "queries_by_type": dict(Counter(q["name"].split(":")[0] for q in queries))})
        data["candidates"] += 1
        data["fixture_hashes"].add(json_digest(fixture))
        data["oracle_hashes"].add(json_digest(expected))
        data["query_name_hashes"].add(json_digest(sorted(query_names_by_id[folder.name])))
        data["realization_ids"].add(fixture["realization_id"])
        lifecycles.append(json.loads((folder / "schema_lifecycle.json").read_text()))

    oracle_checks = 0
    oracle_errors = []
    query_records = []
    query_row_counts = {}
    for row in rows:
        sid = row["schema_id"]
        expected = expected_by_id.get(sid, {})
        if set(row["query_results"]) != query_names_by_id.get(sid, set()) or set(row["query_times_ms"]) != query_names_by_id.get(sid, set()):
            missing_query_errors.append({"schema_id": sid, "repeat": row["repeat"]})
        for name, result in row["query_results"].items():
            oracle_checks += 1
            if result != expected.get(name):
                oracle_errors.append({"schema_id": sid, "repeat": row["repeat"], "query": name})
            query_row_counts.setdefault(name.split(":")[0], set()).add(result["rows"])
        if not row["warmup"]:
            for name, duration in row["query_times_ms"].items():
                query_records.append({"schema_id": sid, "object_model": row["object_model"],
                    "repeat": row["repeat"], "query": name, "query_type": name.split(":")[0], "ms": duration})

    repeat_frontiers = {}
    repeat_rank_correlations = {}
    for model, group in summary.groupby("object_model"):
        ids = group.schema_id.to_numpy()
        median_set = set(ids[nondominated(group[OBJECTIVES].to_numpy())])
        trials = []
        for rep in range(5):
            values = measured[(measured.object_model == model) & (measured.repeat == rep)].set_index("schema_id").loc[ids]
            repeat_set = set(ids[nondominated(values[OBJECTIVES].to_numpy())])
            intersection = len(median_set & repeat_set)
            trials.append({"repeat": rep, "frontier_count": len(repeat_set),
                "jaccard_with_median_frontier": intersection / len(median_set | repeat_set),
                "f1_with_median_frontier": 2 * intersection / (len(median_set) + len(repeat_set))})
        repeat_frontiers[model] = {"candidate_count": len(ids), "median_frontier_count": len(median_set), "repetitions": trials}
        model_measured = measured[measured.object_model == model]
        correlations = {}
        for metric in OBJECTIVES:
            pivot = model_measured.pivot(index="schema_id", columns="repeat", values=metric)
            pairs = [{"repeat_a": i, "repeat_b": j, "spearman_rho": float(spearmanr(pivot[i], pivot[j]).statistic)}
                     for i, j in combinations(range(5), 2)]
            correlations[metric] = {"pair_correlations": pairs,
                "median_rho": float(np.median([item["spearman_rho"] for item in pairs]))}
        repeat_rank_correlations[model] = correlations

    storage_changes = [{"schema_id": sid, "bytes_by_repeat": group.sort_values("repeat").storage_bytes.tolist()}
                       for sid, group in measured.groupby("schema_id") if group.storage_bytes.nunique() > 1]
    repeat_order = {}
    for metric in OBJECTIVES[:2]:
        pivot = measured.pivot(index="schema_id", columns="repeat", values=metric)
        repeat_order[metric] = {"median_last_over_first_ratio": float(np.median(pivot[4] / pivot[0])),
            "fraction_last_greater_than_first": float(np.mean(pivot[4] > pivot[0])),
            "mean_by_repeat": {str(i): float(pivot[i].mean()) for i in range(5)}}
    for data in systems.values():
        for key in ["fixture_hashes", "oracle_hashes", "query_name_hashes", "realization_ids"]:
            data["unique_" + key] = len(data.pop(key))

    query_frame = pd.DataFrame(query_records)
    report = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source), "source_sha256": source_hashes,
        "audit_scope": "Read-only checks of exported artifacts; no database connection or new profiling.",
        "counts": {"summary_rows": len(summary), "unique_schema_ids": int(raw.schema_id.nunique()),
            "candidate_directories": len(candidate_dirs), "source_systems": int(summary.object_model.nunique()),
            "raw_records": len(raw), "warmup_records": int(raw.warmup.sum()), "measured_records": len(measured),
            "recorded_query_result_checks_including_warmup": oracle_checks,
            "measured_individual_query_timings": len(query_frame)},
        "consistency": {"repeat_structure_error_ids": structure_errors,
            "summary_vs_nonwarmup_median_max_absolute_difference": median_errors,
            "query_mean_max_absolute_difference": float(max(abs(row["query_ms"] - np.mean(list(row["query_times_ms"].values()))) for row in rows)),
            "storage_mb_vs_decimal_bytes_max_absolute_difference": float(max(abs(row["storage_mb"] - row["storage_bytes"] / 1e6) for row in rows)),
            "nonpositive_objective_records": int(((raw[OBJECTIVES] <= 0).any(axis=1)).sum()),
            "missing_candidate_artifacts": artifact_errors, "fixture_count_mismatch_ids": fixture_errors,
            "oracle_mismatch_records": oracle_errors, "query_name_mismatch_records": missing_query_errors,
            "lifecycle_created_by_this_run_count": sum(item.get("created_by_this_run") is True for item in lifecycles),
            "lifecycle_dropped_count": sum(item.get("cleanup_status") == "dropped" for item in lifecycles),
            "query_rows_by_type": {k: sorted(v) for k, v in query_row_counts.items()}},
        "objective_summary_distributions": {metric: distribution(summary[metric]) for metric in OBJECTIVES},
        "repeat_variability": {metric: {"sample_cv": distribution(variability[metric + "_cv"]),
            "candidates_with_cv_above_0_20": int((variability[metric + "_cv"] > .2).sum()),
            "max_min_ratio": distribution(variability[metric + "_max_min_ratio"]),
            "iqr_over_median": distribution(variability[metric + "_iqr_over_median"])} for metric in OBJECTIVES},
        "storage_changes": storage_changes,
        "systems": systems,
        "repeat_frontier_stability": repeat_frontiers,
        "within_system_cross_repeat_rank_correlations": repeat_rank_correlations,
        "order_diagnostics": {"raw_candidate_blocks": sum(1 for _ in groupby(raw.schema_id)),
            "candidate_order_system_blocks": [{"system": k, "candidates": len(list(g))} for k, g in groupby(summary.object_model)],
            "last_vs_first_measured_repeat": repeat_order,
            "interpretation": "File order is observed; timestamps/randomized block identifiers are absent. Order diagnostics do not establish causation."},
        "query_timing_by_type": {name: distribution(group.ms) for name, group in query_frame.groupby("query_type")},
        "largest_individual_query_times": query_frame.nlargest(10, "ms").to_dict("records"),
        "profile_yaml": (source / "profile.yaml").read_text(),
        "workload": json.loads((measurements / "workload.json").read_text()),
        "environment": json.loads((measurements / "environment.json").read_text()),
        "evidence_limits": [
            "Exported records and oracle files agree; live database execution and timer implementation were not independently verified.",
            "One seed, one fixed workload ID, one environment ID, 100 objects per declared concrete class; this is a new finite executable pilot.",
            "Only 7 to 33 fixed logical queries per candidate: point lookups, ordered limited scans, and association joins; no aggregate-query family.",
            "All scans/joins return 100 rows under LIMIT 100; oracle agreement does not prove full-content equivalence or mutation correctness.",
            "No retained live per-table PostgreSQL row-count snapshot, full fetched result sets, server hardware/load characterization, or run timestamps.",
            "Hospital/ERP fixture models have 11/12 logical classes, not the original manuscript's 35/42-class subjects.",
            "Query timing includes client/network/result fetching; insert timing is for the complete logical-fixture transaction.",
            "Per-repeat frontier comparisons are sensitivity diagnostics, not an alternative ground truth or an independent replication."
        ],
        "recommendation": "Diagnose representation/label relationships and timing stability before rerunning all profiling. Preserve all observations. Use a targeted controlled randomized/interleaved rerun to assess timing reliability; design any larger journal experiment after diagnosis."
    }
    (output / "measurement_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output / "measurement_audit.json"), "counts": report["counts"],
                      "oracle_mismatches": len(oracle_errors), "repeat_structure_errors": len(structure_errors)}, indent=2))


if __name__ == "__main__":
    main()
