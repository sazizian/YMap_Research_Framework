#!/usr/bin/env python3
"""Reconstruct preference regret from saved evaluation artifacts, without fitting.

Only Python's standard library is used. Inputs are read-only. The chosen top-one
schema minimizes recorded preference_scores within the returned set, with exact
ties resolved by the lowest original candidate index. Held-out labels never
choose that schema. They are used only to evaluate its realized utility regret.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

OBJECTIVES = ("insert_ms", "query_ms", "storage_mb")
INPUT_NAMES = ("recommendations.json", "oof_predictions.csv", "fold_audit.json", "per_system_metrics.csv", "evaluation_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite value in {name}")
    return result


def utilities(targets, normalization, weights):
    minimum = [finite(v, "normalization minimum") for v in normalization["minimum"]]
    scale = [finite(v, "normalization scale") for v in normalization["scale"]]
    if len(minimum) != 3 or len(scale) != 3 or any(v <= 0 for v in scale):
        raise ValueError("Recorded normalization must have three positive scales")
    if len(weights) != 3 or any(v < 0 for v in weights) or sum(weights) <= 0:
        raise ValueError("Recorded preference weights must be nonnegative and nonzero")
    return [sum(weights[j] * (row[j] - minimum[j]) / scale[j] for j in range(3)) for row in targets]


def compute_regrets(targets, selected_indices, preference_scores, test_normalization, training_normalization, weights):
    """Choose from scores first; then evaluate with the separately supplied labels."""
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Returned set must be nonempty and contain unique indices")
    if len(preference_scores) != len(targets):
        raise ValueError("Score vector must cover the same candidate pool")
    if any(i < 0 or i >= len(targets) for i in selected_indices):
        raise ValueError("Returned index is outside the candidate pool")
    # This decision reads only the recorded scores and returned indices.
    chosen = min(selected_indices, key=lambda i: (preference_scores[i], i))
    tied = sum(preference_scores[i] == preference_scores[chosen] for i in selected_indices)
    test_costs = utilities(targets, test_normalization, weights)
    training_costs = utilities(targets, training_normalization, weights)
    test_optimum = min(test_costs)
    training_optimum = min(training_costs)
    return {
        "top1_index": chosen,
        "top1_exact_tie_count": tied,
        "top1_recorded_preference_score": preference_scores[chosen],
        "best_in_returned_set_regret_test_normalized": min(test_costs[i] for i in selected_indices) - test_optimum,
        "actual_top1_regret_test_normalized": test_costs[chosen] - test_optimum,
        "actual_top1_regret_training_normalized": training_costs[chosen] - training_optimum,
        "best_in_returned_set_regret_training_normalized": min(training_costs[i] for i in selected_indices) - training_optimum,
    }


def evaluation_directory(review_root):
    if (review_root / "oof_predictions.csv").is_file():
        return review_root
    matches = sorted(path.parent for path in review_root.glob("evaluation_*/oof_predictions.csv"))
    if len(matches) != 1:
        raise ValueError("Supply an evaluation directory or a review root with exactly one evaluation_*/oof_predictions.csv")
    return matches[0]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def reconstruct(review_root, output):
    review_root = review_root.resolve()
    source = evaluation_directory(review_root)
    output = output.resolve()
    if output == review_root or output.is_relative_to(review_root):
        raise ValueError("Derived output must be outside the supplied review root")
    if output.exists():
        raise ValueError("Output already exists; choose a new derived-output directory")
    hashes = {name: sha256(source / name) for name in INPUT_NAMES}
    recommendations = json.loads((source / "recommendations.json").read_text(encoding="utf-8"))
    fold_rows = json.loads((source / "fold_audit.json").read_text(encoding="utf-8"))
    fold_map = {row["test_group"]: row for row in fold_rows}
    if len(fold_map) != len(fold_rows):
        raise ValueError("Duplicate test groups in fold audit")
    metrics_rows = read_csv(source / "per_system_metrics.csv")
    metrics = {(row["model"], row["object_model"]): row for row in metrics_rows}
    if len(metrics) != len(metrics_rows):
        raise ValueError("Duplicate method/system metrics")
    predictions = read_csv(source / "oof_predictions.csv")
    grouped = {}
    negative = {}
    for row in predictions:
        key = (row["model"], row["object_model"])
        grouped.setdefault(key, []).append(row)
        counts = negative.setdefault(row["model"], {"prediction_rows": 0, "negative_mean_predictions": dict.fromkeys(OBJECTIVES, 0)})
        counts["prediction_rows"] += 1
        for objective in OBJECTIVES:
            if finite(row["mean_" + objective], "mean prediction") < 0:
                counts["negative_mean_predictions"][objective] += 1
    output_rows = []
    seen = set()
    for recommendation in recommendations:
        key = (recommendation["model"], recommendation["object_model"])
        if key in seen:
            raise ValueError(f"Duplicate recommendation: {key}")
        seen.add(key)
        metric = metrics[key]
        prediction_model = metric["prediction_model"]
        pool = grouped[(prediction_model, key[1])]
        ids = [row["schema_id"] for row in pool]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate OOF schema ID within {key}")
        selection = recommendation["selection"]
        indices = selection["indices"]
        if [ids[i] for i in indices] != recommendation["selected_schema_ids"]:
            raise ValueError(f"OOF order does not align with selected indices for {key}")
        scores = [finite(value, "recorded preference score") for value in selection["preference_scores"]]
        targets = [[finite(row["true_" + objective], objective) for objective in OBJECTIVES] for row in pool]
        weights = [finite(value, "preference weight") for value in selection["weights"]]
        calculated = compute_regrets(targets, indices, scores, recommendation["measured_metrics"]["metric_normalization"], fold_map[key[1]]["selection_normalization"], weights)
        stored_json = finite(recommendation["measured_metrics"]["preference_regret"], "JSON regret")
        stored_csv = finite(metric["preference_regret"], "CSV regret")
        calculated_set = calculated["best_in_returned_set_regret_test_normalized"]
        json_matches = math.isclose(stored_json, calculated_set, rel_tol=1e-9, abs_tol=1e-10)
        csv_matches = math.isclose(stored_csv, calculated_set, rel_tol=1e-9, abs_tol=1e-10)
        if not (json_matches and csv_matches):
            raise ValueError(f"Existing regret does not match reconstructed set regret for {key}")
        row = {
            "model": key[0], "prediction_model": prediction_model, "object_model": key[1],
            "fold": int(metric["fold"]), "selection_mode": selection["mode"],
            "risk_lambda": finite(selection["risk_lambda"], "risk lambda"),
            "pool_size": len(pool), "returned_count": len(indices),
            "actual_top1_schema_id": ids[calculated["top1_index"]],
            "stored_preference_regret_json": stored_json,
            "stored_preference_regret_csv": stored_csv,
            "existing_json_regret_matches": json_matches,
            "existing_csv_regret_matches": csv_matches,
            **calculated,
        }
        output_rows.append(row)
    if seen != set(metrics):
        raise ValueError("Recommendation and per-system metric keys do not cover the same rows")
    names = (
        "stored_preference_regret_json", "best_in_returned_set_regret_test_normalized",
        "actual_top1_regret_test_normalized", "actual_top1_regret_training_normalized",
        "best_in_returned_set_regret_training_normalized",
    )
    macro = {}
    for model in sorted({row["model"] for row in output_rows}):
        selected_rows = [row for row in output_rows if row["model"] == model]
        macro[model] = {name: statistics.mean(row[name] for row in selected_rows) for name in names}
        macro[model].update({
            "system_count": len(selected_rows),
            "systems_with_more_than_one_exact_top1_score_tie": sum(row["top1_exact_tie_count"] > 1 for row in selected_rows),
            "systems_where_top1_test_regret_exceeds_best_in_set": sum(row["actual_top1_regret_test_normalized"] > row["best_in_returned_set_regret_test_normalized"] + 1e-10 for row in selected_rows),
        })
    after = {name: sha256(source / name) for name in INPUT_NAMES}
    if hashes != after:
        raise ValueError("Source files changed while reconstructing the report")
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "preference_regret_by_method_system.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "status": "completed", "evaluation_directory": str(source), "row_count": len(output_rows),
        "method_count": len(macro), "source_system_count": len(fold_map),
        "all_existing_regret_values_match_reconstructed_best_in_returned_set": True,
        "match_tolerance": {"relative": 1e-9, "absolute": 1e-10},
        "selection_rule": "Minimum recorded preference_scores within recorded returned indices; exact score ties resolved by lowest original candidate index. Test labels are never used to choose top1.",
        "normalizations": {
            "test_normalized": "Recorded full-test-pool metric minimum/scale; descriptive utility distinct from the selection's training-normalized objective weights.",
            "training_normalized": "Recorded proper-training minimum/scale from fold_audit; consistent with selection utility scaling. Realized regret uses measured outcomes, not the predicted uncertainty penalty.",
        },
        "interpretation": "Descriptive reconstruction from one saved run, not evidence that one model is superior. Best-in-returned-set regret assumes an oracle can identify the best measured utility inside that set; actual top1 regret evaluates the schema chosen using recorded predictions only.",
        "source_files_sha256": hashes, "source_hashes_unchanged_after_read": hashes == after,
        "source_files_modified": False, "new_training_or_database_measurements": False,
        "negative_mean_predictions_by_prediction_model": negative,
        "negative_prediction_note": "Counts refer to saved mean point predictions. Shared ensemble5 predictions are counted once, not separately for mean/risk recommendation variants. No predictions are clipped or edited.",
        "macro_system_means": macro,
        "csv": csv_path.name, "csv_sha256": sha256(csv_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    with (output / "preference_regret_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = reconstruct(args.review_root, args.out)
    print(json.dumps({"status": report["status"], "rows": report["row_count"], "all_existing_regrets_match": report["all_existing_regret_values_match_reconstructed_best_in_returned_set"], "output": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
