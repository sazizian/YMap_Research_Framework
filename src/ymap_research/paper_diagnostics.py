"""Matched-candidate diagnostics for the paper's RQ3 and Hibernate supplement.

Measurements are one row per (object_model, schema_id), with insert_ms,
query_ms, and storage_mb in their stated units. Aggregate raw repetitions
within each documented protocol before exporting. Optional provenance columns
are checked when present; a bare CSV is treated as user-supplied measurements.
No missing values, scenarios, or predictions are imputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .common import RESEARCH_SOURCES, TARGETS, sha256_file
from .recommendation import hypervolume_3d, pareto_mask


_KEY = ["object_model", "schema_id"]
_MEANS = [f"mean_{name}" for name in TARGETS]
_PRED = [f"pred_{name}" for name in TARGETS]
_TRUE = [f"true_{name}" for name in TARGETS]
_SCOPE_FIELDS = (
    "protocol_id", "protocol_sha256", "profile_protocol", "scenario_id", "variant",
    "workload_id", "environment_id", "population_id", "dataset_id", "execution_layer",
)
_MEASURED_SOURCES = RESEARCH_SOURCES | {"hibernate_measured", "sql_measured"}
_HV_REFERENCE = (1.1, 1.1, 1.1)


def _csv(path, role: str) -> pd.DataFrame:
    try:
        # Preserve string IDs such as "001" and "NA" exactly.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{role} CSV is empty") from exc
    if frame.empty:
        raise ValueError(f"{role} CSV is empty")
    if any(f"{name}.1" in frame for name in (*_KEY, *TARGETS, *_MEANS, *_PRED, *_TRUE)):
        raise ValueError(f"{role} CSV has repeated objective or identity columns")
    return frame


def _require_text(frame: pd.DataFrame, fields, role: str) -> None:
    missing = sorted(set(fields) - set(frame))
    if missing:
        raise ValueError(f"{role} CSV is missing required columns: {missing}")
    for name in fields:
        if frame[name].str.strip().eq("").any() or frame[name].ne(frame[name].str.strip()).any():
            raise ValueError(f"{role} CSV has a blank or padded {name}")


def _numbers(frame: pd.DataFrame, fields, role: str, *, nonnegative: bool) -> None:
    missing = sorted(set(fields) - set(frame))
    if missing:
        raise ValueError(f"{role} CSV is missing required columns: {missing}")
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
            sign = "nonnegative and " if nonnegative else ""
            raise ValueError(f"{role} {field} must contain {sign}finite numbers")
        frame[field] = values


def _scope(frame: pd.DataFrame, role: str) -> dict[str, str]:
    metadata = {}
    for name in (*_SCOPE_FIELDS, "label_source", "mode"):
        if name in frame:
            _require_text(frame, [name], role)
            if frame[name].nunique() != 1:
                raise ValueError(f"{role} CSV pools different {name} values; use one scenario per file")
            metadata[name] = str(frame[name].iloc[0])
    if "label_source" in metadata and metadata["label_source"] not in _MEASURED_SOURCES:
        raise ValueError(f"{role} CSV has an unsupported label_source; actual measured records are required")
    if "mode" in metadata and metadata["mode"] != "research":
        raise ValueError(f"{role} CSV is not in research mode; actual measured records are required")
    return metadata


def _artifact(path, frame: pd.DataFrame, metadata: dict) -> dict:
    return {
        "file": Path(path).name,
        "sha256": sha256_file(path),
        "rows": len(frame),
        "scenario": metadata,
    }


def _measurements(path, role: str) -> tuple[pd.DataFrame, dict]:
    frame = _csv(path, role)
    _require_text(frame, _KEY, role)
    metadata = _scope(frame, role)
    _numbers(frame, TARGETS, role, nonnegative=True)
    if frame.duplicated(_KEY).any():
        raise ValueError(f"{role} CSV has duplicate candidate records; provide one measured row per candidate")
    artifact = _artifact(path, frame, metadata)
    artifact["provenance"] = "recorded_label_source" if "label_source" in metadata else "user_supplied_measurement_columns"
    return frame.set_index(_KEY).sort_index(), artifact


def _check_fold_audit(path, predictions: pd.DataFrame) -> dict:
    """Check the recorded split independently of the prediction CSV's fold IDs."""
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Fold audit must contain valid JSON") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("Fold audit must be a nonempty list of recorded folds")
    folds = {}
    for row in rows:
        if not isinstance(row, dict) or "fold" not in row:
            raise ValueError("Every fold audit row must name its fold")
        fold = str(row["fold"])
        test = row.get("test_system", row.get("test_group"))
        training = row.get("training_systems", row.get("training_groups"))
        validation = [row["validation_system"]] if "validation_system" in row else row.get("calibration_groups")
        if not isinstance(test, str) or not test.strip():
            raise ValueError(f"Fold audit {fold!r} is missing its test system")
        if any(not isinstance(group, list) or not group or any(not isinstance(x, str) or not x.strip() for x in group)
               for group in (training, validation)):
            raise ValueError(f"Fold audit {fold!r} needs training and validation systems")
        if test in training or test in validation or set(training) & set(validation):
            raise ValueError(f"Fold audit {fold!r} overlaps training, validation, and held-out systems")
        if fold in folds:
            raise ValueError(f"Fold audit repeats fold {fold!r}")
        folds[fold] = (test, validation)
    for fold, group in predictions.groupby("fold"):
        if fold not in folds:
            raise ValueError(f"Prediction fold {fold!r} is missing from the fold audit")
        test, validation = folds[fold]
        if not group.object_model.eq(test).all() or (
            "validation_system" in group and not group.validation_system.isin(validation).all()
        ):
            raise ValueError(f"Prediction fold {fold!r} conflicts with the recorded split")
    return {"file": Path(path).name, "sha256": sha256_file(path), "folds": len(folds)}


def _predictions(path, model: str | None, fold_audit_path) -> tuple[pd.DataFrame, dict]:
    role = "Reference held-out predictions"
    frame = _csv(path, role)
    if "model" in frame:
        _require_text(frame, ["model"], role)
        models = sorted(frame.model.unique().tolist())
        if model is None and len(models) > 1:
            raise ValueError(f"Select one prediction model from {models}")
        model = model if model is not None else models[0]
        frame = frame.loc[frame.model.eq(model)].copy()
        if frame.empty:
            raise ValueError(f"Prediction model {model!r} does not occur in the CSV")
    elif model is not None:
        raise ValueError("Cannot select a model without a model column")

    _require_text(frame, [*_KEY, "fold"], role)
    metadata = _scope(frame, role)
    if "scope" in frame:
        _require_text(frame, ["scope"], role)
        scopes = frame.scope.unique().tolist()
        if len(scopes) != 1 or scopes[0].casefold() in {"demo", "synthetic", "synthetic_demo"}:
            raise ValueError("Prediction CSV must contain one recorded non-demo experiment scope")
        experiment_scope = str(scopes[0])
    else:
        experiment_scope = None
    if set(_PRED).intersection(frame):
        if set(_MEANS).intersection(frame):
            raise ValueError("Prediction CSV has two objective column sets; use pred_* or mean_*")
        frame = frame.rename(columns=dict(zip(_PRED, _MEANS)))
    _numbers(frame, _MEANS, role, nonnegative=False)
    if set(_TRUE).intersection(frame):
        _numbers(frame, _TRUE, role, nonnegative=True)

    repeat_fields = [name for name in ("repeat", "repetition") if name in frame]
    if len(repeat_fields) > 1:
        raise ValueError("Prediction CSV has two repetition columns; use repeat or repetition")
    if repeat_fields:
        _require_text(frame, repeat_fields, role)
        frame["_repeat"] = frame[repeat_fields[0]]
    else:
        frame["_repeat"] = "single"
    if frame.duplicated([*_KEY, "_repeat"]).any():
        raise ValueError("Prediction CSV has duplicate candidate records within a repeat")
    by_system = frame.groupby(["_repeat", "object_model"])["fold"].nunique()
    by_fold = frame.groupby(["_repeat", "fold"])["object_model"].nunique()
    if not by_system.eq(1).all() or not by_fold.eq(1).all():
        raise ValueError("Each held-out fold must correspond to exactly one system per prediction repeat")
    for field in ("test_group", "test_system", "held_out_system", "outer_test_group"):
        if field in frame:
            _require_text(frame, [field], role)
            if not frame[field].eq(frame.object_model).all():
                raise ValueError(f"{role} {field} conflicts with the candidate's object_model")
    if "validation_system" in frame:
        _require_text(frame, ["validation_system"], role)
        by_fold = frame.groupby(["_repeat", "fold"])["validation_system"].nunique()
        if frame.validation_system.eq(frame.object_model).any() or not by_fold.eq(1).all():
            raise ValueError("The validation system must be fixed per fold and separate from its held-out system")

    artifact = _artifact(path, frame, metadata)
    artifact["model"] = model
    artifact["experiment_scope"] = experiment_scope
    artifact["split_checks"] = "prediction CSV records one held-out fold per system per repeat"
    if fold_audit_path is None and experiment_scope in {"paper", "ablation", "smoke"}:
        sibling = Path(path).with_name("folds.json")
        if sibling.is_file():
            fold_audit_path = sibling
    if fold_audit_path is not None:
        artifact["fold_audit"] = _check_fold_audit(fold_audit_path, frame)
        artifact["split_checks"] += "; saved audit has disjoint training, validation, and test systems"
    return frame.set_index(_KEY).sort_index(), artifact


def _match(required: pd.MultiIndex, available: pd.MultiIndex, role: str, *, allow_superset: bool) -> None:
    missing, extra = required.difference(available), available.difference(required)
    if len(missing) or (len(extra) and not allow_superset):
        raise ValueError(
            f"{role} candidate identities do not match: {len(missing)} missing, {len(extra)} extra. "
            f"Missing examples: {missing[:3].tolist()}; extra examples: {extra[:3].tolist()}."
        )


def _correlations(left, right, statistic, objectives=TARGETS) -> dict:
    correlations = {}
    for col, name in enumerate(objectives):
        x, y = left[:, col], right[:, col]
        if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
            correlations[name] = None
        else:
            correlations[name] = float(statistic(x, y).statistic)
    return correlations


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.count_nonzero(left & right) / np.count_nonzero(left | right))


def _summary(values, statistic) -> dict:
    defined = [value for value in values if value is not None]
    return {"value": float(statistic(defined)) if defined else None, "defined": len(defined)}


def _summarize(systems: list[dict], scalar_fields, objective_fields, statistic) -> dict:
    result = {name: _summary((row[name] for row in systems), statistic) for name in scalar_fields}
    for name, objectives in objective_fields.items():
        result[name] = {obj: _summary((row[name][obj] for row in systems), statistic) for obj in objectives}
    return result


def protocol_transfer(
    reference_path, variant_path, predictions_path, *, model: str | None = None,
    allow_reference_superset: bool = False, fold_audit_path=None,
) -> dict:
    """Evaluate one measured variant using the reference's saved held-out predictions.

    Predictions require object_model, schema_id, fold, and all three pred_*
    (or legacy mean_*)
    columns. Select a model if a CSV contains several. If present, repeat
    (or repetition) identifies independent OOF runs; their metrics are averaged
    within each system before taking the paper's median across systems.
    The evaluator's adjacent folds.json is checked when available, or specify
    fold_audit_path for a separate audit. Otherwise only CSV split metadata
    can be checked.

    Exact candidate matching is the default. For a predetermined variant
    sample, allow_reference_superset=True explicitly restricts both reference
    inputs to all supplied variant candidates. Measured fronts, the predicted
    risk-neutral front, and variant HV normalization use that matched pool.
    """
    reference, ref_info = _measurements(reference_path, "Reference measurements")
    variant, variant_info = _measurements(variant_path, "Variant measurements")
    predictions, pred_info = _predictions(predictions_path, model, fold_audit_path)
    for field in sorted(set(ref_info["scenario"]) & set(pred_info["scenario"])):
        if ref_info["scenario"][field] != pred_info["scenario"][field]:
            raise ValueError(f"Predictions' {field} does not match the reference measurements")
    for field in ("environment_id", "profile_protocol"):
        ref_scenario, var_scenario = ref_info["scenario"], variant_info["scenario"]
        if field in ref_scenario and field in var_scenario and ref_scenario[field] != var_scenario[field]:
            raise ValueError(f"Reference and variant measurements must match their recorded {field}")

    _match(variant.index, reference.index, "Reference measurement", allow_superset=allow_reference_superset)
    repeat_frames = {}
    for repeat, run in predictions.groupby("_repeat", sort=True):
        _match(variant.index, run.index, f"Reference prediction repeat {repeat!r}", allow_superset=allow_reference_superset)
        if _TRUE[0] in run:
            shared = reference.index.intersection(run.index)
            if not np.allclose(run.loc[shared, _TRUE], reference.loc[shared, TARGETS], rtol=1e-9, atol=1e-9):
                raise ValueError(f"Predictions' saved true objectives disagree with reference measurements in repeat {repeat!r}")
        repeat_frames[str(repeat)] = run.loc[variant.index, _MEANS]

    systems = []
    for system, variant_group in variant.groupby(level="object_model", sort=True):
        variant_values = variant_group[TARGETS].to_numpy()
        ref_values = reference.loc[variant_group.index, TARGETS].to_numpy()
        ref_front, var_front = pareto_mask(ref_values), pareto_mask(variant_values)
        minimum, span = variant_values.min(axis=0), np.ptp(variant_values, axis=0)
        normalized = (variant_values - minimum) / np.where(span > 0, span, 1.0)
        variant_hv = hypervolume_3d(normalized[var_front], _HV_REFERENCE)
        per_repeat = []
        for repeat, run in repeat_frames.items():
            means = run.loc[variant_group.index].to_numpy()
            pred_front = pareto_mask(means)
            predicted_hv = hypervolume_3d(normalized[pred_front], _HV_REFERENCE)
            per_repeat.append({
                "repeat": repeat,
                "prediction_transfer_spearman": _correlations(means, variant_values, spearmanr),
                "selected_front_count": int(np.count_nonzero(pred_front)),
                "selected_hypervolume": predicted_hv,
                "hypervolume_ratio": predicted_hv / variant_hv,
            })
        means = _summarize(
            per_repeat, ("hypervolume_ratio",), {"prediction_transfer_spearman": TARGETS}, np.mean,
        )
        systems.append({
            "object_model": system,
            "matched_candidates": len(variant_group),
            "reference_front_count": int(np.count_nonzero(ref_front)),
            "variant_front_count": int(np.count_nonzero(var_front)),
            "variant_measured_hypervolume": variant_hv,
            "measured_rank_kendall_tau_b": _correlations(ref_values, variant_values, kendalltau),
            "measured_front_jaccard": _jaccard(ref_front, var_front),
            "prediction_transfer_spearman": {
                name: value["value"] for name, value in means["prediction_transfer_spearman"].items()
            },
            "hypervolume_ratio": means["hypervolume_ratio"]["value"],
            "prediction_repeats": per_repeat,
        })
    return {
        "analysis": "RQ3_protocol_transfer",
        "claim": "Analysis of the supplied matched records; this does not establish reproduction of the historical results.",
        "inputs": {"reference": ref_info, "variant": variant_info, "reference_predictions": pred_info},
        "matched_candidates": len(variant),
        "system_count": len(systems),
        "prediction_repeat_count": len(repeat_frames),
        "candidate_scope": "all variant candidates; reference superset allowed" if allow_reference_superset else "identical candidate sets",
        "hypervolume_reference_point": list(_HV_REFERENCE),
        "hypervolume_normalization_scope": "matched variant measurements within each system",
        "predicted_front": "risk-neutral nondominated reference prediction means in the matched candidate pool",
        "aggregation": "median over systems after averaging prediction metrics over repeats within each system",
        "per_system": systems,
        "median_over_systems": _summarize(
            systems, ("measured_front_jaccard", "hypervolume_ratio"),
            {"measured_rank_kendall_tau_b": TARGETS, "prediction_transfer_spearman": TARGETS}, np.median,
        ),
    }


def orm_comparison(sql_path, hibernate_path) -> dict:
    """Compare recorded SQL and Hibernate measurements of the same candidates.

    Report the scope of the supplied subset; this does not execute Hibernate,
    establish mapping expressibility, or estimate performance on other schemas.
    """
    sql, sql_info = _measurements(sql_path, "SQL measurements")
    hibernate, hibernate_info = _measurements(hibernate_path, "Hibernate measurements")
    if Path(sql_path).samefile(hibernate_path):
        raise ValueError("Provide separate SQL and Hibernate measurement records, not the same input")
    for info, expected in ((sql_info, "sql"), (hibernate_info, "hibernate")):
        scenario = info["scenario"]
        source = scenario.get("label_source")
        layer = scenario.get("execution_layer")
        if (layer and layer.casefold() != expected) or source == f"{'hibernate' if expected == 'sql' else 'sql'}_measured":
            raise ValueError(f"The recorded execution layer does not match the {expected} measurements")
    for field in ("workload_id", "population_id", "dataset_id"):
        sql_scenario, hibernate_scenario = sql_info["scenario"], hibernate_info["scenario"]
        if field in sql_scenario and field in hibernate_scenario and sql_scenario[field] != hibernate_scenario[field]:
            raise ValueError(f"SQL and Hibernate measurements must match their recorded {field}")
    _match(sql.index, hibernate.index, "Hibernate measurement", allow_superset=False)
    latency = TARGETS[:2]
    systems = []
    for system, sql_group in sql.groupby(level="object_model", sort=True):
        sql_values = sql_group[TARGETS].to_numpy()
        orm_values = hibernate.loc[sql_group.index, TARGETS].to_numpy()
        sql_front, orm_front = pareto_mask(sql_values), pareto_mask(orm_values)
        slowdown = {}
        for i, name in enumerate(latency):
            usable = sql_values[:, i] > 0
            with np.errstate(over="ignore"):
                ratios = orm_values[usable, i] / sql_values[usable, i]
            slowdown[name] = _summary(ratios[np.isfinite(ratios)], np.median)
        systems.append({
            "object_model": system,
            "matched_candidates": len(sql_group),
            "sql_front_count": int(np.count_nonzero(sql_front)),
            "hibernate_front_count": int(np.count_nonzero(orm_front)),
            "latency_rank_spearman": _correlations(sql_values, orm_values, spearmanr, latency),
            "measured_front_jaccard": _jaccard(sql_front, orm_front),
            "median_candidate_latency_ratio": slowdown,
        })
    summary = _summarize(systems, ("measured_front_jaccard",), {"latency_rank_spearman": latency}, np.median)
    summary["median_candidate_latency_ratio"] = {
        name: _summary((row["median_candidate_latency_ratio"][name]["value"] for row in systems), np.median)
        for name in latency
    }
    return {
        "analysis": "supplementary_sql_hibernate_comparison",
        "inputs": {"sql": sql_info, "hibernate": hibernate_info},
        "matched_candidates": len(sql),
        "system_count": len(systems),
        "candidate_scope": "supplied matched subset; execution and expressibility are not inferred",
        "latency_ratio_scope": "matched candidates with positive SQL latency and finite Hibernate/SQL ratio",
        "aggregation": "median over systems, reported explicitly because the manuscript does not specify pooling",
        "per_system": systems,
        "median_over_systems": summary,
    }
