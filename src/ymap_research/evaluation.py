"""Leakage-resistant leave-one-object-model-out evaluation.

No measured held-out outcomes are passed into fit, predict, normalization or
recommendation. Labels are used only afterward to calculate metrics. Fixed
hyperparameters are supplied in config; no candidate-random validation or tuning.
Repeated measurement seeds are averaged within schema, never counted as systems.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .recommendation import fit_normalization, recommend, recommendation_metrics
from .statistics import compare_system_metrics

OBJECTIVES = ("insert_ms", "query_ms", "storage_mb")
RESERVED_FEATURES = set(OBJECTIVES) | {"insert_latency_ms", "query_latency_ms", "memory_mb", "storage_bytes", "label_source", "workload_id", "environment_id", "schema_id", "object_model", "seed", "target", "label"}


def _json_clean(obj):
    """Return standard-JSON values, using null for undefined numeric metrics."""
    if isinstance(obj, dict):
        return {str(key): _json_clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _json_write(path, value):
    Path(path).write_text(json.dumps(_json_clean(value), indent=2, allow_nan=False), encoding="utf-8")


def feature_matrix(records, feature_names):
    """Extract declared structural features only; missing values remain NaN."""
    output = []
    for record in records:
        values = record.get("features", {})
        row = []
        for name in feature_names:
            value = values.get(name)
            if value is None:
                row.append(np.nan)
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                if np.isinf(value):
                    raise ValueError(f"Infinite feature {name} in {record.get('schema_id')}")
                row.append(float(value))
            else:
                raise ValueError(f"Feature {name} must be numeric or null")
        output.append(row)
    return np.asarray(output, dtype=float).reshape(len(records), len(feature_names))


def load_dataset(candidates_path, measurements_path, config=None):
    """Read one declared workload/environment scenario and average measurement seeds.

    Returns (candidate_records, aligned_measurements_dataframe, feature_names).
    Features are sorted declared numeric keys; their values are never fitted here.
    Measured objectives cannot appear as features. Feature availability is not a
    substitute for a scientific provenance audit of user-supplied feature values.
    """
    config = config or {}
    records = []
    for line_number, line in enumerate(Path(candidates_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not all(isinstance(record.get(key), str) and record[key].strip() for key in ("schema_id", "object_model")) or not isinstance(record.get("features"), dict):
            raise ValueError(f"Candidate line {line_number} needs schema_id, object_model and numerical features")
        records.append(record)
    if not records:
        raise ValueError("Candidate file is empty")
    ids = [str(record["schema_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate schema_id values must be unique")
    feature_names = sorted(set().union(*(record["features"] for record in records)))
    if not feature_names:
        raise ValueError("At least one structural feature is required")
    forbidden = [name for name in feature_names if name.lower() in RESERVED_FEATURES or name.lower().startswith(("target_", "true_", "measured_"))]
    if forbidden:
        raise ValueError(f"Outcome/provenance keys cannot be prediction features: {forbidden}")
    feature_matrix(records, feature_names)
    frame = pd.read_csv(measurements_path, keep_default_na=False, dtype={"schema_id": str, "object_model": str, "label_source": str, "workload_id": str, "environment_id": str})
    required = {"schema_id", "object_model", *OBJECTIVES, "label_source", "workload_id", "environment_id", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Measurement CSV missing columns {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Measurement CSV is empty")
    if frame[list(required)].isna().any().any() or any(frame[key].astype(str).str.strip().eq("").any() for key in required):
        raise ValueError("Measurement IDs, objectives, label_source, scenario and seed must not be missing")
    for key in ("workload_id", "environment_id"):
        if config.get(key) is not None:
            frame = frame[frame[key] == str(config[key])]
    if frame.empty:
        raise ValueError("No measurements match the requested scenario")
    if frame[["workload_id", "environment_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Evaluate exactly one workload_id/environment_id scenario; filter explicitly to avoid pooling scenarios")
    label_sources = set(frame.label_source.astype(str))
    mode = config.get("mode", "research")
    if mode not in {"research", "demo"}:
        raise ValueError("mode must be research or demo")
    if mode == "research":
        allowed = set(config.get("allowed_label_sources", ["postgresql_measured", "imported_measured"]))
        if any("synthetic" in source.lower() or "demo" in source.lower() for source in label_sources) or not label_sources.issubset(allowed):
            raise ValueError(f"Research evaluation requires documented measured labels; found {sorted(label_sources)}")
    if len(label_sources) != 1:
        raise ValueError("Do not combine label sources in one evaluation")
    for objective in OBJECTIVES:
        frame[objective] = pd.to_numeric(frame[objective], errors="raise")
        if not np.isfinite(frame[objective]).all() or (frame[objective] < 0).any():
            raise ValueError(f"Objective {objective} must be finite and nonnegative")
    if frame.duplicated(["schema_id", "workload_id", "environment_id", "seed"]).any():
        raise ValueError("Duplicate schema/scenario/seed rows; repeated timings belong in separate raw logs")
    by_id = {str(record["schema_id"]): record for record in records}
    for row in frame.itertuples():
        if row.schema_id not in by_id:
            raise ValueError(f"Measurement references unknown schema {row.schema_id}")
        if str(by_id[row.schema_id]["object_model"]) != row.object_model:
            raise ValueError(f"object_model mismatch for schema {row.schema_id}")
    grouped = frame.groupby(["schema_id", "object_model", "label_source", "workload_id", "environment_id"], sort=True, as_index=False)
    averaged = grouped[list(OBJECTIVES)].mean()
    repeat_counts = grouped.size().rename(columns={"size": "measurement_seed_count"})
    averaged = averaged.merge(repeat_counts, on=["schema_id", "object_model", "label_source", "workload_id", "environment_id"], validate="one_to_one")
    records = [by_id[str(schema_id)] for schema_id in averaged.schema_id]
    # Exactly identical DDL in different groups would undermine group independence.
    hashes = {}
    for record in records:
        ddl = record.get("ddl", "")
        if ddl:
            signature = hashlib.sha256(" ".join(ddl.split()).encode()).hexdigest()
            previous = hashes.setdefault(signature, str(record["object_model"]))
            if previous != str(record["object_model"]):
                raise ValueError("Identical DDL occurs under different object_model groups; reconcile source-system identity")
    averaged.attrs["unmeasured_candidate_count"] = len(by_id) - len(records)
    return records, averaged, feature_names


def make_regressor(name: str, seed: int = 42, model_config=None):
    """Build unfitted train-only preprocessing + multiobjective regression.

    'xgboost' explicitly raises ImportError when unavailable; no substitutions.
    MLP scales its targets using training-only TransformedTargetRegressor.
    """
    cfg = dict(model_config or {})
    if name in {"linear", "linear_regression"}:
        estimator = LinearRegression(**cfg)
    elif name == "ridge":
        estimator = Ridge(alpha=cfg.pop("alpha", 1.0), **cfg)
    elif name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=cfg.pop("n_estimators", 100), random_state=seed, n_jobs=cfg.pop("n_jobs", 1), **cfg)
    elif name == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=cfg.pop("n_estimators", 100), random_state=seed, n_jobs=cfg.pop("n_jobs", 1), **cfg)
    elif name == "mlp":
        if cfg.pop("early_stopping", False):
            raise ValueError("MLP random internal validation is disabled; use fixed max_iter with groupwise external validation")
        raw = MLPRegressor(hidden_layer_sizes=tuple(cfg.pop("hidden_layer_sizes", [64, 32])), max_iter=cfg.pop("max_iter", 500), early_stopping=False, random_state=seed, **cfg)
        estimator = TransformedTargetRegressor(regressor=raw, transformer=StandardScaler())
    elif name == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError("xgboost is not installed; requested baseline is unavailable, not substituted")
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        estimator = MultiOutputRegressor(XGBRegressor(n_estimators=cfg.pop("n_estimators", 100), random_state=seed, n_jobs=cfg.pop("n_jobs", 1), **cfg))
    else:
        raise ValueError(f"Unknown baseline {name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler()), ("regressor", estimator)])


def _neural_config(config, seed, ensemble_size):
    from .neural import NeuralConfig, EncoderConfig
    values = dict(config.get("neural", {}))
    encoder_values = values.pop("text_encoders", [])
    values["text_encoders"] = tuple(EncoderConfig(**value) for value in encoder_values)
    values["seed"] = seed
    values["ensemble_size"] = ensemble_size
    if tuple(values.get("target_names", OBJECTIVES)) != OBJECTIVES:
        raise ValueError("Neural evaluation target_names must match insert_ms, query_ms, storage_mb in that order")
    return NeuralConfig(**values)


def _synchronize_neural_device(config):
    # Avoid reporting only asynchronous enqueue time on accelerator backends.
    device = str(config.get("neural", {}).get("device", "cpu"))
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device=device)
    elif device.startswith("mps"):
        import torch
        torch.mps.synchronize()


def _fit_predict(name, count, records_train, y_train, records_test, records_calibration, feature_names, seed, config):
    """Return actual member predictions, warnings and wall-clock stage timings.

    Structural feature extraction is reported separately. Structural fit includes
    construction and all pipeline/target preprocessing. Neural fit and prediction
    timers include their representation preparation, cache work and device work;
    these are real end-to-end API call times, not GPU-kernel-only timings.
    """
    timing = {"feature_preparation": 0.0, "fit": 0.0, "test_prediction": 0.0, "calibration_prediction": 0.0}
    if name == "neural":
        from .neural import NeuralMultiViewRegressor
        started = time.perf_counter()
        fitted = NeuralMultiViewRegressor(_neural_config(config, seed, count)).fit(records_train, y_train)
        _synchronize_neural_device(config)
        timing["fit"] = time.perf_counter() - started
        started = time.perf_counter()
        test = fitted.predict_members(records_test)
        _synchronize_neural_device(config)
        timing["test_prediction"] = time.perf_counter() - started
        if records_calibration:
            started = time.perf_counter()
            calibration = fitted.predict_members(records_calibration)
            _synchronize_neural_device(config)
            timing["calibration_prediction"] = time.perf_counter() - started
        else:
            calibration = np.empty((count, 0, 3))
        return np.asarray(test), np.asarray(calibration), [], timing
    started = time.perf_counter()
    x_train = feature_matrix(records_train, feature_names)
    x_test = feature_matrix(records_test, feature_names)
    x_calibration = feature_matrix(records_calibration, feature_names)
    timing["feature_preparation"] = time.perf_counter() - started
    test_predictions, calibration_predictions, captured = [], [], []
    for member in range(count):
        started = time.perf_counter()
        model = make_regressor(name, seed + member, config.get("model_parameters", {}).get(name, {}))
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            model.fit(x_train, y_train)
        timing["fit"] += time.perf_counter() - started
        captured.extend(sorted(set(str(message.message) for message in messages)))
        started = time.perf_counter()
        test_predictions.append(model.predict(x_test))
        timing["test_prediction"] += time.perf_counter() - started
        if len(x_calibration):
            started = time.perf_counter()
            calibration_predictions.append(model.predict(x_calibration))
            timing["calibration_prediction"] += time.perf_counter() - started
        else:
            calibration_predictions.append(np.empty((0, 3)))
    return np.asarray(test_predictions), np.asarray(calibration_predictions), sorted(set(captured)), timing


def _regression_metrics(truth, predictions, training_truth):
    output = {}
    for j, name in enumerate(OBJECTIVES):
        error = predictions[:, j] - truth[:, j]
        span = float(np.ptp(training_truth[:, j]))
        test_constant = len(truth) < 2 or np.ptp(truth[:, j]) == 0
        output[f"{name}_mae"] = float(np.abs(error).mean())
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        output[f"{name}_nrmse"] = output[f"{name}_rmse"] / span if span > 0 else None
        output[f"{name}_r2"] = None if test_constant else float(r2_score(truth[:, j], predictions[:, j]))
        output[f"{name}_training_constant"] = span == 0
        output[f"{name}_test_constant"] = test_constant
    normalized = [output[f"{name}_nrmse"] for name in OBJECTIVES]
    output["nrmse"] = float(np.mean(normalized)) if all(value is not None for value in normalized) else None
    return output


def _uncertainty_metrics(truth, mean, std, calibration_truth, calibration_mean, alpha):
    """Empirical held-out residual intervals, without conformal guarantees.

    Residual quantiles are pooled within explicitly held-out calibration groups.
    Candidate residuals in a source group may be dependent; therefore these are
    descriptive marginal intervals, not finite-sample-valid conformal intervals.
    Coverage is measured only on the outer held-out source model.
    """
    radii = np.full(3, np.nan)
    metrics = {}
    for j, name in enumerate(OBJECTIVES):
        errors = np.abs(truth[:, j] - mean[:, j])
        if len(calibration_truth):
            residuals = np.abs(calibration_truth[:, j] - calibration_mean[:, j])
            radii[j] = np.quantile(residuals, 1 - alpha, method="higher")
            metrics[f"{name}_interval_coverage"] = float((errors <= radii[j]).mean())
            metrics[f"{name}_interval_width"] = float(2 * radii[j])
        else:
            metrics[f"{name}_interval_coverage"] = None
            metrics[f"{name}_interval_width"] = None
        if len(errors) >= 3 and np.ptp(errors) > 0 and np.ptp(std[:, j]) > 0:
            metrics[f"{name}_uncertainty_error_spearman"] = float(spearmanr(std[:, j], errors).statistic)
        else:
            metrics[f"{name}_uncertainty_error_spearman"] = None
    return metrics, radii


def evaluate(candidates_path, measurements_path, out_dir, config: dict | None = None) -> dict:
    """Run honest LOSO baselines, fixed five-member control, metrics and statistics.

    Important config keys: mode='research', models=[...], seed=42,
    ensemble_base='extra_trees', ensemble_size=5, model_parameters={name:{...}},
    calibration=True, calibration_fraction=.25, calibration_alpha=.1,
    selection_mode='pareto'|'preference'|'beam', weights=[1,1,1], risk_lambda=.25,
    beam_budget/beam_width/beam_iterations/proposals_per_beam, reference_model.
    Explicit workload_id/environment_id filters select one scenario. All model
    hyperparameters are fixed a priori; calibration groups do not tune models.
    """
    workflow_started = time.perf_counter()
    config = dict(config or {})
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, data, feature_names = load_dataset(candidates_path, measurements_path, config)
    groups = sorted(set(data.object_model))
    if len(groups) < 2:
        raise ValueError("LOSO requires at least two distinct object_model groups")
    calibration_enabled = config.get("calibration", True)
    if not isinstance(calibration_enabled, bool):
        raise ValueError("calibration must be a JSON/YAML boolean")
    if config.get("mode", "research") == "research" and calibration_enabled and len(groups) < 3:
        raise ValueError("Research evaluation with grouped calibration requires at least three source groups")
    models = list(dict.fromkeys(config.get("models", ["linear", "ridge", "random_forest", "extra_trees", "mlp"])))
    ensemble_base = config.get("ensemble_base", "extra_trees")
    ensemble_size = config.get("ensemble_size", 5)
    if isinstance(ensemble_size, bool) or not isinstance(ensemble_size, (int, np.integer)) or ensemble_size != 5:
        raise ValueError("Controlled ensemble comparison requires exactly five members")
    seed = int(config.get("seed", 42))
    risk_lambda = float(config.get("risk_lambda", 0.25))
    alpha = float(config.get("calibration_alpha", 0.1))
    calibration_fraction = float(config.get("calibration_fraction", 0.25))
    if not (0 < alpha < 1) or not (0 < calibration_fraction < 1):
        raise ValueError("Calibration alpha and fraction must be between zero and one")
    specifications = [(name, name, 1) for name in models] + [("ensemble5", ensemble_base, 5)]
    unavailable = []
    active = []
    for label, name, count in specifications:
        try:
            if name == "neural":
                if importlib.util.find_spec("torch") is None:
                    raise ImportError("Neural model unavailable: optional torch dependency is absent; no fallback is substituted")
                neural_cfg = _neural_config(config, seed, count)
                if neural_cfg.text_encoders and importlib.util.find_spec("transformers") is None:
                    raise ImportError("Text encoders unavailable: optional transformers dependency is absent; no fallback is substituted")
            else:
                make_regressor(name, seed, config.get("model_parameters", {}).get(name, {}))
            active.append((label, name, count))
        except ImportError as exc:
            unavailable.append({"model": label, "requested_implementation": name, "status": "unavailable", "reason": str(exc)})
    if not active:
        raise ValueError("None of the requested models is available")
    y = data[list(OBJECTIVES)].to_numpy(dtype=float)
    per_system, prediction_rows, calibration_rows, audits, recommendations = [], [], [], [], []
    warning_rows, runtime_rows = [], []
    for fold, test_group in enumerate(groups):
        remaining = [group for group in groups if group != test_group]
        calibration_groups = []
        if calibration_enabled and len(remaining) >= 2:
            rng = np.random.default_rng(seed + fold)
            shuffled = list(rng.permutation(remaining))
            count = min(len(remaining) - 1, max(1, int(math.ceil(len(remaining) * calibration_fraction))))
            calibration_groups = sorted(shuffled[:count])
        training_groups = sorted(set(remaining) - set(calibration_groups))
        train_idx = np.flatnonzero(data.object_model.isin(training_groups).to_numpy())
        test_idx = np.flatnonzero((data.object_model == test_group).to_numpy())
        cal_idx = np.flatnonzero(data.object_model.isin(calibration_groups).to_numpy())
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        calibration_records = [records[i] for i in cal_idx]
        normalization = fit_normalization(y[train_idx])
        audits.append({"fold": fold, "test_group": test_group, "training_groups": training_groups, "calibration_groups": calibration_groups, "training_schema_ids": data.iloc[train_idx].schema_id.tolist(), "calibration_schema_ids": data.iloc[cal_idx].schema_id.tolist(), "test_schema_ids": data.iloc[test_idx].schema_id.tolist(), "selection_normalization": normalization, "feature_names": feature_names, "validation": "No hyperparameter tuning; group-held-out calibration only; no candidate-random splits"})
        for label, name, count in active:
            members, calibration_members, captured, stage_times = _fit_predict(name, count, train_records, y[train_idx], test_records, calibration_records, feature_names, seed + fold * 1000, config)
            for stage, seconds in stage_times.items():
                boundary = "Includes neural representation preparation and cache/device work inside the call" if name == "neural" else "Feature arrays extracted separately; fit/predict include their pipeline preprocessing"
                runtime_rows.append({"fold": fold, "test_group": test_group, "prediction_model": label, "variant": "", "stage": stage, "seconds": seconds, "member_count": count, "boundary": boundary})
            if members.shape != (count, len(test_idx), 3) or not np.isfinite(members).all():
                raise ValueError(f"Invalid prediction shape/values from {label}")
            if calibration_members.shape != (count, len(cal_idx), 3) or not np.isfinite(calibration_members).all():
                raise ValueError(f"Invalid calibration prediction shape/values from {label}")
            mean = members.mean(axis=0)
            std = members.std(axis=0, ddof=1) if count > 1 else np.zeros_like(mean)
            calibration_mean = calibration_members.mean(axis=0)
            metrics_started = time.perf_counter()
            regression = _regression_metrics(y[test_idx], mean, y[train_idx])
            uncertainty, radii = _uncertainty_metrics(y[test_idx], mean, std, y[cal_idx], calibration_mean, alpha)
            runtime_rows.append({"fold": fold, "test_group": test_group, "prediction_model": label, "variant": "", "stage": "prediction_metric_calculation", "seconds": time.perf_counter() - metrics_started, "member_count": count, "boundary": "Evaluation-only regression, residual interval and uncertainty-error metrics; reads held-out outcomes"})
            if captured:
                warning_rows.append({"fold": fold, "model": label, "warnings": captured})
            for position, index in enumerate(test_idx):
                row = {"model": label, "object_model": test_group, "schema_id": records[index]["schema_id"], "fold": fold, "measurement_seed_count": int(data.iloc[index].measurement_seed_count), "workload_id": data.iloc[index].workload_id, "environment_id": data.iloc[index].environment_id}
                for j, objective in enumerate(OBJECTIVES):
                    row.update({f"true_{objective}": y[index, j], f"mean_{objective}": mean[position, j], f"std_{objective}": std[position, j], f"lower_{objective}": mean[position, j] - radii[j], f"upper_{objective}": mean[position, j] + radii[j]})
                    for member in range(count):
                        row[f"member_{member}_{objective}"] = members[member, position, j]
                prediction_rows.append(row)
            for position, index in enumerate(cal_idx):
                row = {"model": label, "outer_test_group": test_group, "calibration_object_model": records[index]["object_model"], "schema_id": records[index]["schema_id"], "fold": fold}
                for j, objective in enumerate(OBJECTIVES):
                    row.update({f"true_{objective}": y[index, j], f"mean_{objective}": calibration_mean[position, j], f"absolute_residual_{objective}": abs(y[index, j] - calibration_mean[position, j])})
                calibration_rows.append(row)
            variants = [(label, 0.0)] if label != "ensemble5" else [("ensemble_mean", 0.0), ("ensemble_risk", risk_lambda)]
            for variant, penalty in variants:
                selection_started = time.perf_counter()
                chosen = recommend(mean, std, normalization, weights=config.get("weights"), risk_lambda=penalty, mode=config.get("selection_mode", "pareto"), budget=config.get("beam_budget"), seed=seed + fold, beam_width=config.get("beam_width", 8), max_iterations=config.get("beam_iterations", 20), proposals_per_beam=config.get("proposals_per_beam", 4), exploration_eta=float(config.get("exploration_eta", 0.0)))
                runtime_rows.append({"fold": fold, "test_group": test_group, "prediction_model": label, "variant": variant, "stage": "selection", "seconds": time.perf_counter() - selection_started, "member_count": count, "boundary": "Final-stage finite-pool selection from existing predictions; no synthesis, encoding, model fitting or DB execution"})
                metrics_started = time.perf_counter()
                metrics = recommendation_metrics(y[test_idx], chosen["indices"], chosen["weights"], config.get("hypervolume_reference", [1.1, 1.1, 1.1]), preference_index=chosen["preference_index"], selection_normalization=normalization)
                runtime_rows.append({"fold": fold, "test_group": test_group, "prediction_model": label, "variant": variant, "stage": "recommendation_metric_calculation", "seconds": time.perf_counter() - metrics_started, "member_count": count, "boundary": "Evaluation-only measured-coordinate Pareto/GD/HV/IGD+/regret; excluded from recommendation algorithm runtime"})
                scalar_metrics = {key: value for key, value in metrics.items() if key != "preference_decision" and not isinstance(value, (dict, list))}
                per_system.append({"model": variant, "prediction_model": label, "object_model": test_group, "fold": fold, "training_systems": len(training_groups), "calibration_systems": len(calibration_groups), "test_candidates": len(test_idx), "risk_lambda": penalty, **regression, **uncertainty, **scalar_metrics})
                preference_schema_id = test_records[chosen["preference_index"]]["schema_id"] if chosen["preference_index"] is not None else None
                recommendations.append({"model": variant, "object_model": test_group, "selected_schema_ids": [test_records[i]["schema_id"] for i in chosen["indices"]], "preference_schema_id": preference_schema_id, "selection": chosen, "measured_metrics": metrics})
    predictions = pd.DataFrame(prediction_rows)
    system_metrics = pd.DataFrame(per_system)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    system_metrics.to_csv(output_dir / "per_system_metrics.csv", index=False)
    calibration_columns = ["model", "outer_test_group", "calibration_object_model", "schema_id", "fold"] + [f"{prefix}_{objective}" for objective in OBJECTIVES for prefix in ("true", "mean", "absolute_residual")]
    pd.DataFrame(calibration_rows, columns=calibration_columns).to_csv(output_dir / "calibration_predictions.csv", index=False)
    data.to_csv(output_dir / "aggregated_measurements.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "runtime_by_fold.csv", index=False)
    _json_write(output_dir / "fold_audit.json", audits)
    _json_write(output_dir / "recommendations.json", recommendations)
    _json_write(output_dir / "training_warnings.json", warning_rows)
    available_variants = set(system_metrics.model)
    reference = config.get("reference_model", "ensemble_risk" if "ensemble_risk" in available_variants else sorted(available_variants)[0])
    statistical = compare_system_metrics(system_metrics, reference, config.get("statistical_metrics", ["f1", "gd", "hypervolume", "igd_plus", "preference_regret"]))
    _json_write(output_dir / "paired_statistics.json", statistical)
    numeric = [column for column in system_metrics.columns if column not in {"model", "prediction_model", "object_model", "fold"}]
    for column in numeric:
        system_metrics[column] = pd.to_numeric(system_metrics[column], errors="raise")
    averages = system_metrics.groupby("model", sort=True)[numeric].mean().to_dict(orient="index")
    summary = {"status": "completed", "mode": config.get("mode", "research"), "label_source": str(data.label_source.iloc[0]), "scenario": {"workload_id": str(data.workload_id.iloc[0]), "environment_id": str(data.environment_id.iloc[0])}, "independent_unit": "object_model", "system_count": len(groups), "measured_candidate_count": len(data), "unmeasured_candidate_count": int(data.attrs.get("unmeasured_candidate_count", 0)), "objectives": list(OBJECTIVES), "features": feature_names, "unavailable_models": unavailable, "available_prediction_models": [item[0] for item in active], "ensemble_implementation": ensemble_base, "ensemble_status": "evaluated" if "ensemble5" in {item[0] for item in active} else "unavailable", "statistical_reference_model": reference, "macro_system_means": averages, "nrmse_definition": "per-objective RMSE divided by that outer fold's proper-training target range; overall nRMSE is mean of three; undefined when a training objective is constant", "constant_target_policy": "No epsilon-based fabricated nRMSE; constant-test R2 is undefined; training normalization uses unit scale for constant dimensions", "uncertainty": {"members": 5 if "ensemble5" in {item[0] for item in active} else 0, "std_definition": "sample standard deviation across identical-base independently seeded fitted members", "control": "ensemble_mean and ensemble_risk reuse exactly the same five predictions, normalization and candidate pool", "risk_objectives": "(mean - training_min)/training_range + lambda * std/training_range", "intervals": "empirical absolute-residual quantile on group-held-out calibration candidates; descriptive marginal intervals, no finite-sample conformal guarantee; within-system dependence acknowledged", "nominal_coverage": 1 - alpha, "test_coverage": "reported per source system/objective, evaluated on the outer held-out group"}, "statistics": {"method": "exact two-sided conditional sign-permutation signed-rank with zero removal and midranks for ties", "holm_family": "all requested metrics and comparators against the reference in paired_statistics.json", "caution": "Few independent systems limit power/generalization; candidates and seeds are not independent systems"}, "search_scope": "finite already-generated/encoded/predicted pool; beam budget excludes synthesis and full-pool inference", "files": ["oof_predictions.csv", "per_system_metrics.csv", "calibration_predictions.csv", "runtime_by_fold.csv", "evaluation_runtime.json", "aggregated_measurements.csv", "fold_audit.json", "recommendations.json", "training_warnings.json", "paired_statistics.json"]}
    summary["preference_regret_definition"] = {"decision": "Prediction-only lowest weighted risk score among returned candidates; recorded as preference_index and preference_schema_id", "evaluation": "Realized weighted measured utility of that decision minus the best feasible measured utility, both using the selection weights and proper-training normalization", "undefined": "Null when no explicit preference decision is supplied or the returned set is empty", "oracle_set_regret": "Separate best-measured-member diagnostic using measured-pool normalization; not prediction-based decision regret", "historical_results": "Earlier preference_regret values used the oracle set diagnostic and must be recomputed before comparing with corrected results"}
    _json_write(output_dir / "evaluation_config.json", config)
    runtime = {"clock": "time.perf_counter wall-clock seconds", "total_evaluation_workflow_seconds": time.perf_counter() - workflow_started, "total_boundary": "From evaluate() entry through input validation, aggregation, folds, model fits, predictions, selections, metric/statistical computation and output writes, stopping before the final runtime/summary manifest serialization", "stage_totals_seconds": pd.DataFrame(runtime_rows).groupby("stage").seconds.sum().to_dict(), "shared_ensemble_accounting": "One fit/prediction record for ensemble5; mean/risk variants reuse those predictions and each receive a separate selection record. Do not count shared stages twice.", "neural_boundary": "Representation preparation, checkpoint/cache operations and device transfers performed by fit/predict APIs are included in their respective call timers; feature_preparation=0 means included within these calls, not absent", "excluded_before_evaluation": ["object-model acquisition", "schema synthesis/enumeration", "schema encoding done before evaluate()", "database setup/data loading and measurement profiling"], "comparison_rule": "These evaluation times cannot establish end-to-end speedups against competitors unless all methods report matching acquisition/synthesis/representation/training/inference/selection boundaries on the same workload and hardware", "overhead": "Input loading, fold assembly, normalization, aggregation, statistics, bookkeeping and serialization are included in workflow total but not assigned to model stage records"}
    summary["runtime"] = runtime
    summary = _json_clean(summary)
    _json_write(output_dir / "evaluation_runtime.json", runtime)
    _json_write(output_dir / "summary.json", summary)
    return summary
