"""Fit deployable structural baselines without reading target labels at recommendation time."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .common import TARGETS, aligned_training_data, environment, read_jsonl, validate_output_schema_ids, write_json
from .recommendation import recommend, fit_normalization


def make_estimator(name, seed, trees=100):
    if name in {"linear", "linear_regression"}:
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=trees, random_state=seed, n_jobs=1)
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed, early_stopping=False)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError as exc:
            raise RuntimeError("Install the xgboost extra; no alternative model is substituted.") from exc
        return MultiOutputRegressor(XGBRegressor(n_estimators=trees, random_state=seed, n_jobs=1))
    raise ValueError(f"Unsupported structural model: {name}")


def train(candidates_path, measurements_path, model_out, config):
    started = time.perf_counter()
    mode = config.get("mode", "research")
    records, X, y, report = aligned_training_data(candidates_path, measurements_path, mode)
    x_scaler, y_scaler = StandardScaler().fit(X), StandardScaler().fit(y)
    name = config.get("model", "extra_trees")
    size = int(config.get("ensemble_size", 5))
    if size < 1:
        raise ValueError("ensemble_size must be positive")
    seed = int(config.get("seed", 42))
    models = []
    for member in range(size):
        model = make_estimator(name, seed + member, int(config.get("trees", 100)))
        model.fit(x_scaler.transform(X), y_scaler.transform(y))
        models.append(model)
    payload = {"format": "ymap_structural_ensemble_v1", "models": models, "x_scaler": x_scaler,
               "y_scaler": y_scaler, "feature_names": report["feature_names"], "targets": TARGETS,
               "mode": mode, "model": name, "normalization": fit_normalization(y),
               "training_schema_ids": [row["schema_id"] for row in records], "training_provenance": report}
    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    metadata = {key: value for key, value in payload.items() if key not in {"models", "x_scaler", "y_scaler"}}
    metadata.update({"elapsed_fit_seconds": time.perf_counter() - started, "environment": environment(),
                     "evaluation_note": "Fitted on all supplied data for deployment; use evaluate for held-out performance."})
    write_json(str(path) + ".json", metadata)
    return metadata


def predict_and_recommend(candidates_path, model_path, out_dir, weights=None, risk_lambda=0., selection_mode="pareto", budget=None, seed=42, allow_demo=False):
    # joblib/pickle models execute Python when loaded. Only load artifacts you created or trust.
    payload = joblib.load(model_path)
    if payload.get("format") != "ymap_structural_ensemble_v1":
        raise ValueError("Unrecognized model format")
    if payload["mode"] != "research" and not allow_demo:
        raise ValueError("Demo-trained model requires --allow-demo; it cannot be presented as a research model.")
    records = read_jsonl(candidates_path)
    if not records:
        raise ValueError("No candidates")
    validate_output_schema_ids(records)
    if len({record["object_model"] for record in records}) != 1:
        raise ValueError("Recommend for one source object model at a time")
    names = payload["feature_names"]
    X = np.asarray([[row["features"][name] for name in names] for row in records], dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Candidate features must be finite")
    started = time.perf_counter()
    transformed = payload["x_scaler"].transform(X)
    members = np.stack([payload["y_scaler"].inverse_transform(model.predict(transformed)) for model in payload["models"]])
    if not np.isfinite(members).all():
        raise ValueError("Model returned non-finite predictions")
    prediction_seconds = time.perf_counter() - started
    mean = members.mean(axis=0)
    std = members.std(axis=0, ddof=1) if len(members) > 1 else np.zeros_like(mean)
    result = recommend(mean, std, payload["normalization"], weights=weights, risk_lambda=risk_lambda,
                       mode=selection_mode, budget=budget, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"schema_id": [row["schema_id"] for row in records], "object_model": [row["object_model"] for row in records]})
    selected = set(int(i) for i in result["indices"])
    for j, target in enumerate(TARGETS):
        frame[f"pred_{target}"] = mean[:, j]
        frame[f"std_{target}"] = std[:, j]
    frame["selected"] = [i in selected for i in range(len(records))]
    frame.to_csv(out / "predictions.csv", index=False)
    frame.loc[frame.selected].to_csv(out / "recommendations.csv", index=False)
    for i in sorted(selected):
        (out / f"{records[i]['schema_id']}.sql").write_text(records[i]["ddl"], encoding="utf-8")
    report = {"model_mode": payload["mode"], "source_model": records[0]["object_model"], "selection_mode": selection_mode,
              "selected_schema_ids": [records[i]["schema_id"] for i in sorted(selected)], "prediction_seconds": prediction_seconds,
              "normalization": payload["normalization"], "selection_provenance": result.get("provenance", {}),
              "trace": result.get("trace", []), "claim_boundary": "Predictions under training scenario; no measurements used by this command."}
    write_json(out / "recommendation_manifest.json", report)
    return report
