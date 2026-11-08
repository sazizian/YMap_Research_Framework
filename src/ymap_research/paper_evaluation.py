"""Evaluate the manuscript method against actual measurements from one protocol.

The test system is used for scoring only. Selection scales come from candidate
predictions, while reported regret and front quality use measured-pool scales.
This workflow does not reconstruct historical results or implement the external
research baselines: their selected schema IDs must be supplied with provenance.
"""
from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .common import RESEARCH_SOURCES, environment, json_ready, sha256_file, write_json
from .evaluation import OBJECTIVES, feature_matrix, load_dataset
from .recommendation import normalize_predictions, pareto_mask, recommendation_metrics
from .statistics import holm_adjust, paired_wilcoxon


PAPER_SEEDS = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14))
RISK_LEVELS = (0.0, 0.5, 1.0, 2.0)
BASELINE_GRIDS = {
    "training_mean": ({},),
    "ridge": tuple({"alpha": alpha} for alpha in (0.01, 0.1, 1, 10, 100)),
    "random_forest": tuple(
        {"n_estimators": trees, "max_depth": depth}
        for trees, depth in product((100, 300, 500), (None, 10, 20))
    ),
    "xgboost": tuple(
        {"n_estimators": trees, "max_depth": depth, "learning_rate": rate}
        for trees, depth, rate in product((100, 300, 600), (3, 6, 9), (0.03, 0.1))
    ),
    "mlp": tuple(
        {"hidden_layer_sizes": layers, "alpha": alpha}
        for layers, alpha in product(((64,), (128, 64)), (1e-4, 1e-3))
    ),
}
RESEARCH_COMPARATORS = ("dts", "triorm", "leant")
FRAMEWORK_COMPARATORS = ("hibernate", "django", "entity_framework_core")
PRIMARY_METRICS = ("hypervolume", "igd_plus", "f1")


def _objective_matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty, finite n-by-3 matrix")
    return values


def candidate_normalization(means, training_objectives):
    """Use candidate means for scale and origin without looking at test labels.

    When an objective has zero prediction span, use its training measured range
    or, if that is also zero, its positive training magnitude. Both fallbacks
    are in the objective's units; a change of units leaves scores unchanged.
    """
    means = _objective_matrix(means, "Candidate means")
    train = _objective_matrix(training_objectives, "Training objectives")
    minimum = means.min(axis=0)
    span = np.ptp(means, axis=0)
    train_span = np.ptp(train, axis=0)
    train_magnitude = np.max(np.abs(train), axis=0)
    scale = np.where(span > 0, span, np.where(train_span > 0, train_span, train_magnitude))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("A zero predicted range needs a positive training range or magnitude")
    return {
        "minimum": minimum.tolist(),
        "scale": scale.tolist(),
        "constant": (span == 0).tolist(),
        "source": "candidate_prediction_means",
        "scale_source": np.where(span > 0, "prediction_range", np.where(train_span > 0, "training_range", "training_magnitude")).tolist(),
    }


def gaussian_multiplier(validation_truth, validation_means, validation_std):
    """Return the per-objective Gaussian MLE scale of Eq. (5).

    A zero validation spread makes the equation undefined for that objective.
    Keep it unavailable (NaN, serialized as null) instead of silently adding an
    arbitrary floor. A perfect validation fit returns the zero-scale limit.
    """
    truth = _objective_matrix(validation_truth, "Validation objectives")
    mean = _objective_matrix(validation_means, "Validation means")
    std = _objective_matrix(validation_std, "Validation spread")
    if truth.shape != mean.shape or truth.shape != std.shape or np.any(std < 0):
        raise ValueError("Validation objectives, means, and nonnegative spreads must align")
    valid = (std > 0).all(axis=0)
    scale = np.full(3, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (truth[:, valid] - mean[:, valid]) / std[:, valid]
        # Scale the RMS calculation to avoid overflow when squaring.
        peak = np.max(np.abs(residual), axis=0)
        scaled = np.divide(residual, peak, out=np.zeros_like(residual), where=peak > 0)
        scale[valid] = peak * np.sqrt(np.mean(scaled ** 2, axis=0))
    return np.where(np.isfinite(scale), scale, np.nan)


def preference_weights():
    """Return all 66 three-objective simplex points at step 0.1."""
    return np.array([(i, j, 10 - i - j) for i in range(11) for j in range(11 - i)], dtype=float) / 10


def _preference_decisions(risk):
    scores = risk @ preference_weights().T
    # Candidate rows are sorted by schema ID by load_dataset().
    chosen = scores.argmin(axis=0)
    return chosen, scores[chosen, np.arange(len(chosen))]


def _realized_regrets(truth, chosen, selection_normalization=None):
    truth = _objective_matrix(truth, "Measured objectives")
    weights = preference_weights()
    measured_span = np.ptp(truth, axis=0)
    measured_points = np.divide(
        truth - truth.min(axis=0), measured_span,
        out=np.zeros_like(truth), where=measured_span > 0,
    )
    def regret(points):
        utility = points @ weights.T
        return utility[chosen, np.arange(len(weights))] - utility.min(axis=0)

    measured_regret = regret(measured_points)
    if selection_normalization is None:
        return measured_regret, None
    selection_scale, _ = normalize_predictions(truth, None, selection_normalization)
    return measured_regret, regret(selection_scale)


def _rho(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def prediction_metrics(truth, prediction):
    """Per-objective metrics in raw units, using held-out measured ranges."""
    truth = _objective_matrix(truth, "Measured objectives")
    prediction = _objective_matrix(prediction, "Predicted objectives")
    if truth.shape != prediction.shape or np.any(truth <= 0):
        raise ValueError("Measured objectives must be positive and align with predictions")
    output = []
    for objective, actual, expected in zip(OBJECTIVES, truth.T, prediction.T):
        error = expected - actual
        residual = float(error @ error)
        rmse = float(np.sqrt(residual / len(actual)))
        span = float(np.ptp(actual))
        centered = actual - actual.mean()
        total = float(centered @ centered)
        output.append({
            "objective": objective, "rmse": rmse,
            "nrmse": rmse / span if span > 0 else None,
            "r2": 1 - residual / total if total > 0 else None,
            "mdape_percent": float(100 * np.median(np.abs(error) / actual)),
            "spearman": _rho(actual, expected),
        })
    return output


def _uncertainty_metrics(truth, means, spread, factors):
    raw_std = spread
    recal_std = spread * factors
    output = []
    for j, objective in enumerate(OBJECTIVES):
        errors = np.abs(truth[:, j] - means[:, j])
        row = {"objective": objective, "gaussian_multiplier": factors[j],
               "error_uncertainty_spearman": _rho(errors, raw_std[:, j])}
        for label, std in (("raw", raw_std), ("recalibrated", recal_std)):
            for name, z in (("68_3", 1.0), ("95", 1.96)):
                row[f"{label}_coverage_{name}"] = (
                    float(np.mean(errors <= z * std[:, j])) if np.isfinite(std[:, j]).all() else None
                )
        output.append(row)
    return output


def _baseline_estimator(name, params, seed):
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(**params, n_jobs=1, random_state=seed)
    if name == "mlp":
        return MLPRegressor(**params, max_iter=500, early_stopping=False, random_state=seed)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install the xgboost extra to run the manuscript's XGBoost baseline") from exc
        return XGBRegressor(**params, objective="reg:squarederror", n_jobs=1, random_state=seed)
    raise ValueError(f"Unknown paper baseline: {name}")


def _descriptor_baselines(x, y, train, val, test, names, seed):
    scaler = StandardScaler().fit(x[train])
    x_train, x_val, x_test = (scaler.transform(x[rows]) for rows in (train, val, test))
    log_train = np.log(y[train])
    targets = StandardScaler().fit(log_train)
    z_train = targets.transform(log_train)
    z_val = targets.transform(np.log(y[val]))
    for name in names:
        started = time.perf_counter()
        predictions = np.empty((len(x_test), 3))
        choices = {}
        for j, objective in enumerate(OBJECTIVES):
            if name == "training_mean":
                predictions[:, j] = z_train[:, j].mean()
                choices[objective] = {"parameters": {}, "validation_standardized_log_mse": float(np.mean((z_val[:, j] - predictions[0, j]) ** 2))}
                continue
            best_score = np.inf
            best = None
            for params in BASELINE_GRIDS[name]:
                estimator = _baseline_estimator(name, params, seed)
                estimator.fit(x_train, z_train[:, j])
                score = float(np.mean((estimator.predict(x_val) - z_val[:, j]) ** 2))
                if score < best_score:
                    best_score, best, best_params = score, estimator, params
            if best is None:
                raise ValueError(f"{name} returned no finite validation score for {objective}")
            predictions[:, j] = best.predict(x_test)
            choices[objective] = {"parameters": best_params, "validation_standardized_log_mse": best_score}
        yield name, np.exp(targets.inverse_transform(predictions)), choices, time.perf_counter() - started


def _jsonl_row(handle, row):
    handle.write(json.dumps(json_ready(row), separators=(",", ":"), allow_nan=False) + "\n")


def _ids(value, name):
    if not isinstance(value, (tuple, list)) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} cannot contain duplicate IDs")
    return tuple(value)


def _load_paper_data(candidates_path, measurements_path, config):
    # Reuse the shared ingestion checks for scenario, IDs, seeds and measured
    # labels; the manuscript label uses a median rather than the legacy mean.
    allowed = config.get("allowed_label_sources", RESEARCH_SOURCES)
    if not isinstance(allowed, (list, tuple, set, frozenset)) or set(allowed) - RESEARCH_SOURCES:
        raise ValueError("allowed_label_sources must contain documented measured sources")
    records, data, _ = load_dataset(candidates_path, measurements_path, {
        **config, "mode": "research", "allowed_label_sources": allowed,
    })
    scenario = {key: str(data[key].iloc[0]) for key in ("workload_id", "environment_id")}
    raw = pd.read_csv(measurements_path, keep_default_na=False, dtype={key: str for key in ("schema_id", *scenario)})
    for key, value in scenario.items():
        raw = raw[raw[key] == value]
    for objective in OBJECTIVES:
        raw[objective] = pd.to_numeric(raw[objective], errors="raise")
    if np.any(raw[list(OBJECTIVES)].to_numpy(dtype=float) <= 0):
        raise ValueError("Paper evaluation requires strictly positive measured outcomes for log targets")
    medians = raw.groupby("schema_id")[list(OBJECTIVES)].median()
    data[list(OBJECTIVES)] = medians.loc[data.schema_id].to_numpy()
    return records, data, scenario


def _settings(config, groups):
    scope = config.get("scope", "paper")
    if scope not in ("paper", "ablation", "smoke"):
        raise ValueError("scope must be paper, ablation, or smoke")
    full = scope == "paper"
    default_order = None if full else sorted(groups)
    order = _ids(config.get("system_order", default_order), "system_order")
    if len(order) < 3 or set(order) - groups:
        raise ValueError("system_order must contain at least three measured system IDs")
    if full and (len(order) != 11 or set(order) != groups):
        raise ValueError("Paper scope requires exactly 11 measured systems in an explicit cyclic order")
    shared = _ids(config.get("shared_systems", None if full else []), "shared_systems")
    if (full and len(shared) != 9) or set(shared) - set(order):
        raise ValueError("shared_systems must be within system_order, with exactly nine in paper scope")
    raw_seeds = config.get("repetition_seeds", PAPER_SEEDS if full else ((0,), (5,), (10,)) if scope == "ablation" else ((0,),))
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
        raise ValueError("repetition_seeds must contain at least one nonempty seed list")
    seeds = []
    for row in raw_seeds:
        if not isinstance(row, (list, tuple)) or not row or any(type(seed) is not int or seed < 0 for seed in row):
            raise ValueError("Each repetition requires a list of nonnegative integer member seeds")
        seeds.append(tuple(row))
    seeds = tuple(seeds)
    if len(set(map(len, seeds))) != 1 or len(set(sum(seeds, ()))) != sum(map(len, seeds)):
        raise ValueError("Repetitions require equal member counts and distinct seeds")
    if full and seeds != PAPER_SEEDS:
        raise ValueError("Paper scope requires three fixed five-member seed sets (0–4, 5–9, 10–14)")
    baselines = _ids(config.get("prediction_baselines", list(BASELINE_GRIDS) if full else ["training_mean", "ridge"] if scope == "smoke" else []), "prediction_baselines")
    if set(baselines) - BASELINE_GRIDS.keys() or (full and set(baselines) != BASELINE_GRIDS.keys()):
        raise ValueError("Paper scope requires all five manuscript prediction baselines")
    views = config.get("views", ("graph", "symbolic") if scope == "smoke" else None)
    if (full and views is not None) or (scope == "ablation" and views is None):
        raise ValueError("Use all five views for paper scope, or specify views explicitly for ablation")
    if views is not None:
        views = _ids(views, "views")
        if not views and scope != "smoke":
            raise ValueError("views cannot be empty")
    if full and config.get("neural", {}).get("fusion", "gated") != "gated":
        raise ValueError("Paper scope requires gated fusion; use ablation for concatenation")
    return scope, order, shared, seeds, baselines, views


def _load_comparators(source, scenario, order, repetitions, ids_by_system):
    if source is None:
        return {}
    document = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or any(document.get(k) != v for k, v in scenario.items()):
        raise ValueError("Comparator selections must declare the same workload_id and environment_id")
    methods = document.get("methods")
    if not isinstance(methods, dict) or set(methods) - set(RESEARCH_COMPARATORS + FRAMEWORK_COMPARATORS):
        raise ValueError("Unknown comparator method in supplied selections")
    validated = {}
    for name, specification in methods.items():
        if not isinstance(specification, dict):
            raise ValueError(f"{name} needs a provenance and a systems mapping")
        provenance = specification.get("provenance")
        systems = specification.get("systems")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError(f"Comparator {name} requires provenance for its supplied selections")
        if not isinstance(systems, dict) or set(systems) - set(order):
            raise ValueError(f"Comparator {name} references unknown systems")
        fronts = {}
        for system, repeat_fronts in systems.items():
            if not isinstance(repeat_fronts, list) or len(repeat_fronts) != repetitions:
                raise ValueError(f"Comparator {name}/{system} needs one ID list per repetition")
            fronts[system] = tuple(_ids(row, f"{name}/{system} schema IDs") for row in repeat_fronts)
            if any(set(ids) - ids_by_system[system] for ids in fronts[system]):
                raise ValueError(f"Comparator {name}/{system} references IDs outside its measured pool")
            if name in FRAMEWORK_COMPARATORS and any(len(ids) != 1 for ids in fronts[system]):
                raise ValueError(f"Framework default {name} must return exactly one schema")
        preferences = specification.get("preferences", {})
        if not isinstance(preferences, dict) or set(preferences) - set(systems) or (name in (*FRAMEWORK_COMPARATORS, "dts") and preferences):
            raise ValueError(f"Comparator {name} cannot have the supplied preference decisions")
        for system, repeat_decisions in preferences.items():
            if not isinstance(repeat_decisions, list) or len(repeat_decisions) != repetitions:
                raise ValueError(f"Comparator {name}/{system} needs one preference list per repetition")
            if any(not isinstance(row, list) or len(row) != 66 or any(not isinstance(v, str) or v not in ids_by_system[system] for v in row) for row in repeat_decisions):
                raise ValueError(f"Comparator {name}/{system} needs 66 known schema IDs in preference_weights() order")
        validated[name] = {"provenance": provenance, "systems": fronts, "preferences": preferences}
    return validated


def _hodges_lehmann_ci(differences):
    """Invert exact two-sided signed-rank tests at alpha=.05.

    Signed ranks change only at Walsh averages. Check open regions and their
    boundary points (where ties or zeros may change the conditional test).
    Report the closed hull of accepted shifts; null bounds mean unbounded.
    """
    data = [Fraction.from_float(float(d)) for d in differences]
    walsh = sorted((a + b) / 2 for a, b in combinations_with_replacement(data, 2))
    hl = float((walsh[(len(walsh) - 1) // 2] + walsh[len(walsh) // 2]) / 2)
    breaks = sorted(set(walsh))
    bounds = [None, *breaks, None]
    accepted = []

    def accepts(shift):
        residual = [value - shift for value in data]
        # Ordinal absolute magnitudes preserve rank and exact rational ties.
        ranks = {value: i for i, value in enumerate(sorted(set(map(abs, residual))))}
        signs = [0 if d == 0 else (ranks[abs(d)] + 1) * (1 if d > 0 else -1) for d in residual]
        return paired_wilcoxon(signs, np.zeros(len(signs)))["p_value"] > 0.05

    for lower, upper in zip(bounds, bounds[1:]):
        probe = upper - 1 if lower is None else lower + 1 if upper is None else (lower + upper) / 2
        if accepts(probe):
            accepted.append((lower, upper))
        if upper is not None and accepts(upper):
            accepted.append((upper, upper))
    low, high = accepted[0][0], accepted[-1][1]
    return hl, [None if value is None else float(value) for value in (low, high)]


def _paired_family(recommendations, shared, scope):
    if scope != "paper":
        return {"status": "not_run", "reason": "Confirmatory tests require full paper scope", "tests": []}
    eligible = recommendations[
        recommendations.model.isin(("ymap", *RESEARCH_COMPARATORS))
        & recommendations.variant.isin(("ensemble", "supplied"))
        & (recommendations.kappa.fillna(0) == 0)
        & recommendations.object_model.isin(shared)
    ]
    measures = eligible.set_index(["model", "object_model"])
    required = pd.MultiIndex.from_product((("ymap", *RESEARCH_COMPARATORS), shared))
    incomplete = measures.reindex(required)
    if incomplete[list(PRIMARY_METRICS)].isna().any().any() or (incomplete.repeats != len(PAPER_SEEDS)).any():
        return {"status": "not_run", "reason": "The entire nine-test family requires complete selections from DTS, TriORM and LEANT on all nine shared systems", "tests": []}
    results = []
    for metric, name in product(PRIMARY_METRICS, RESEARCH_COMPARATORS):
        ymap = measures.loc["ymap", metric].reindex(shared).to_numpy()
        other = measures.loc[name, metric].reindex(shared).to_numpy()
        differences = ymap - other
        direction = -1 if metric == "igd_plus" else 1
        stats = paired_wilcoxon(ymap, other)
        hl, ci = _hodges_lehmann_ci(differences)
        stats.update({
            "reference": "ymap", "comparator": name, "metric": metric,
            "systems": list(shared), "unit_of_analysis": "object_model",
            "wins": int(np.sum(direction * differences > 0)),
            "rank_biserial_favoring_reference": direction * stats["rank_biserial"],
            "hodges_lehmann_ymap_minus_comparator": hl, "hodges_lehmann_exact_95_ci": ci,
            "ci_method": "closed_hull_inverting_exact_two_sided_signed_rank_tests_null_bound_is_unbounded",
        })
        results.append(stats)
    for result, adjusted in zip(results, holm_adjust([r["p_value"] for r in results])):
        result.update(holm_p_value=adjusted, holm_family_size=len(results))
    return {"status": "completed", "tests": results}


def _front_metrics(truth, selected):
    values = recommendation_metrics(truth, selected)
    values.pop("preference_regret")
    values.pop("preference_decision")
    values.pop("oracle_set_regret")
    values.pop("metric_normalization")
    values.pop("hypervolume_reference")
    values["hv_ratio"] = values["hypervolume"] / values["true_hypervolume"]
    return values


def _front_eval(context, model, variant, mean, std, train_y, truth, ids, true_ids, levels, fronts, preferences):
    normalization = candidate_normalization(mean, train_y)
    mean_norm, std_norm = normalize_predictions(mean, std, normalization)
    neutral = pareto_mask(mean)
    output = []
    for kappa in levels:
        risk = mean_norm + kappa * std_norm
        mask = neutral if kappa == 0 else pareto_mask(risk)
        selected = np.flatnonzero(mask)
        decisions, scores = _preference_decisions(risk)
        regret, same_scale = _realized_regrets(truth, decisions, normalization)
        metadata = {**context, "model": model, "variant": variant, "kappa": kappa}
        output.append({
            **metadata, **_front_metrics(truth, selected),
            "front_symmetric_difference": int(np.count_nonzero(mask != neutral)),
            "front_added": int(np.count_nonzero(mask & ~neutral)),
            "front_removed": int(np.count_nonzero(~mask & neutral)),
            "preference_regret": float(regret.mean()),
            "preference_regret_on_selection_scale": float(same_scale.mean()),
        })
        _jsonl_row(fronts, {**metadata, "selected_schema_ids": ids[selected], "measured_front_schema_ids": true_ids, "selection_normalization": normalization})
        for w, decision, score, r, s in zip(preference_weights(), decisions, scores, regret, same_scale):
            _jsonl_row(preferences, {**metadata, "weights": w, "selected_schema_id": ids[decision],
                                    "predicted_weighted_score": score, "realized_regret": r,
                                    "realized_regret_on_selection_scale": s})
    return output


def _write_predictions(writer, context, model, ids, truth, means, std=None, calibrated=None):
    if std is None:
        std = np.full_like(means, np.nan)
    if calibrated is None:
        calibrated = np.full_like(means, np.nan)
    for schema_id, observed, expected, raw, adjusted in zip(ids, truth, means, std, calibrated):
        writer.writerow({
            **context, "model": model, "schema_id": schema_id,
            **{f"true_{q}": y for q, y in zip(OBJECTIVES, observed)},
            **{f"pred_{q}": y for q, y in zip(OBJECTIVES, expected)},
            **{f"raw_std_{q}": y if np.isfinite(y) else "" for q, y in zip(OBJECTIVES, raw)},
            **{f"recalibrated_std_{q}": y if np.isfinite(y) else "" for q, y in zip(OBJECTIVES, adjusted)},
        })


def _reports(output, stem, rows, keys, shared):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{stem}_per_repetition.csv", index=False)
    metric_fields = [k for k in frame.columns if k not in {"object_model", "fold", "repeat", "validation_system", "scope", "workload_id", "environment_id", *keys}]
    for col in metric_fields:
        frame[col] = pd.to_numeric(frame[col])
    group_keys = ["object_model", *keys]
    groups = frame.groupby(group_keys, dropna=False, sort=False)
    # A partially defined per-system mean is not the stipulated average over
    # all repetitions. Keep the mean undefined and publish the usable counts.
    system = groups[metric_fields].mean().where(groups[metric_fields].count().eq(groups.size(), axis=0))
    for col in metric_fields:
        system[f"{col}_defined_repeats"] = groups[col].count()
    system["repeats"] = groups.size()
    system = system.reset_index()
    system.to_csv(output / f"{stem}_per_system.csv", index=False)
    overall = []
    subsets = [("all", system)]
    if shared:
        subsets.append(("shared", system[system.object_model.isin(shared)]))
    for label, subset in subsets:
        grouped = subset.groupby(list(keys), dropna=False, sort=False)
        summary = grouped[metric_fields].mean()
        for col in metric_fields:
            summary[f"{col}_defined_systems"] = grouped[col].count()
        summary["systems_present"] = grouped.size()
        summary["systems_expected"] = len(set(system.object_model)) if label == "all" else len(shared)
        summary["system_set"] = label
        overall.append(summary.reset_index())
    pd.concat(overall, ignore_index=True).to_csv(output / f"{stem}_summary.csv", index=False)
    return system


def _row_requirements(data, config, scope):
    mode = config.get("measurement_rows", "raw_repetitions")
    if mode not in ("raw_repetitions", "protocol_medians"):
        raise ValueError("measurement_rows must be raw_repetitions or protocol_medians")
    if mode == "protocol_medians" and (data.measurement_seed_count != 1).any():
        raise ValueError("protocol_medians requires one precomputed median row per candidate")
    if scope == "paper":
        if data.attrs["unmeasured_candidate_count"]:
            raise ValueError("Paper scope requires every candidate to have a matched measurement")
        provenance = config.get("protocol_provenance")
        if not isinstance(provenance, (str, dict)) or not provenance or (isinstance(provenance, str) and not provenance.strip()):
            raise ValueError("Paper scope requires documented reference protocol provenance")
        if mode == "raw_repetitions" and (data.measurement_seed_count != 5).any():
            raise ValueError("Paper scope requires five measured repetitions per candidate or documented protocol medians")
    return mode


def _validate_canonical_pool(records):
    hashes = set()
    for record in records:
        ddl = record.get("paper_ddl")
        if not isinstance(ddl, str) or not ddl.strip():
            raise ValueError("Paper scope requires canonical Stage B paper_ddl for every measured candidate")
        fingerprint = hashlib.sha256(ddl.encode("utf-8")).hexdigest()
        if record.get("paper_ddl_sha256", fingerprint) != fingerprint:
            raise ValueError(f"Canonical DDL fingerprint mismatch for {record['schema_id']}")
        if fingerprint in hashes:
            raise ValueError("Duplicate canonical DDL in paper scope; reconcile the candidate pool before evaluation")
        hashes.add(fingerprint)


def evaluate_paper(candidates_path, measurements_path, out_dir, config=None):
    """Run cyclic held-out-system evaluation on one documented measured scenario.

    Full scope requires an explicit 11-system order and nine shared system IDs.
    Ablation uses explicit views. Smoke can disable the neural model via views=[].
    Optional comparator JSON provides its scenario, method provenance, and a
    list of selected candidate ID lists (one per repetition) for each system.
    """
    import csv

    config = dict(config or {})
    start = time.perf_counter()
    records, data, scenario = _load_paper_data(candidates_path, measurements_path, config)
    scope, order, shared, repeats, baselines, views = _settings(config, set(data.object_model))
    measurement_rows = _row_requirements(data, config, scope)
    if scope == "paper":
        _validate_canonical_pool(records)
    raw_levels = config.get("risk_levels", RISK_LEVELS)
    if not isinstance(raw_levels, (list, tuple)) or any(type(k) not in (int, float) or not np.isfinite(k) or k < 0 for k in raw_levels):
        raise ValueError("risk_levels must contain nonnegative finite numbers")
    levels = tuple(dict.fromkeys((0.0, *raw_levels)))
    if scope == "paper" and levels != RISK_LEVELS:
        raise ValueError("Paper scope requires the kappa grid 0, 0.5, 1, 2")
    run_neural = views != ()
    if run_neural:
        from .paper_model import fit_predict_paper
    elif not baselines:
        raise ValueError("An evaluation requires at least one model or prediction baseline")
    groups = data.object_model.to_numpy()
    y = data[list(OBJECTIVES)].to_numpy(dtype=float)
    schema_ids = data.schema_id.to_numpy()
    group_positions = {system: np.flatnonzero(groups == system) for system in order}
    comparators = _load_comparators(config.get("comparator_selections"), scenario, order, len(repeats),
                                    {system: set(schema_ids[pos]) for system, pos in group_positions.items()})
    x = None
    feature_names = None
    descriptor_source = config.get("descriptor_source", "declared" if not run_neural else "paper")
    if descriptor_source not in ("paper", "declared") or (descriptor_source == "declared" and (scope != "smoke" or run_neural)):
        raise ValueError("Declared native features are permitted only in a descriptor-only smoke run")
    if baselines:
        if descriptor_source == "paper":
            from .paper_schema import PAPER_FEATURES, paper_features

            feature_names = PAPER_FEATURES
            values = [paper_features(record) for record in records]
            x = np.asarray([[row[name] for name in PAPER_FEATURES] for row in values], dtype=float)
        else:
            feature_names = sorted(set().union(*(record["features"] for record in records)))
            x = feature_matrix(records, feature_names)
        if not np.isfinite(x).all():
            raise ValueError("Structural descriptors must be present and finite for every measured schema")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions, recommendation, uncertainty, members = [], [], [], []
    folds, fits, unavailable = [], [], []
    model = {"paper": "ymap", "ablation": "ymap_ablation", "smoke": "ymap_smoke"}[scope]
    with ExitStack() as stack:
        def open_output(name):
            return stack.enter_context((output / name).open("w", encoding="utf-8", newline=""))

        fronts = open_output("fronts.jsonl")
        preferences = open_output("preferences.jsonl")
        member_output = open_output("member_predictions.jsonl")
        validation_output = open_output("validation_predictions.jsonl")
        fields = ["scope", "fold", "repeat", "object_model", "validation_system", *scenario, "model", "schema_id"]
        fields += [f"{prefix}_{q}" for prefix in ("true", "pred", "raw_std", "recalibrated_std") for q in OBJECTIVES]
        writer = csv.DictWriter(open_output("oof_predictions.csv"), fieldnames=fields)
        writer.writeheader()
        for i, system in enumerate(order):
            validation_system = order[(i + 1) % len(order)]
            training_systems = [g for g in order if g not in (system, validation_system)]
            train = np.concatenate([group_positions[g] for g in training_systems])
            val = group_positions[validation_system]
            test = group_positions[system]
            ids = schema_ids[test]
            true_positions = np.flatnonzero(pareto_mask(y[test]))
            true_ids = ids[true_positions]
            id_to_position = {schema_id: j for j, schema_id in enumerate(ids)}
            folds.append({"fold": i + 1, "test_system": system, "validation_system": validation_system,
                          "training_systems": training_systems, "training_candidates": len(train),
                          "validation_candidates": len(val), "test_candidates": len(test)})
            for r, seeds in enumerate(repeats, 1):
                ctx = {"scope": scope, "fold": i + 1, "repeat": r, "object_model": system,
                       "validation_system": validation_system, **scenario}
                for name, baseline, choices, seconds in _descriptor_baselines(x, y, train, val, test, baselines, seeds[0]) if baselines else ():
                    fits.append({**ctx, "model": name, "seed": seeds[0], "choices": choices, "elapsed_seconds": seconds})
                    _write_predictions(writer, ctx, name, ids, y[test], baseline)
                    predictions.extend({**ctx, "model": name, **m} for m in prediction_metrics(y[test], baseline))
                    if scope == "smoke":
                        recommendation.extend(_front_eval(ctx, name, "structural_smoke", baseline, None, y[train], y[test], ids, true_ids, (0.0,), fronts, preferences))
                if run_neural:
                    started = time.perf_counter()
                    test_members, val_members, metadata = fit_predict_paper(
                        [records[j] for j in train], y[train], [records[j] for j in val], y[val],
                        [records[j] for j in test], config, seeds, views=views,
                    )
                    test_members = np.asarray(test_members, dtype=float)
                    val_members = np.asarray(val_members, dtype=float)
                    for values, size in ((test_members, len(test)), (val_members, len(val))):
                        if values.shape != (len(seeds), size, 3) or not np.isfinite(values).all() or np.any(values <= 0):
                            raise ValueError("Neural member predictions must be positive, finite and aligned with seeds and candidates")
                    fits.append({**ctx, "model": model, "member_seeds": seeds, "details": metadata,
                                 "elapsed_seconds": time.perf_counter() - started})
                    means = test_members.mean(axis=0)
                    val_means = val_members.mean(axis=0)
                    std = test_members.std(axis=0, ddof=1) if len(seeds) > 1 else None
                    val_std = val_members.std(axis=0, ddof=1) if len(seeds) > 1 else None
                    factors = gaussian_multiplier(y[val], val_means, val_std) if val_std is not None else None
                    calibrated = std * factors if factors is not None else None
                    _write_predictions(writer, ctx, model, ids, y[test], means, std, calibrated)
                    predictions.extend({**ctx, "model": model, **m} for m in prediction_metrics(y[test], means))
                    for j, seed in enumerate(seeds):
                        members.extend({**ctx, "model": model, "seed": seed, **m} for m in prediction_metrics(y[test], test_members[j]))
                    for j, schema_id in enumerate(ids):
                        _jsonl_row(member_output, {**ctx, "model": model, "schema_id": schema_id,
                                                  "member_seeds": seeds, "predictions": test_members[:, j], "measured": y[test[j]]})
                    for j, schema_id in enumerate(schema_ids[val]):
                        _jsonl_row(validation_output, {**ctx, "model": model, "schema_id": schema_id,
                                                      "member_seeds": seeds, "predictions": val_members[:, j],
                                                      "measured": y[val[j]], "mean": val_means[j],
                                                      "raw_std": None if val_std is None else val_std[j]})
                    if factors is not None:
                        uncertainty.extend({**ctx, "model": model, **m} for m in _uncertainty_metrics(y[test], means, std, factors))
                    usable = calibrated is not None and np.isfinite(calibrated).all()
                    if not usable and len(seeds) > 1:
                        unavailable.append({**ctx, "reason": "undefined validation calibration or nonfinite calibrated spread",
                                            "factors": factors})
                    variant = "ensemble" if len(seeds) > 1 else "single_member"
                    recommendation.extend(_front_eval(ctx, model, variant, means, calibrated if usable else None,
                                                        y[train], y[test], ids, true_ids, levels if usable else (0.0,),
                                                        fronts, preferences))
                    if len(seeds) > 1:
                        recommendation.extend(_front_eval(ctx, model, "first_seed_single_member", test_members[0], None,
                                                            y[train], y[test], ids, true_ids, (0.0,), fronts, preferences))
                ref = {**ctx, "model": "exhaustive_measured_reference", "variant": "oracle_reference", "kappa": None}
                recommendation.append({**ref, **_front_metrics(y[test], true_positions), "preference_regret": 0.0})
                _jsonl_row(fronts, {**ref, "selected_schema_ids": true_ids, "measured_front_schema_ids": true_ids})
                for name, supplied in comparators.items():
                    if system not in supplied["systems"]:
                        continue
                    front_ids = supplied["systems"][system][r - 1]
                    selected = [id_to_position[schema_id] for schema_id in front_ids]
                    values = _front_metrics(y[test], selected)
                    comparator = {**ctx, "model": name, "variant": "supplied", "kappa": None}
                    _jsonl_row(fronts, {**comparator, "selected_schema_ids": front_ids, "measured_front_schema_ids": true_ids})
                    if name in FRAMEWORK_COMPARATORS:
                        values.update(precision=None, recall=None, f1=None)
                        decisions = selected * 66
                    elif system in supplied["preferences"]:
                        decisions = [id_to_position[schema_id] for schema_id in supplied["preferences"][system][r - 1]]
                    else:
                        decisions = None
                    if decisions is not None:
                        regret, _ = _realized_regrets(y[test], decisions)
                        values["preference_regret"] = float(regret.mean())
                        for w, j, regret_w in zip(preference_weights(), decisions, regret):
                            _jsonl_row(preferences, {**comparator, "weights": w, "selected_schema_id": ids[j], "realized_regret": regret_w})
                    recommendation.append({**comparator, **values})
    data.to_csv(output / "aggregated_measurements.csv", index=False)
    _reports(output, "prediction_metrics", predictions, ("model", "objective"), shared)
    recommendation_systems = _reports(output, "recommendation_metrics", recommendation, ("model", "variant", "kappa"), shared)
    _reports(output, "uncertainty_metrics", uncertainty, ("model", "objective"), shared)
    if members:
        pd.DataFrame(members).to_csv(output / "prediction_metrics_per_seed.csv", index=False)
    paired = _paired_family(recommendation_systems, shared, scope)
    write_json(output / "folds.json", folds)
    write_json(output / "training_details.json", fits)
    write_json(output / "paired_tests.json", paired)
    summary = {
        "scope": scope,
        "claim": "Evaluation of the declared measured dataset; no claim of historical result reproduction.",
        **scenario, "label_source": data.label_source.iloc[0],
        "dataset": config.get("dataset"),
        "protocol_provenance": config.get("protocol_provenance"),
        "input_sha256": {"candidates": sha256_file(candidates_path), "measurements": sha256_file(measurements_path)},
        "system_order": order, "shared_systems": shared, "repetition_seeds": repeats, "views": views,
        "neural_model": model if run_neural else None,
        "prediction_baselines": baselines, "structural_features": feature_names,
        "structural_feature_source": descriptor_source if baselines else None,
        "risk_levels_requested": levels, "default_kappa": 1.0,
        "evaluated_candidates": int(sum(len(pos) for pos in group_positions.values())),
        "unmeasured_candidates": data.attrs["unmeasured_candidate_count"],
        "measurement_rows": measurement_rows,
        "measurement_aggregation": "per-candidate per-objective median across actual measurement repetitions" if measurement_rows == "raw_repetitions" else "supplied per-candidate protocol medians",
        "baseline_tuning": "minimum validation standardized log MSE, separately per objective; first grid entry breaks ties",
        "baseline_random_seed": "first neural member seed in each repetition",
        "prediction_normalization": "candidate predicted means range; zero span uses training measured range, then positive training magnitude",
        "preference_regret": "predicted top one vs feasible measured optimum, both scored on measured pool min-max scales across 66 simplex weights; measured constant objective contributes zero",
        "preference_regret_on_selection_scale": "same decision and measured optimum, both scored on prediction-based scales as a sensitivity check",
        "tie_break": "first schema ID in ascending string order for exactly equal computed scores",
        "uncertainty": "sample standard deviation in original units; validation-only Gaussian scale; zero validation spread yields unavailable calibration, zero residuals yield zero scale",
        "unavailable_risk_folds": unavailable,
        "coverage": "fraction within central Gaussian interval; descriptive system-weighted means",
        "supplied_comparator_provenance": {name: info["provenance"] for name, info in comparators.items()},
        "paired_tests_status": paired["status"],
        "elapsed_seconds": time.perf_counter() - start,
        "software": environment(),
    }
    write_json(output / "summary.json", summary)
    return json_ready(summary)
