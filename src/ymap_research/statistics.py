"""System-level paired inference; repeated seeds are never independent systems."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def paired_wilcoxon(x, y, zero_tolerance: float = 0.0) -> dict:
    """Exact two-sided conditional signed-rank test, including ties.

    Zero differences are removed (Wilcox convention). Tied absolute differences
    receive average ranks. A dynamic-programming sign-permutation distribution
    gives the exact conditional null distribution, avoiding SciPy's asymptotic
    fallback for ties. Rank-biserial effect is positive when x > y. Scientific
    interpretation still assumes independent, exchangeable paired systems and
    symmetric differences; a small p-value cannot repair dependent observations.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired arrays must be equal-length, nonempty, finite vectors")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    with np.errstate(over="ignore"):
        differences = a - b
    if not np.isfinite(differences).all():
        raise ValueError("Paired differences overflow; rescale observations before testing")
    active = np.abs(differences) > zero_tolerance
    nonzero = differences[active]
    n = len(nonzero)
    if not n:
        return {"n_pairs": len(a), "n_nonzero": 0, "n_zero": len(a), "has_ties": False, "statistic": 0.0, "p_value": 1.0, "rank_biserial": 0.0, "median_difference": 0.0, "method": "exact_two_sided_sign_permutation_signed_ranks", "zero_method": "wilcox"}
    ranks = rankdata(np.abs(nonzero), method="average")
    integer_ranks = np.rint(2 * ranks).astype(int)
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    observed = int(min(integer_ranks[nonzero > 0].sum(), integer_ranks[nonzero < 0].sum()))
    total = int(integer_ranks.sum())
    distribution: Counter = Counter({0: 1})
    for rank in integer_ranks:
        following = distribution.copy()
        for value, count in distribution.items():
            following[value + int(rank)] += count
        distribution = following
    extreme = sum(count for value, count in distribution.items() if min(value, total - value) <= observed)
    p_value = min(1.0, extreme / (2 ** n))
    return {"n_pairs": len(a), "n_nonzero": n, "n_zero": int((~active).sum()), "has_ties": len(np.unique(np.abs(nonzero))) != n, "statistic": min(positive, negative), "p_value": p_value, "rank_biserial": (positive - negative) / (positive + negative), "median_difference": float(np.median(differences)), "method": "exact_two_sided_sign_permutation_signed_ranks", "zero_method": "wilcox"}


def holm_adjust(p_values) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(values < 0) or np.any(values > 1) or not np.isfinite(values).all():
        raise ValueError("p-values must be finite and within [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, float(values[index]) * (len(values) - position))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compare_system_metrics(per_system_df, reference_model: str, metrics=None) -> list[dict]:
    """Compare reference with each method using paired object-model outcomes.

    Holm correction is ONE family over all requested metrics and comparators.
    Inputs require one row per method/system; seeds or candidate schemas must be
    aggregated before calling. Missing method-system pairs use available matched
    systems and report the system IDs. Do not compare different scenarios here.
    """
    frame = pd.DataFrame(per_system_df).copy()
    metrics = ["f1", "gd", "hypervolume", "igd_plus", "preference_regret"] if metrics is None else list(metrics)
    if len(metrics) != len(set(metrics)):
        raise ValueError("Statistical metric names must be unique within the Holm family")
    required = {"model", "object_model", *metrics}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing per-system columns: {sorted(required - set(frame.columns))}")
    for key in ("model", "object_model"):
        if frame[key].isna().any() or not frame[key].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{key} must contain nonempty string IDs")
    for scenario_key in ("workload_id", "environment_id"):
        if scenario_key in frame and (frame[scenario_key].isna().any() or frame[scenario_key].nunique() != 1):
            raise ValueError("Paired statistics require exactly one workload/environment scenario")
    for metric in metrics:
        try:
            frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric {metric} must contain numeric values or null") from exc
    if frame.duplicated(["model", "object_model"]).any():
        raise ValueError("Statistics require one row per method and independent object model")
    if reference_model not in set(frame["model"]):
        raise ValueError("Reference model is absent")
    reference = frame[frame.model == reference_model].set_index("object_model")
    results = []
    for model in sorted(set(frame.model) - {reference_model}):
        other = frame[frame.model == model].set_index("object_model")
        common = sorted(set(reference.index) & set(other.index))
        for metric in metrics:
            valid = [group for group in common if pd.notna(reference.loc[group, metric]) and pd.notna(other.loc[group, metric]) and np.isfinite(reference.loc[group, metric]) and np.isfinite(other.loc[group, metric])]
            if not valid:
                continue
            stats = paired_wilcoxon(reference.loc[valid, metric].to_numpy(dtype=float), other.loc[valid, metric].to_numpy(dtype=float))
            direction = -1 if metric in {"gd", "igd_plus", "preference_regret", "mae", "rmse", "nrmse"} or metric.endswith(("_mae", "_rmse", "_nrmse")) else 1
            stats.update({"reference": reference_model, "comparator": model, "metric": metric, "systems": valid, "rank_biserial_favoring_reference": direction * stats["rank_biserial"], "favorable_direction": "lower" if direction < 0 else "higher"})
            results.append(stats)
    adjusted = holm_adjust([record["p_value"] for record in results])
    for record, p_value in zip(results, adjusted):
        record["holm_p_value"] = p_value
        record["holm_family_size"] = len(results)
        record["unit_of_analysis"] = "object_model"
    return results
