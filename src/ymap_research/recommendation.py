"""Finite-pool recommendation from predictions, and measured-coordinate assessment.

All objectives are minimized. `recommend` never receives measured test objectives.
Selection normalization MUST be learned from training objectives. Evaluation metrics
use a separate fixed normalization of the entire measured test candidate pool.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _matrix(values, name: str = "objectives") -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim == 1 and out.size == 0:
        out = out.reshape(0, 3)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def pareto_mask(values) -> np.ndarray:
    """Non-dominated mask for minimization; identical points all remain eligible."""
    points = _matrix(values)
    mask = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        mask[i] = not np.any(np.all(points <= point, axis=1) & np.any(points < point, axis=1))
    return mask


def fit_normalization(training_objectives) -> dict[str, list]:
    values = _matrix(training_objectives, "training objectives")
    if not len(values):
        raise ValueError("Cannot fit normalization on an empty training set")
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    return {"minimum": minimum.tolist(), "scale": np.where(span > 0, span, 1.0).tolist(), "constant": (span == 0).tolist(), "source": "training_objectives"}


def normalize_predictions(means, std, normalization: dict) -> tuple[np.ndarray, np.ndarray]:
    mu = _matrix(means, "prediction means")
    sigma = np.zeros_like(mu) if std is None else _matrix(std, "prediction standard deviations")
    if sigma.shape != mu.shape or np.any(sigma < 0):
        raise ValueError("Standard deviations must be nonnegative and match means")
    minimum = np.asarray(normalization["minimum"], dtype=float)
    scale = np.asarray(normalization["scale"], dtype=float)
    if minimum.shape != (3,) or scale.shape != (3,) or not np.isfinite(minimum).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Normalization needs three finite minima and positive scales")
    # Do not clip extrapolations: held-out predictions can lie outside train range.
    return (mu - minimum) / scale, sigma / scale


def recommend(predictions_mean, predictions_std=None, normalization=None, weights=None, risk_lambda: float = 0.0, mode: str = "pareto", budget: int | None = None, seed: int = 0, beam_width: int = 8, max_iterations: int = 20, proposals_per_beam: int = 4, embeddings=None, exploration_eta: float = 0.0) -> dict[str, Any]:
    """Select indices using dimensionless mean + lambda * std.

    Modes: 'pareto' returns all non-dominated risk coordinates; 'preference'
    returns the single lowest weighted score; 'beam' evaluates a finite pool in
    seeded batches and returns its visited non-dominated subset. Beam search is
    only a final-stage search over already generated/encoded/predicted candidates.
    Its budget counts inspected candidate scores, not synthesis or inference.
    Optional embeddings define latent nearest-neighbor proposals. The beam retains
    highest acquisition = -weighted(mean + lambda*std) + eta*weighted(std),
    with every term normalized using the same training objective scales.
    """
    if normalization is None:
        raise ValueError("Pass normalization fitted on training objectives")
    if not np.isfinite(exploration_eta) or exploration_eta < 0:
        raise ValueError("exploration_eta must be finite and nonnegative")
    if not np.isfinite(risk_lambda) or risk_lambda < 0:
        raise ValueError("risk_lambda must be finite and nonnegative")
    mu, sigma = normalize_predictions(predictions_mean, predictions_std, normalization)
    n = len(mu)
    w = np.full(3, 1 / 3) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0) or w.max() <= 0:
        raise ValueError("weights must be three finite nonnegative values with positive sum")
    w = w / w.max()
    w = w / w.sum()
    risk = mu + risk_lambda * sigma
    scalar = risk @ w
    acquisition = -scalar + exploration_eta * (sigma @ w)
    coordinates = risk
    if embeddings is not None:
        coordinates = np.asarray(embeddings, dtype=float)
        if coordinates.ndim != 2 or len(coordinates) != n or not np.isfinite(coordinates).all():
            raise ValueError("embeddings must be a finite n-by-d matrix aligned with predictions")
    trace: list[dict] = []
    if mode == "pareto":
        visited = np.arange(n)
        indices = visited[pareto_mask(risk)]
    elif mode == "preference":
        visited = np.arange(n)
        indices = np.array([int(np.argmin(scalar))], dtype=int) if n else np.array([], dtype=int)
    elif mode == "beam":
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0 for value in (beam_width, max_iterations, proposals_per_beam)):
            raise ValueError("Beam search settings must be positive integers")
        if budget is not None and (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer))):
            raise ValueError("budget must be an integer or None")
        limit = min(n, n if budget is None else budget)
        if limit < 0:
            raise ValueError("budget must be nonnegative")
        rng = np.random.default_rng(seed)
        unvisited = set(range(n))
        visited_list: list[int] = []
        beam: list[int] = []
        # Optional latent neighborhood or normalized predicted-objective fallback.
        # All coordinates are precomputed; this does not bypass full-pool inference.
        for iteration in range(max_iterations):
            remaining = limit - len(visited_list)
            if remaining <= 0 or not unvisited:
                break
            if not beam:
                proposals = rng.choice(sorted(unvisited), size=min(beam_width, remaining, len(unvisited)), replace=False).tolist()
            else:
                proposals = []
                for parent in beam:
                    available = sorted(unvisited.difference(proposals))
                    if not available:
                        break
                    distances = np.linalg.norm(coordinates[available] - coordinates[parent], axis=1)
                    for pos in np.argsort(distances, kind="stable")[:proposals_per_beam]:
                        proposals.append(available[int(pos)])
                # One seeded exploration proposal avoids repeatedly inspecting only local neighbors.
                available = sorted(unvisited.difference(proposals))
                if available:
                    proposals.append(int(rng.choice(available)))
                proposals = proposals[:remaining]
            for index in proposals:
                unvisited.remove(index)
                visited_list.append(index)
            beam = sorted(visited_list, key=lambda idx: (-acquisition[idx], idx))[:beam_width]
            current = np.asarray(visited_list, dtype=int)
            front = current[pareto_mask(risk[current])]
            trace.append({"iteration": iteration, "new_indices": proposals, "evaluated_candidates": len(visited_list), "front_indices": front.tolist(), "best_predicted_preference": float(scalar[beam[0]]) if beam else None, "best_acquisition": float(acquisition[beam[0]]) if beam else None})
        visited = np.asarray(visited_list, dtype=int)
        indices = visited[pareto_mask(risk[visited])] if len(visited) else visited
    else:
        raise ValueError("mode must be 'pareto', 'preference', or 'beam'")
    return {"indices": indices.tolist(), "visited_indices": visited.tolist(), "risk_objectives": risk.tolist(), "preference_scores": scalar.tolist(), "weights": w.tolist(), "risk_lambda": float(risk_lambda), "exploration_eta": float(exploration_eta), "acquisition_scores": acquisition.tolist(), "neighborhood": "provided_embeddings" if embeddings is not None else "normalized_predicted_objectives", "mode": mode, "trace": trace, "provenance": "final_stage_finite_candidate_pool; all candidates already generated, encoded and predicted; no measured test objectives used for selection"}


def _union_area_2d(points: np.ndarray, reference: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    ys = sorted(set(points[:, 0].tolist() + [float(reference[0])]))
    area = 0.0
    for left, right in zip(ys[:-1], ys[1:]):
        eligible = points[points[:, 0] <= left]
        if len(eligible):
            area += (right - left) * max(0.0, reference[1] - float(eligible[:, 1].min()))
    return float(area)


def hypervolume_3d(front, reference=(1.1, 1.1, 1.1)) -> float:
    """Exact union of minimization boxes [point, reference], no Monte Carlo."""
    points = _matrix(front)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (3,) or not np.isfinite(ref).all():
        raise ValueError("reference must contain three finite coordinates")
    points = points[np.all(points < ref, axis=1)]
    if not len(points):
        return 0.0
    points = np.unique(points[pareto_mask(points)], axis=0)
    xs = sorted(set(points[:, 0].tolist() + [float(ref[0])]))
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        eligible = points[points[:, 0] <= left]
        volume += (right - left) * _union_area_2d(eligible[:, 1:], ref[1:])
    return float(volume)


def recommendation_metrics(true_objectives, selected_indices, weights=None, reference_point=(1.1, 1.1, 1.1)) -> dict[str, Any]:
    """Assess selected IDs at measured coordinates; test labels are metric-only.

    GD = mean Euclidean distance from measured recommended points to true front.
    IGD+ = mean_r min_a ||max(a-r,0)|| over true-front points r.
    Precision/recall compare IDs (duplicates with different IDs remain distinct).
    Normalization uses the whole measured candidate pool, fixed for every method.
    """
    truth = _matrix(true_objectives)
    if not len(truth):
        raise ValueError("Measured candidate pool is empty")
    raw_selected = np.asarray(selected_indices)
    if raw_selected.ndim != 1 or (raw_selected.size and (not np.issubdtype(raw_selected.dtype, np.integer) or np.issubdtype(raw_selected.dtype, np.bool_))):
        raise ValueError("Selected indices must be integer candidate positions")
    selected = raw_selected.astype(int)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= len(truth)):
        raise ValueError("Selected indices must be unique valid candidate positions")
    true_mask = pareto_mask(truth)
    selected_mask = np.zeros(len(truth), dtype=bool)
    selected_mask[selected] = True
    overlap = int(np.sum(true_mask & selected_mask))
    precision = overlap / len(selected) if len(selected) else 0.0
    recall = overlap / int(true_mask.sum())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    minimum = truth.min(axis=0)
    span = np.ptp(truth, axis=0)
    normalized = (truth - minimum) / np.where(span > 0, span, 1.0)
    reference_front = normalized[true_mask]
    selected_points = normalized[selected]
    w = np.full(3, 1 / 3) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (3,) or np.any(w < 0) or not np.isfinite(w).all() or w.max() <= 0:
        raise ValueError("Invalid preference weights")
    w = w / w.max()
    w = w / w.sum()
    if len(selected):
        distances = np.linalg.norm(selected_points[:, None, :] - reference_front[None, :, :], axis=2)
        gd = float(distances.min(axis=1).mean())
        igd_plus = float(np.linalg.norm(np.maximum(selected_points[None, :, :] - reference_front[:, None, :], 0), axis=2).min(axis=1).mean())
        regret = float((selected_points @ w).min() - (normalized @ w).min())
    else:
        gd = igd_plus = regret = None
    return {"precision": precision, "recall": recall, "f1": f1, "gd": gd, "igd_plus": igd_plus, "hypervolume": hypervolume_3d(selected_points, reference_point), "true_hypervolume": hypervolume_3d(reference_front, reference_point), "preference_regret": regret, "true_pareto_count": int(true_mask.sum()), "selected_count": int(len(selected)), "pool_size": len(truth), "metric_normalization": {"minimum": minimum.tolist(), "scale": np.where(span > 0, span, 1.0).tolist(), "source": "full_measured_test_pool_for_metrics_only"}, "hypervolume_reference": list(reference_point)}
