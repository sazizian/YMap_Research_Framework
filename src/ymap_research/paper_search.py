"""Stage D: interleave synthesis, deduplicate, predict and stop on a stable front."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
import hashlib
import time

import numpy as np

from .recommendation import normalize_predictions, pareto_mask


def interleave(streams: Sequence[Iterable[dict]]) -> Iterator[dict]:
    """Round-robin live streams and close every solver on early termination."""
    active = deque(iter(stream) for stream in streams)
    all_iterators = list(active)
    try:
        while active:
            stream = active.popleft()
            try:
                candidate = next(stream)
            except StopIteration:
                continue
            active.append(stream)
            yield candidate
    finally:
        for stream in all_iterators:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def _positive_int(value: int | None, name: str, *, unlimited: bool = False) -> None:
    if value is None and unlimited:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer" + (" or None" if unlimited else ""))


def search_candidates(
    streams: Sequence[Iterable[dict]],
    predict_members: Callable[[list[dict]], np.ndarray],
    training_objectives,
    calibration_factors,
    *,
    batch_size: int = 100,
    patience: int | None = 5,
    budget: int | None = 4000,
    kappa: float = 1.0,
    weights=None,
) -> dict:
    """Algorithm 1, with a unique-schema budget and no measured target inputs.

    Predictions are produced only for unique candidates already enumerated. The
    stopping rule uses the risk-neutral front; final risk scoring uses all
    visited candidates, since a mean-dominated schema may have lower uncertainty.
    Passing None for patience and budget exhausts the supplied streams.
    """
    from .paper_evaluation import candidate_normalization

    _positive_int(batch_size, "batch_size")
    _positive_int(patience, "patience", unlimited=True)
    _positive_int(budget, "budget", unlimited=True)
    if not streams:
        raise ValueError("At least one candidate stream is required")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    factors = np.asarray(calibration_factors, dtype=float)
    if factors.shape != (3,) or (kappa and (not np.isfinite(factors).all() or np.any(factors < 0))):
        raise ValueError("Risk-aware search needs three nonnegative, finite validation calibration factors")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError("weights must be three finite nonnegative values that sum to one")

    source = interleave(streams)
    records: list[dict] = []
    seen: set[str] = set()
    identifiers: set[str] = set()
    means = np.empty((0, 3), dtype=float)
    stds = np.empty((0, 3), dtype=float)
    neutral: list[int] = []
    trace = []
    stable = raw_total = 0
    member_count = None
    stop = "streams_exhausted"
    timing = {"enumeration_and_dedup_seconds": 0.0, "encoding_and_prediction_seconds": 0.0, "ranking_seconds": 0.0}
    started = time.perf_counter()
    try:
        while True:
            batch = []
            raw = 0
            exhausted = False
            tick = time.perf_counter()
            while raw < batch_size and (budget is None or len(records) + len(batch) < budget):
                try:
                    record = next(source)
                except StopIteration:
                    exhausted = True
                    break
                raw += 1
                ddl = record.get("paper_ddl")
                sid = record.get("schema_id")
                if not isinstance(ddl, str) or not ddl or not isinstance(sid, str) or not sid:
                    raise ValueError("Search candidates require canonical paper_ddl and schema_id")
                fingerprint = hashlib.sha256(ddl.encode("utf-8")).hexdigest()
                if fingerprint in seen:
                    continue
                if sid in identifiers:
                    raise ValueError(f"Candidate ID refers to different canonical DDL: {sid}")
                identifiers.add(sid)
                seen.add(fingerprint)
                batch.append(record)
            timing["enumeration_and_dedup_seconds"] += time.perf_counter() - tick
            raw_total += raw
            if batch:
                tick = time.perf_counter()
                members = np.asarray(predict_members(batch), dtype=float)
                if members.ndim != 3 or members.shape[1:] != (len(batch), 3) or len(members) < 2 or not np.isfinite(members).all() or np.any(members <= 0):
                    raise ValueError("Predictor must return at least two aligned members with finite positive original-unit predictions")
                if member_count is not None and len(members) != member_count:
                    raise ValueError("Ensemble size changed across batches")
                member_count = len(members)
                means = np.vstack((means, members.mean(axis=0)))
                stds = np.vstack((stds, members.std(axis=0, ddof=1)))
                timing["encoding_and_prediction_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                contenders = np.asarray([*neutral, *range(len(records), len(records) + len(batch))], dtype=int)
                next_front = contenders[pareto_mask(means[contenders])].tolist()
                stable = stable + 1 if next_front == neutral else 0
                neutral = next_front
                records.extend(batch)
                timing["ranking_seconds"] += time.perf_counter() - tick
            if raw:
                trace.append({
                    "batch": len(trace) + 1, "raw_candidates": raw, "new_unique": len(batch),
                    "unique_seen": len(records), "stable_batches": stable, "neutral_front_size": len(neutral),
                    "neutral_front_schema_ids": [records[i]["schema_id"] for i in neutral],
                })
            if budget is not None and len(records) >= budget:
                stop = "unique_budget"
                break
            if patience is not None and stable >= patience:
                stop = "neutral_front_stable"
                break
            if exhausted:
                break
    finally:
        source.close()

    result = {"status": "completed" if records else "no_admissible_candidate", "stop_reason": stop, "raw_candidates_seen": raw_total, "unique_candidates_seen": len(records), "member_count": member_count, "trace": trace, "candidates": records, "means": means, "sample_std": stds, "neutral_front_indices": neutral}
    tick = time.perf_counter()
    if records:
        # A neutral front never needs rescaling to break a constant objective.
        if not kappa and weights is None:
            chosen = front = neutral
            normalization = None
        else:
            normalization = candidate_normalization(means, training_objectives)
            mu, sigma = normalize_predictions(means, stds * factors if kappa else None, normalization)
            pessimistic = mu + kappa * sigma
            front = np.flatnonzero(pareto_mask(pessimistic)).tolist()
            if weights is None:
                chosen = front
            else:
                scores = pessimistic @ weights
                chosen = [min(range(len(records)), key=lambda i: (scores[i], records[i]["schema_id"]))]
        result.update({"risk_front_indices": front, "selected_indices": chosen, "normalization": normalization})
    else:
        result.update({"risk_front_indices": [], "selected_indices": [], "normalization": None})
    timing["ranking_seconds"] += time.perf_counter() - tick
    timing["wall_clock_seconds"] = time.perf_counter() - started
    result["timings"] = timing
    return result
