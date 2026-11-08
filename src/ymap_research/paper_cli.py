"""Command-line entry points for the manuscript workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .cli import load_config
from .common import TARGETS, json_ready, read_jsonl, sha256_file, write_json, write_jsonl
from .paper_data import DATASET_NAMES, resolve_dataset


def _new_output(path):
    out = Path(path)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _models(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob("*.json"))
    if not paths:
        raise ValueError("Provide an object-model JSON file or a directory of JSON models")
    models = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    names = [model.get("name") if isinstance(model, dict) else None for _, model in models]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Object models must have unique, nonempty names")
    return models


def _generate(args):
    from .paper_schema import paper_candidates

    models = _models(args.models)
    scopes = load_config(args.scopes)
    if set(scopes) != {model["name"] for _, model in models}:
        raise ValueError("The scope file must map every input system name to its maximum table count")
    out = _new_output(args.out)
    ddl_dir = out / "ddl"
    ddl_dir.mkdir()
    audit = {}
    model_sources = {}

    def records():
        for path, model in models:
            name = model["name"]
            model_sources[name] = {"source": path.name, "sha256": sha256_file(path), "max_tables": scopes[name]}
            audit[name] = {}
            for row in paper_candidates(model, max_tables=scopes[name], backend=args.backend,
                                        alloy_jar=args.alloy_jar, artifact_dir=out / "solver" / path.stem,
                                        counts=audit[name]):
                (ddl_dir / f"{row['schema_id']}.sql").write_bytes(row["paper_ddl"].encode("utf-8"))
                yield row

    partial = out / "candidates.jsonl.partial"
    write_jsonl(partial, records())
    if not all(counts["unique"] > 0 for counts in audit.values()):
        raise ValueError("Every input system must produce at least one candidate within its table scope")
    path = out / "candidates.jsonl"
    partial.replace(path)
    result = {"status": "generated", "backend": args.backend, "models": model_sources, "counts": audit,
              "candidates": path.name, "candidate_sha256": sha256_file(path),
              "scope_note": "Bounded finite mapping decisions; unique means byte-identical canonical DDL deduplication."}
    write_json(out / "generation_manifest.json", result)
    return result


def _dataset(args):
    from .evaluation import load_dataset

    dataset = resolve_dataset(args.project_root, args.dataset, dataset_root=args.dataset_root, manifest_path=args.manifest)
    records, data, _ = load_dataset(dataset.candidates_path, dataset.measurements_path, dataset.evaluation_config())
    return dataset, records, data, dataset.check_alignment(data)


def _dataset_config(args, dataset):
    config = load_config(args.config)
    config.update(dataset.evaluation_config())
    config["dataset"] = dataset.name
    config["protocol_provenance"] = dataset.provenance
    return config


def _dataset_metadata(dataset, counts):
    return {**dataset.metadata(), "observed_counts": counts, "input_sha256": dataset.input_hashes()}


def _check(args):
    dataset, _, _, counts = _dataset(args)
    result = _dataset_metadata(dataset, counts)
    if args.out:
        write_json(args.out, result)
    # The full provenance (including attribution) remains in the manifest and output file.
    return {"dataset": dataset.name, "scope": dataset.scope, "counts": counts,
            "units_confirmed": dataset.units_confirmed, "manifest": str(dataset.manifest_path)}


def _evaluate(args):
    dataset, _, _, counts = _dataset(args)
    config = _dataset_config(args, dataset)
    scope = config.get("scope", "paper" if dataset.name == "paper-reference" else "archive-check")
    if scope == "archive-check":
        from .evaluation import evaluate

        if dataset.name == "paper-reference":
            raise ValueError("Set scope to paper, ablation, or smoke for paper-reference")
        out = _new_output(args.out)
        report = evaluate(dataset.candidates_path, dataset.measurements_path, out, config)
    else:
        from .paper_evaluation import evaluate_paper

        native_smoke = dataset.name == "native" and scope == "smoke" and config.get("views") == []
        if dataset.name != "paper-reference" and not native_smoke:
            raise ValueError("Manuscript-method evaluation requires a verified paper-reference dataset. Native data supports scope: smoke with views: []; use archive-check for released structural evaluations.")
        if native_smoke:
            config["measurement_rows"] = "protocol_medians"
        config["scope"] = scope
        out = _new_output(args.out)
        report = evaluate_paper(dataset.candidates_path, dataset.measurements_path, out, config)
    write_json(out / "dataset_manifest.json", _dataset_metadata(dataset, counts))
    return {"status": "evaluated", "dataset": dataset.name, "scope": scope, "output": str(out), "report": report}


def _fit(args):
    from .paper_evaluation import _load_paper_data, _row_requirements, _validate_canonical_pool, gaussian_multiplier
    from .paper_model import fit_paper_ensemble

    if args.dataset != "paper-reference":
        raise ValueError("The manuscript model requires a verified paper-reference dataset")
    dataset = resolve_dataset(args.project_root, args.dataset, dataset_root=args.dataset_root, manifest_path=args.manifest)
    config = _dataset_config(args, dataset)
    records, data, _ = _load_paper_data(dataset.candidates_path, dataset.measurements_path, config)
    counts = dataset.check_alignment(data)
    _row_requirements(data, config, "paper")
    _validate_canonical_pool(records)
    systems = set(data["object_model"])
    holdout = set(args.holdout_system)
    if args.validation_system not in systems or not holdout <= systems or args.validation_system in holdout:
        raise ValueError("Specify a measured validation system and distinct known holdout systems")
    train = data[~data["object_model"].isin(holdout | {args.validation_system})]
    valid = data[data["object_model"] == args.validation_system]
    if train.empty:
        raise ValueError("Fit needs at least one training system disjoint from validation and holdout systems")
    seeds = config.get("deployment_seeds", [0, 1, 2, 3, 4])
    out = _new_output(args.out)
    by_id = {r["schema_id"]: r for r in records} if isinstance(records, list) else records
    y_train, y_val = train[list(TARGETS)].to_numpy(), valid[list(TARGETS)].to_numpy()
    val_records = [by_id[sid] for sid in valid["schema_id"]]
    ensemble = fit_paper_ensemble([by_id[sid] for sid in train["schema_id"]], y_train, val_records, y_val, config, seeds)
    members = ensemble.predict_members(val_records)
    factors = gaussian_multiplier(y_val, members.mean(axis=0), members.std(axis=0, ddof=1))
    ensemble.save(out / "ensemble")
    result = {"status": "fitted", "validation_system": args.validation_system,
              "train_systems": sorted(set(train["object_model"])), "holdout_systems": sorted(holdout),
              "training_objective_extrema": [y_train.min(axis=0), y_train.max(axis=0)],
              "calibration_factors": factors, "dataset": _dataset_metadata(dataset, counts)}
    write_json(out / "fit_manifest.json", result)
    return result


def _recommend(args):
    from .paper_model import PaperEnsemble
    from .paper_schema import paper_candidates
    from .paper_search import search_candidates

    if not Path(args.model).is_file():
        raise ValueError("Recommendation requires a single object-model JSON file")
    path, model = _models(args.model)[0]
    root = Path(args.model_dir)
    fitted = json.loads((root / "fit_manifest.json").read_text(encoding="utf-8"))
    overrides = load_config(args.config).get("neural", {}) if args.config else {}
    ensemble = PaperEnsemble.load(root / "ensemble", **{key: overrides[key] for key in ("device", "checkpoints", "cache_dir") if key in overrides})
    out = _new_output(args.out)
    seeds = list(range(4))
    streams = [
        paper_candidates(model, max_tables=args.max_tables, backend=args.backend, alloy_jar=args.alloy_jar,
                         artifact_dir=out / "solver" / str(seed), seed=seed, deduplicate=False)
        for seed in seeds
    ]
    result = search_candidates(streams, ensemble.predict_members, fitted["training_objective_extrema"],
                               fitted["calibration_factors"], batch_size=args.batch_size,
                               patience=None if args.exhaustive else args.patience,
                               budget=None if args.exhaustive else args.budget, kappa=args.kappa, weights=args.weights)
    records = result.pop("candidates")
    means, stds = result.pop("means"), result.pop("sample_std")
    for kind in ("neutral_front", "risk_front", "selected"):
        result[f"{kind}_schema_ids"] = [records[i]["schema_id"] for i in result.pop(f"{kind}_indices")]
    write_jsonl(out / "visited_predictions.jsonl",
                ({"schema_id": row["schema_id"], "mean": mean.tolist(), "sample_std": std.tolist()}
                 for row, mean, std in zip(records, means, stds)))
    selected = set(result["selected_schema_ids"])
    for row in records:
        if row["schema_id"] in selected:
            ddl = out / "selected_ddl" / f"{row['schema_id']}.sql"
            ddl.parent.mkdir(exist_ok=True)
            ddl.write_bytes(row["paper_ddl"].encode("utf-8"))
    result.update({"object_model_sha256": sha256_file(path), "model_manifest_sha256": sha256_file(root / "fit_manifest.json"),
                   "object_model": model["name"], "backend": args.backend,
                   "choice_order_seeds": seeds, "max_tables": args.max_tables,
                   "exhaustive": args.exhaustive, "batch_size": args.batch_size,
                   "patience": None if args.exhaustive else args.patience,
                   "budget": None if args.exhaustive else args.budget,
                   "kappa": args.kappa, "weights": args.weights})
    write_json(out / "recommendation.json", result)
    return result


def _inputs(parser, default="native"):
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=default)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest", help="Verified dataset manifest for paper-reference")


def _solver(parser):
    parser.add_argument("--backend", choices=("alloy", "python"), default="alloy")
    parser.add_argument("--alloy-jar")
    parser.add_argument("--out", required=True)


def main(argv):
    parser = argparse.ArgumentParser(prog="ymap paper", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Verify a single dataset scenario and its declared counts")
    _inputs(check)
    check.add_argument("--out", help="Write the full dataset provenance and input hashes as JSON")

    gen = sub.add_parser("generate", help="Enumerate and deduplicate canonical DDL")
    gen.add_argument("--models", required=True)
    gen.add_argument("--scopes", required=True, help="YAML system name to maximum table count")
    _solver(gen)

    profile = sub.add_parser("profile", help="Measure generated candidates on a disposable PostgreSQL server")
    profile.add_argument("--candidates", required=True)
    profile.add_argument("--config", required=True)
    profile.add_argument("--out", required=True)

    evaluate = sub.add_parser("evaluate", help="Run system-held-out evaluation for one scenario")
    _inputs(evaluate)
    evaluate.add_argument("--config")
    evaluate.add_argument("--out", required=True)

    fit = sub.add_parser("fit", help="Train and calibrate a deployable manuscript ensemble")
    _inputs(fit, "paper-reference")
    fit.add_argument("--validation-system", required=True)
    fit.add_argument("--holdout-system", action="append", default=[])
    fit.add_argument("--config", required=True)
    fit.add_argument("--out", required=True)

    rec = sub.add_parser("recommend", help="Run bounded incremental search for a new object model")
    rec.add_argument("--model", required=True)
    rec.add_argument("--model-dir", required=True, help="Output directory from paper fit")
    rec.add_argument("--config", help="Optional neural checkpoint paths and device override")
    rec.add_argument("--max-tables", type=int, required=True)
    rec.add_argument("--batch-size", type=int, default=100)
    rec.add_argument("--patience", type=int, default=5)
    rec.add_argument("--budget", type=int, default=4000)
    rec.add_argument("--exhaustive", action="store_true")
    rec.add_argument("--kappa", type=float, default=1.0)
    rec.add_argument("--weights", nargs=3, type=float)
    _solver(rec)

    transfer = sub.add_parser("transfer", help="Compare measured workload variants and out-of-fold predictions")
    transfer.add_argument("--reference", required=True)
    transfer.add_argument("--variant", required=True)
    transfer.add_argument("--predictions", required=True)
    transfer.add_argument("--model")
    transfer.add_argument("--allow-reference-superset", action="store_true")
    transfer.add_argument("--out", required=True)
    orm = sub.add_parser("compare-orm", help="Summarize paired SQL and ORM measurements")
    orm.add_argument("--sql", required=True)
    orm.add_argument("--hibernate", required=True)
    orm.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    try:
        handlers = {"check": _check, "generate": _generate, "evaluate": _evaluate, "fit": _fit, "recommend": _recommend}
        if args.command in handlers:
            result = handlers[args.command](args)
        elif args.command == "profile":
            from .paper_profile import profile_paper

            out = _new_output(args.out)
            result = {"measurements": str(profile_paper(read_jsonl(args.candidates), load_config(args.config), out))}
        else:
            from .paper_diagnostics import orm_comparison, protocol_transfer

            if Path(args.out).exists():
                raise ValueError("Use a new output path to preserve the previous result")
            if args.command == "transfer":
                result = protocol_transfer(args.reference, args.variant, args.predictions, model=args.model,
                                           allow_reference_superset=args.allow_reference_superset)
            else:
                result = orm_comparison(args.sql, args.hibernate)
            write_json(args.out, result)
        print(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Y-Map paper: {exc}", file=sys.stderr)
        return 2
