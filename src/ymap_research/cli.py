from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .common import TARGETS, environment, json_ready, read_jsonl, validate_dataset, write_json, write_jsonl


def load_config(path):
    if path is None:
        return {}
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return value


def generate(models_path, out_dir, maximum=None, backend="python", alloy_jar=None):
    from .synthesis import generate_candidates
    source = Path(models_path)
    paths = [source] if source.is_file() else sorted(source.glob("*.json"))
    if not paths:
        raise ValueError("No object-model JSON inputs")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "candidates.jsonl").exists():
        raise ValueError("Candidate output already exists; use a new output directory to preserve the previous run")
    counts, unique_counts, timings, seen = {}, {}, {}, set()
    started = time.perf_counter()
    partial = out / "candidates.jsonl.partial"
    with partial.open("w", encoding="utf-8") as handle:
        for path in paths:
            model = json.loads(path.read_text(encoding="utf-8"))
            if model["name"] in counts:
                raise ValueError("Source object-model names must be unique")
            count, fingerprints = 0, set()
            model_start = time.perf_counter()
            generated = generate_candidates(model, max_candidates=maximum, backend=backend, alloy_jar=alloy_jar, artifact_dir=out / "synthesis_artifacts" / path.stem)
            for row in generated:
                if row["schema_id"] in seen:
                    raise ValueError("Candidate IDs collide across input object models")
                seen.add(row["schema_id"])
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                ddl_path = out / "ddl" / f"{row['schema_id']}.sql"
                ddl_path.parent.mkdir(parents=True, exist_ok=True)
                ddl_path.write_text(row["ddl"], encoding="utf-8")
                fingerprints.add(hashlib.sha256(row["ddl"].encode()).hexdigest())
                count += 1
            counts[model["name"]] = count
            unique_counts[model["name"]] = len(fingerprints)
            timings[model["name"]] = time.perf_counter() - model_start
    partial.replace(out / "candidates.jsonl")
    result = {"candidate_count": len(seen), "per_system": counts, "backend": backend,
              "per_system_unique_ddl_count": unique_counts, "ddl_identity_note": "Exact emitted-DDL byte identity only; no SQL-semantic or graph-isomorphism equivalence claim; decision variants retained.",
              "per_system_generation_seconds": timings, "generation_workflow_seconds": time.perf_counter() - started,
              "timing_boundary": "Includes source reading, generation, structural checks, representations, serialization and DDL writing; excludes profiling/training/prediction/selection.",
              "max_candidates_per_system": maximum, "enumeration_note": "A cap is a deterministic prefix, not a representative random sample or exhaustive claim.",
              "formal_boundary": "Python structural validation is not Alloy solving or a semantic-preservation proof.", "environment": environment()}
    write_json(out / "generation_manifest.json", result)
    return result


def run_demo(out_dir):
    """Explicit software fixture: these invented targets are NOT research evidence."""
    from .evaluation import evaluate
    from .inference import train, predict_and_recommend
    out = Path(out_dir)
    examples = Path(__file__).resolve().parent / "examples" / "object_models"
    if not examples.is_dir():
        raise ValueError("Packaged example models are missing; reinstall the complete Y-Map distribution")
    generate(examples, out / "candidates", maximum=6)
    candidates_path = out / "candidates" / "candidates.jsonl"
    records = read_jsonl(candidates_path)
    rows = []
    for i, row in enumerate(records):
        features = np.asarray(list(row["features"].values()), dtype=float)
        # Intentionally obvious toy formulas: no claim of DB measurement or manuscript replication.
        size = float(np.abs(features).sum())
        targets = [1 + size * .03 + (i % 3) * .1, 2 + size * .02 + (i % 4) * .2, .5 + size * .01]
        rows.append({"schema_id": row["schema_id"], "object_model": row["object_model"], **dict(zip(TARGETS, targets)),
                     "label_source": "synthetic_demo", "workload_id": "SOFTWARE_SMOKE_FIXTURE_ONLY", "environment_id": "NO_DATABASE_MEASUREMENTS", "seed": 42})
    measurements_path = out / "DEMO_ONLY_measurements.csv"
    pd.DataFrame(rows).to_csv(measurements_path, index=False)
    report = evaluate(candidates_path, measurements_path, out / "evaluation", {"mode": "demo", "models": ["ridge", "extra_trees"], "ensemble_size": 5, "model_parameters": {"extra_trees": {"n_estimators": 20}}, "seed": 42})
    train(candidates_path, measurements_path, out / "demo_model.joblib", {"mode": "demo", "model": "extra_trees", "ensemble_size": 5, "trees": 20})
    single = [row for row in records if row["object_model"] == records[0]["object_model"]]
    write_jsonl(out / "one_object_model.jsonl", single)
    predict_and_recommend(out / "one_object_model.jsonl", out / "demo_model.joblib", out / "recommendations", allow_demo=True)
    return {"status": "software_smoke_completed", "warning": "ALL demo targets are synthetic fixtures. Do not use these outputs as evidence for the paper.", "out_dir": str(out), "evaluation": report}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="ymap", description="Y-Map research framework.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report dependencies without printing credentials")
    gen = sub.add_parser("generate", help="Generate candidates; does not measure or certify application semantics")
    gen.add_argument("--models", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--max-candidates", type=int)
    gen.add_argument("--backend", choices=["python", "alloy"], default="python")
    gen.add_argument("--alloy-jar")
    profile = sub.add_parser("profile", help="Execute real PostgreSQL profiling in isolated temporary schemas")
    profile.add_argument("--candidates", required=True)
    profile.add_argument("--config", required=True)
    profile.add_argument("--out", required=True)
    for name in ["validate", "evaluate", "train"]:
        item = sub.add_parser(name)
        item.add_argument("--candidates", required=True)
        item.add_argument("--measurements", required=True)
        item.add_argument("--mode", choices=["research", "demo"], default="research")
        item.add_argument("--config")
        item.add_argument("--out", required=name != "validate")
    rec = sub.add_parser("recommend", help="Predict from candidates and a trusted fitted model; needs no measurements")
    rec.add_argument("--candidates", required=True)
    rec.add_argument("--model", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--weights", type=float, nargs=3)
    rec.add_argument("--risk-lambda", type=float, default=0.)
    rec.add_argument("--selection", choices=["pareto", "preference", "beam"], default="pareto")
    rec.add_argument("--budget", type=int)
    rec.add_argument("--allow-demo", action="store_true")
    stats = sub.add_parser("statistics", help="System-level paired comparisons from existing per-system results")
    stats.add_argument("--per-system", required=True)
    stats.add_argument("--reference", required=True)
    stats.add_argument("--metrics", nargs="+", default=["gd", "hypervolume", "f1"])
    stats.add_argument("--out", required=True)
    demo = sub.add_parser("demo", help="Software smoke fixture only; never manuscript evidence")
    demo.add_argument("--out", default="runs/DEMO_ONLY")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = environment()
            result["executables"] = {name: shutil.which(name) for name in ["java", "psql", "docker"]}
            result["optional_dependencies"] = {name: importlib.util.find_spec(name) is not None for name in ["torch", "transformers", "psycopg", "xgboost"]}
        elif args.command == "generate":
            if args.max_candidates is not None and args.max_candidates < 1:
                raise ValueError("max-candidates must be positive")
            result = generate(args.models, args.out, args.max_candidates, args.backend, args.alloy_jar)
        elif args.command == "profile":
            from .profiling import profile_candidates
            result = {"measurements": str(profile_candidates(read_jsonl(args.candidates), load_config(args.config), Path(args.out)))}
        elif args.command == "validate":
            result = validate_dataset(args.candidates, args.measurements, args.mode)
            if args.out:
                write_json(args.out, result)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            config = load_config(args.config)
            config["mode"] = args.mode
            result = evaluate(args.candidates, args.measurements, args.out, config)
        elif args.command == "train":
            from .inference import train
            config = load_config(args.config)
            config["mode"] = args.mode
            result = train(args.candidates, args.measurements, args.out, config)
        elif args.command == "recommend":
            from .inference import predict_and_recommend
            result = predict_and_recommend(args.candidates, args.model, args.out, args.weights, args.risk_lambda, args.selection, args.budget, allow_demo=args.allow_demo)
        elif args.command == "statistics":
            from .statistics import compare_system_metrics
            result = compare_system_metrics(pd.read_csv(args.per_system, keep_default_na=False), args.reference, args.metrics)
            write_json(args.out, result)
        else:
            result = run_demo(args.out)
        print(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Y-Map: {exc}", file=sys.stderr)
        return 2
