#!/usr/bin/env python3
"""Y-Map: runnable five-view predictor for the manuscript's Stage C.

Commands:
  check-cohort  Inspect the 12,500-schema DDL inventory without training.
  check-data    Check measured model inputs and the eleven system folds.
  check-model   Check the local encoder configuration and CUDA dependencies.
  fit           Train and calibrate one paper fold, saving its five members.
  evaluate      Run all eleven held-out systems and all three repetitions.
  predict       Predict from prepared Stage B JSONL and a fitted ensemble.

The Y-Map v0.5 repository supplies the Stage B and neural architecture
modules. Run from the repository root,
or specify --project-root. Nothing is downloaded by this entry point.

The default Llama backend requires an explicitly supplied local GPTQ INT4
Llama 2 7B checkpoint. GPTQ is a documented INT4 implementation choice; the
historical quantizer and checkpoint hashes were not stated in the manuscript.
FP4 is a separate compatibility option and must be chosen explicitly.

Example from the repository root:
    python -m ymap_research.ymap_paper_main check-cohort runs/ymap_cohort_12500_postgresql_ready
    python -m ymap_research.ymap_paper_main check-data --dataset native --config config/native_five_view.example.yaml
    python -m ymap_research.ymap_paper_main fit --dataset native --config runs/native_five_view.yaml --test-system banking --out runs/native_paper_fit
    python -m ymap_research.ymap_paper_main predict --candidates runs/native_prediction_inputs/candidates.jsonl --model-dir runs/native_paper_fit --out runs/native_paper_predictions

The native dataset has 728 measured schemas and is not the historical 12,500
paper cohort. For paper-reference, pass a verified --manifest and the paper
study config. The 12,500-schema SQL inventory is not a measured model dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


TARGET_ORDER = ("insert_ms", "query_ms", "storage_mb")
SEED_SETS = (tuple(range(5)), tuple(range(5, 10)), tuple(range(10, 15)))
SPEC = {
    "reference": "Y-Map manuscript, Stages B-C, uncertainty recalibration, Appendix B",
    "views": ["graphcodebert", "codet5", "llama", "graph", "symbolic"],
    "targets": list(TARGET_ORDER),
    "projection": 512, "trunk": [512, 512], "dropout": 0.2,
    "gate": "shared softmax(v^T tanh(Wa h_m)); weighted sum, then LayerNorm",
    "loss": "summed Huber(delta=1) on training-fold standardized log targets",
    "ensemble_seeds": [list(x) for x in SEED_SETS],
    "folds": "11 held-out systems; next cyclic system is validation; remaining nine are training",
    "calibration": "Gaussian validation-only per-objective multiplier; undefined if validation spread is zero",
    "quantization_note": "GPTQ INT4 is supported as an explicit implementation choice; the historical quantizer is unknown.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"Expected an object in {path}")
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def check_cohort(root):
    """Report readiness of the supplied PostgreSQL cohort without using labels."""
    root = Path(root).resolve()
    manifest = read_json(root / "postgresql_bundle_manifest.json")
    path = root / "candidates.csv"
    require(digest_file(path) == manifest.get("candidates_csv_sha256"), "candidates.csv does not match its bundle hash")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"schema_id", "scenario", "object_model", "mapping_status", "common_protocol_status",
                  "sql_dialect_status", "sql_path", "postgresql_sql_sha256"}
        require(needed <= set(reader.fieldnames or ()), "Not a Y-Map prepared SQL inventory")
        rows = list(reader)
    require(len(rows) == len({r["schema_id"] for r in rows}) == manifest.get("sql_file_count") == 12500,
            "Inventory is not a 12,500-candidate cohort with unique schema IDs")
    scenario = Counter(r["scenario"] for r in rows)
    require(scenario == {"native": 728, "archive": 11772}, f"Unexpected inventory sources: {scenario}")
    sql_paths = set()
    for row in rows:
        relative = Path(row["sql_path"])
        sql = (root / relative).resolve()
        require(not relative.is_absolute() and sql.is_relative_to(root) and sql.suffix.lower() == ".sql",
                f"Invalid SQL path for {row['schema_id']}")
        require(sql not in sql_paths, f"Duplicate SQL path: {relative}")
        require(digest_file(sql) == row["postgresql_sql_sha256"], f"SQL checksum mismatch: {relative}")
        sql_paths.add(sql)
    system_count = len({r["object_model"] for r in rows})
    missing_mapping = sum(r["mapping_status"] != "recorded" for r in rows)
    unverified_protocol = sum(r["common_protocol_status"] != "verified" for r in rows)
    blockers = []
    if missing_mapping:
        blockers.append(f"{missing_mapping:,} candidates lack a recorded mapping for the five views.")
    if unverified_protocol or manifest.get("common_protocol_verified") is not True:
        blockers.append("This SQL inventory lacks verified common-protocol PostgreSQL performance measurements.")
    if system_count != 11:
        blockers.append(f"The inventory has {system_count} catalog systems; the paper uses 11 defined system folds.")
    blockers.append("This CSV lists SQL; fit/evaluate need measured Stage B JSONL and an aligned measurements CSV.")
    return {
        "status": "inventory_checked",
        "inventory_candidates": len(rows),
        "verified_sql_files": len(sql_paths),
        "by_scenario": dict(scenario),
        "catalog_systems": system_count,
        "mapping_not_recorded": missing_mapping,
        "protocol_not_verified": unverified_protocol,
        "can_use_inventory_directly_for_fit_or_evaluate": False,
        "blockers": blockers,
    }


def _repo_bootstrap(project_root):
    src = Path(project_root).expanduser().resolve() / "src"
    if (src / "ymap_research" / "paper_model.py").is_file():
        sys.path.insert(0, str(src))


def _config(args):
    from ymap_research.cli import load_config

    result = load_config(args.config)
    require(isinstance(result, dict), "The experiment config must be a YAML mapping")
    return dict(result)


def _load_measured(args):
    """Preserve dataset provenance and never invent the archive mappings."""
    from ymap_research.paper_data import resolve_dataset
    from ymap_research.paper_evaluation import (
        NATIVE_FIVE_VIEW, _load_paper_data, _row_requirements, _validate_canonical_pool,
    )
    from ymap_research.paper_prediction_inputs import native_prediction_records

    dataset = resolve_dataset(args.project_root, args.dataset, dataset_root=args.dataset_root, manifest_path=args.manifest)
    cfg = _config(args)
    scope = NATIVE_FIVE_VIEW if dataset.name == "native" else "paper"
    require(cfg.get("scope") == scope, f"Expected scope: {scope} for --dataset {dataset.name}")
    require(cfg.get("views") is None, "The manuscript model requires all five views; omit the views override")
    require(cfg.get("repetition_seeds") is None or cfg["repetition_seeds"] == [list(s) for s in SEED_SETS],
            "The manuscript fixes seed sets 0–4, 5–9, and 10–14")
    cfg.update(dataset.evaluation_config())
    cfg.update(dataset=dataset.name, protocol_provenance=dataset.provenance)
    if scope == NATIVE_FIVE_VIEW:
        cfg["measurement_rows"] = "protocol_medians"
    records, measured, _ = _load_paper_data(dataset.candidates_path, dataset.measurements_path, cfg)
    counts = dataset.check_alignment(measured)
    rows_mode = _row_requirements(measured, cfg, scope)
    if scope == "paper":
        require(dataset.label_source == "postgresql_measured",
                "The reference workflow requires documented PostgreSQL measurements")
        require(counts["measured_candidates"] == 12500, "The paper-reference cohort must contain 12,500 measured candidates")
    if scope == NATIVE_FIVE_VIEW:
        records = native_prediction_records(records, include_text=True)
    _validate_canonical_pool(records)
    order = cfg.get("system_order")
    require(isinstance(order, (list, tuple)) and len(order) == 11
            and all(isinstance(v, str) and v for v in order)
            and len(set(order)) == 11 and set(order) == set(measured["object_model"]),
            "system_order must explicitly list all 11 measured system IDs in their cyclic order")
    require([(r["schema_id"], r["object_model"]) for r in records]
            == list(zip(measured["schema_id"], measured["object_model"])),
            "Stage B records and measurement rows do not align")
    dataset_meta = {
        **dataset.metadata(), "observed_counts": counts,
        "input_sha256": dataset.input_hashes(), "measurement_rows_mode": rows_mode,
        "reference_status": ("declared_reference_input_not_independently_certified" if scope == "paper"
                             else "separate_native_study_not_manuscript_reference"),
    }
    return records, measured, cfg, dataset_meta


def _output(path):
    from ymap_research.paper_cli import _new_output
    return _new_output(Path(path).expanduser().resolve())


def _fit_command(args, runner):
    import numpy as np
    from ymap_research.common import TARGETS, json_ready, write_json
    from ymap_research.paper_evaluation import gaussian_multiplier
    from ymap_research.paper_predict import _estimate

    require(tuple(TARGETS) == TARGET_ORDER, "Installed Y-Map objective order differs from the manuscript runner")
    records, measured, cfg, dataset_meta = _load_measured(args)
    order = list(cfg["system_order"])
    require(args.test_system in order, f"Unknown test system {args.test_system!r}; choose one of {order}")
    validation = order[(order.index(args.test_system) + 1) % len(order)]
    systems = measured["object_model"].to_numpy()
    train_systems = [s for s in order if s not in {validation, args.test_system}]
    train_i = np.concatenate([np.flatnonzero(systems == s) for s in train_systems])
    val_i = np.flatnonzero(systems == validation)
    test_i = np.flatnonzero(systems == args.test_system)
    target = measured[list(TARGETS)].to_numpy(dtype=np.float64)

    runner.validate_backend(cfg, args.llama_backend, runtime=True)
    out = _output(args.out)
    seeds = SEED_SETS[args.repetition - 1]
    ensemble, _, validation_members = runner.fit_fold(
        [records[i] for i in train_i], target[train_i],
        [records[i] for i in val_i], target[val_i],
        [], cfg, seeds, args.llama_backend,
    )
    _, means, std, _ = _estimate(validation_members, 5, len(val_i), np.ones(len(TARGETS)))
    factors = gaussian_multiplier(target[val_i], means, std)
    ensemble_dir = out / "ensemble"
    ensemble.save(ensemble_dir)
    manifest = {
        "status": "fitted", "scope": cfg["scope"], "backend": runner.normalize_backend(args.llama_backend),
        "repetition": args.repetition, "seeds": list(seeds), "test_system": args.test_system,
        "validation_system": validation, "system_order": order,
        "train_systems": train_systems,
        "split_sizes": {"train": len(train_i), "validation": len(val_i), "test": len(test_i)},
        "dataset": dataset_meta, "calibration_factors": factors,
        "ensemble_manifest_sha256": digest_file(ensemble_dir / "manifest.json"),
        "ensemble_member_sha256": {p.name: digest_file(p) for p in sorted(ensemble_dir.glob("member-*.npz"))},
        "claims": "Fits one paper-design fold; the held-out test system is not used for training or calibration.",
    }
    write_json(out / "fit_manifest.json", manifest)
    emit(json_ready({**manifest, "output": str(out)}))


def _predict_command(args, runner):
    import numpy as np
    from ymap_research.common import TARGETS, environment, json_ready, write_json
    from ymap_research.paper_predict import _calibration_factors, _config_overrides, _estimate, _records

    require(tuple(TARGETS) == TARGET_ORDER, "Installed Y-Map objective order differs from the manuscript runner")
    model_dir = args.model_dir.resolve()
    ensemble_dir = model_dir / "ensemble"
    fit_path = model_dir / "fit_manifest.json"
    fit = read_json(fit_path)
    require(fit.get("status") == "fitted", "Pass the model folder created by fit")
    backend = runner.normalize_backend(fit.get("backend"))
    if args.llama_backend is not None:
        require(runner.normalize_backend(args.llama_backend) == backend, "Requested backend differs from the fitted model")
    manifest_path = ensemble_dir / "manifest.json"
    doc = read_json(manifest_path)
    metadata = doc["metadata"]
    views = tuple(metadata.get("views", ()))
    seeds = tuple(metadata.get("seeds", ()))
    require(views == tuple(SPEC["views"]) and seeds in SEED_SETS, "Prediction requires a fitted five-view, five-member ensemble")
    require(seeds == tuple(fit.get("seeds", ())) and metadata.get("llama_backend") == backend,
            "Model backend or seed order differs from the fit manifest")
    require(digest_file(manifest_path) == fit.get("ensemble_manifest_sha256"),
            "Fit/calibration manifest is not bound to the saved ensemble")
    member_hashes = {f"member-{i}.npz": digest_file(ensemble_dir / f"member-{i}.npz") for i in range(5)}
    require(member_hashes == fit.get("ensemble_member_sha256"), "Fit manifest does not match the saved ensemble members")
    factors = _calibration_factors(fit)
    require(args.batch_size > 0, "--batch-size must be positive")
    candidates = args.candidates.resolve()
    if candidates.suffix.lower() == ".csv":
        raise ValueError("predict needs Stage B candidates.jsonl with DDL, graph and descriptors; the SQL inventory CSV is insufficient")
    require(candidates.is_file(), f"Prepared candidate JSONL was not found: {candidates}")
    overrides = _config_overrides(_config(args) if args.config else None)
    config = {**doc["config"], **overrides}
    runner.validate_backend({"neural": config}, backend, runtime=True)
    ensemble = runner.ensemble_class(backend).load(ensemble_dir, **overrides)
    out = _output(args.out)
    digest, groups, count = hashlib.sha256(), set(), 0
    with TemporaryDirectory(prefix=".prediction-", dir=out) as work:
        csv_path = Path(work) / "predictions.csv"
        report_path = Path(work) / "prediction_manifest.json"
        with candidates.open("rb") as source, csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            prefixes = ["mean", "sample_std", "calibrated_std", *(f"member_seed_{seed}" for seed in seeds)]
            writer.writerow(["schema_id", "object_model", *(f"{prefix}_{target}" for prefix in prefixes for target in TARGETS)])
            candidates_iter = _records(source, views, digest)
            while batch := list(islice(candidates_iter, args.batch_size)):
                members, means, std, calibrated = _estimate(ensemble.predict_members(batch), 5, len(batch), factors)
                for i, record in enumerate(batch):
                    groups.add(record["object_model"])
                    values = np.concatenate((means[i], std[i], calibrated[i], members[:, i, :].ravel()))
                    writer.writerow([record["schema_id"], record["object_model"],
                                     *(float(v) if np.isfinite(v) else None for v in values)])
                count += len(batch)
        require(count > 0, "Prepared candidate file contains no records")
        report = {
            "status": "predicted", "scope": "inference_only", "output": str(out), "backend": backend,
            "candidate_count": count, "object_models": sorted(groups), "views": views, "seeds": seeds,
            "objectives": TARGETS, "prediction_file": csv_path.name, "predictions_sha256": digest_file(csv_path),
            "candidate_input": {"path": str(candidates), "sha256": digest.hexdigest()},
            "fit_manifest_sha256": digest_file(fit_path), "ensemble_member_sha256": member_hashes,
            "calibration_factors": factors, "training_dataset": fit.get("dataset"),
            "text_encoders": ensemble.metadata.get("text_encoders", {}),
            "software": environment(),
            "note": "Mean, raw sample spread and validation-calibrated spread are in original measurement units. No measurements were generated.",
        }
        write_json(report_path, report)
        require(not (out / csv_path.name).exists() and not (out / report_path.name).exists(),
                "Prediction output files already exist")
        csv_path.replace(out / csv_path.name)
        report_path.replace(out / report_path.name)
    emit(json_ready(report))


def _measured_options(parser):
    parser.add_argument("--dataset", choices=("native", "paper-reference"), required=True)
    parser.add_argument("--dataset-root", type=Path, help="optional override of Dataset/YMap")
    parser.add_argument("--manifest", type=Path, help="verified dataset manifest (required for paper-reference)")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with scope and 11-system cyclic order")


def _backend_option(parser, default="gptq-int4"):
    parser.add_argument("--llama-backend", choices=("gptq-int4", "fp4"), default=default,
                        help="GPTQ INT4 (local checkpoint) or explicitly separate FP4 compatibility mode")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Y-Map five-view manuscript predictor (Stage C).")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Y-Map checkout, default: current directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("describe", help="Print the manuscript architecture and training settings")
    cohort = sub.add_parser("check-cohort", help="Check the 12,500-schema DDL inventory, without training")
    cohort.add_argument("cohort", type=Path, help="folder with postgresql_bundle_manifest.json and candidates.csv")
    data = sub.add_parser("check-data", help="Validate measured inputs and the 11 system folds, without GPU")
    _measured_options(data)
    model = sub.add_parser("check-model", help="Validate local checkpoint configuration and CUDA dependencies")
    model.add_argument("--config", type=Path, required=True)
    model.add_argument("--files-only", action="store_true", help="check files without importing CUDA")
    _backend_option(model)
    fit = sub.add_parser("fit", help="Train and save five calibrated members for one held-out paper fold")
    _measured_options(fit)
    _backend_option(fit)
    fit.add_argument("--test-system", required=True, help="held-out system; the next cyclic system is validation")
    fit.add_argument("--repetition", type=int, choices=(1, 2, 3), default=1)
    fit.add_argument("--out", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Run the 11 held-out folds × 3 repetitions × 5 members")
    _measured_options(evaluate)
    _backend_option(evaluate)
    evaluate.add_argument("--out", type=Path, required=True)
    predict = sub.add_parser("predict", help="Score prepared Stage B JSONL using a saved ensemble")
    _backend_option(predict, default=None)
    predict.add_argument("--candidates", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--config", type=Path, help="optional override of checkpoint paths, cache, or device")
    predict.add_argument("--batch-size", type=int, default=256, help="number of candidates per prediction chunk")
    predict.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            emit(SPEC)
            return 0
        if args.command == "check-cohort":
            emit(check_cohort(args.cohort))
            return 0
        _repo_bootstrap(args.project_root)
        if args.command == "check-data":
            _, _, cfg, dataset_meta = _load_measured(args)
            emit({"status": "measured_stage_b_inputs_verified", "dataset": dataset_meta,
                  "scope": cfg["scope"], "system_order": cfg["system_order"], "neural_execution_tested": False})
            return 0
        runner = _model_runner()
        if args.command == "check-model":
            emit(runner.validate_backend(_config(args), args.llama_backend, runtime=not args.files_only))
        elif args.command == "fit":
            _fit_command(args, runner)
        elif args.command == "evaluate":
            from ymap_research.common import json_ready
            records, measured, cfg, dataset_meta = _load_measured(args)
            runner.validate_backend(cfg, args.llama_backend, runtime=True)
            result = runner.evaluate_cross_system(records, measured, cfg, args.llama_backend, _output(args.out), dataset_meta)
            emit(json_ready(result))
        else:
            _predict_command(args, runner)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, csv.Error) as exc:
        print(f"Y-Map: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Use the Y-Map repository's Python environment with the required packages installed.", file=sys.stderr)
        return 2
    return 0


def _model_runner():
    import importlib.metadata
    import json
    from pathlib import Path

    import numpy as np
    from ymap_research import paper_model as pm


    def normalize_backend(backend):
        aliases = {"int4": "gptq-int4", "gptq": "gptq-int4", "bitsandbytes-fp4": "fp4"}
        backend = aliases.get(backend, backend)
        if backend not in ("gptq-int4", "fp4"):
            raise ValueError("Choose the Llama backend gptq-int4 or fp4 (explicit approximation).")
        return backend


    def _gptq_llama_config(path):
        """Check a local, already-quantized Hugging Face Llama 2 7B checkpoint."""
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Llama checkpoint config is not a JSON object: {path}")
        shape = {
            "model_type": "llama", "hidden_size": 4096, "intermediate_size": 11008,
            "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 32000,
        }
        if (any(config.get(key) != value for key, value in shape.items())
                or config.get("num_key_value_heads", 32) != 32
                or config.get("max_position_embeddings") != 4096
                or config.get("rope_scaling") is not None):
            raise ValueError(f"Checkpoint config does not match Llama 2 7B: {path}")
        quant = config.get("quantization_config")
        if (not isinstance(quant, dict) or quant.get("quant_method") != "gptq"
                or type(quant.get("bits")) is not int or quant["bits"] != 4
                or quant.get("load_in_4bit") or quant.get("_load_in_4bit")
                or any(key.startswith("bnb_") for key in quant)):
            raise ValueError(
                "gptq-int4 requires a local prequantized Llama 2 7B checkpoint whose config.json "
                "declares quant_method=gptq and bits=4. FP4 and on-load quantization are not accepted."
            )
        if not any((path / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
            raise FileNotFoundError(f"GPTQ checkpoint needs local Hugging Face safetensors weights: {path}")
        return quant


    def _backend_environment():
        environment = pm._paper_environment()
        for package in ("gptqmodel", "optimum", "auto-gptq"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = None
            environment["packages"][package] = version
        return environment


    def validate_backend(config, backend, *, runtime=True):
        """Validate local checkpoint configuration; optionally require the GPU runtime."""
        backend = normalize_backend(backend)
        config = pm.PaperModelConfig.from_mapping(config)
        pm._check_protocol(config, SEED_SETS[0], None)
        specs = pm._checkpoint_specs(config, pm.PAPER_VIEWS)
        for family, (checkpoint, _) in specs.items():
            if not any(p.is_file() for pattern in ("*.safetensors", "*.bin") for p in checkpoint.glob(pattern)):
                raise FileNotFoundError(f"No local weight files found for {family}: {checkpoint}")
        path, _ = specs["llama"]
        if backend == "gptq-int4":
            quant = _gptq_llama_config(path)
        else:
            raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("quantization_config"):
                raise ValueError("FP4 requires the original unquantized local Llama 2 7B checkpoint.")
            quant = {"quant_method": "bitsandbytes", "bits": 4, "format": "fp4"}
        result = {
            "backend": backend, "llama_checkpoint": str(path), "quantization_config": quant,
            "checkpoint_check": "local configuration only; load verifies actual quantized modules",
        }
        if runtime:
            torch = pm._torch()
            result["device"] = str(pm._device(torch, config, pm.PAPER_VIEWS))
            packages = _backend_environment()["packages"]
            expected = ("transformers", "accelerate", "optimum", "gptqmodel") if backend == "gptq-int4" else (
                "transformers", "accelerate", "bitsandbytes",
            )
            missing = [package for package in expected if packages.get(package) is None]
            if missing:
                raise ImportError(f"Missing {backend} runtime package(s): {', '.join(missing)}")
            result["packages"] = packages
        return result


    class _FrozenGptqLlama(pm._FrozenText):
        def __init__(self, path, batch_size, device, cache_dir, torch):
            quant = _gptq_llama_config(path)
            super().__init__("llama", path, batch_size, device, None)
            self.metadata.update({
                "weight_loading": "hf_prequantized_gptq_int4",
                "quantization_config": quant,
                "quantization_note": (
                    "Local integer GPTQ 4-bit Llama 2 7B checkpoint. The historical checkpoint "
                    "revision and quantizer are not established by the manuscript."
                ),
            })
            self.identity = json.dumps({
                **self.metadata, "version": pm.MODEL_VERSION, "gptq_backend_version": 1,
                "device": str(device), "batch_size": batch_size,
                "packages": _backend_environment()["packages"],
                "cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_capability": torch.cuda.get_device_capability(device),
            }, sort_keys=True).encode()
            self.cache_dir = Path(cache_dir).expanduser() / "llama-gptq-int4" if cache_dir else None
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        def _load(self, torch):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install Transformers and a compatible Optimum/GPTQModel CUDA runtime.") from exc
            try:
                model, diagnostics = AutoModelForCausalLM.from_pretrained(
                    str(self.path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
                    output_loading_info=True, torch_dtype=torch.float16,
                    device_map={"": str(self.device)},
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.path), local_files_only=True, trust_remote_code=False, use_fast=True,
                )
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                raise RuntimeError(f"Cannot load local GPTQ INT4 Llama 2 from {self.path}: {exc}") from exc
            quant = getattr(model.config, "quantization_config", None)
            if hasattr(quant, "to_dict"):
                quant = quant.to_dict()
            method = getattr(model, "quantization_method", None)
            if (not isinstance(quant, dict) or quant.get("bits") != 4
                    or getattr(method, "value", method) != "gptq"
                    or not getattr(model, "is_quantized", False)):
                raise ValueError("Transformers did not load a verified GPTQ 4-bit Llama model.")
            for key in ("missing_keys", "mismatched_keys", "error_msgs"):
                if diagnostics.get(key):
                    raise ValueError(f"Llama checkpoint could not load fully: {key}={diagnostics[key]}")

            backbone = getattr(model, "model", None)
            layers = getattr(backbone, "layers", ())
            if len(layers) != 32:
                raise ValueError("Expected the 32-layer Llama 2 7B backbone.")
            integer_types = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
            for index, layer in enumerate(layers):
                for block_name, projections in (
                    ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                    ("mlp", ("gate_proj", "up_proj", "down_proj")),
                ):
                    block = getattr(layer, block_name, None)
                    for name in projections:
                        linear = getattr(block, name, None)
                        packed = getattr(linear, "qweight", None)
                        if (not torch.is_tensor(packed) or packed.dtype not in integer_types
                                or not packed.numel() or packed.device.type != "cuda"
                                or (self.device.index is not None and packed.device.index != self.device.index)
                                or getattr(linear, "bits", 4) != 4):
                            raise ValueError(
                                f"Llama layer {index}.{block_name}.{name} lacks supported packed GPTQ INT4 weights."
                            )
            backbone.eval()
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            if tokenizer.pad_token_id is None:
                raise ValueError("The Llama tokenizer needs a pad, eos, or unk token.")
            tokenizer.padding_side = "right"
            capacity = pm._usable_token_limit(model.config, tokenizer, "llama")
            if capacity is not None and capacity < self.limit:
                raise ValueError(f"Llama checkpoint supports {capacity} tokens, below the paper limit {self.limit}.")
            # Base encode() pools last_hidden_state. Calling the backbone instead
            # of the language-model head also avoids computing token logits.
            return backbone, tokenizer


    class Int4Ensemble(pm.PaperEnsemble):
        """Same trainable five-view network with a verified GPTQ INT4 Llama view."""

        def __init__(self, config, views, seeds, device, torch, checkpoint_specs):
            if "llama" not in checkpoint_specs or device.type != "cuda":
                raise ValueError("The GPTQ INT4 ensemble requires its local Llama checkpoint and CUDA.")
            super().__init__(config, views, seeds, device, torch, {
                family: spec for family, spec in checkpoint_specs.items() if family != "llama"
            })
            path, batch = checkpoint_specs["llama"]
            self.encoders["llama"] = _FrozenGptqLlama(path, batch, device, config.cache_dir, torch)


    def ensemble_class(backend):
        return Int4Ensemble if normalize_backend(backend) == "gptq-int4" else pm.PaperEnsemble


    def fit_fold(train_records, train_y, val_records, val_y, test_records, config, seeds, backend):
        """Return the ensemble plus test and validation members in original units."""
        backend = normalize_backend(backend)
        if backend == "fp4":
            ensemble, inputs, validation, test = pm._fit(
                train_records, train_y, val_records, val_y, test_records, config, seeds, None,
            )
        else:
            config = pm.PaperModelConfig.from_mapping(config)
            seeds = pm._check_protocol(config, seeds, None)
            views = pm.PAPER_VIEWS
            train_records, val_records, test_records = map(list, (train_records, val_records, test_records))
            pm._check_system_splits(train_records, val_records, test_records)
            train_count, val_count = len(train_records), len(val_records)
            train_y = pm._target_array(train_y, train_count, "training")
            val_y = pm._target_array(val_y, val_count, "validation")
            validate_backend(config, backend)
            torch = pm._torch()
            device = pm._device(torch, config, views)
            ensemble = Int4Ensemble(config, views, seeds, device, torch, pm._checkpoint_specs(config, views))
            ensemble.target_mean, ensemble.target_scale = pm._standardizer(np.log(train_y))
            inputs = ensemble._inputs([*train_records, *val_records, *test_records], train_size=train_count)
            labels = ((np.log(np.concatenate((train_y, val_y))) - ensemble.target_mean) / ensemble.target_scale).astype(np.float32)
            labels_torch = torch.as_tensor(labels, device=device)
            validation = np.arange(train_count, train_count + val_count)
            test = np.arange(train_count + val_count, train_count + val_count + len(test_records))
            # Mirror the repository fold's training and early stopping without a
            # global monkeypatch of its FP4 ensemble or trainable model code.
            criterion = torch.nn.HuberLoss(reduction="sum", delta=1.0)
            audits = []
            for seed in seeds:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                member = ensemble._new_member()
                optimizer = torch.optim.Adam(member.parameters(), lr=1e-4, betas=(0.9, 0.999))
                best_state, best_loss, best_epoch = None, float("inf"), 0
                for epoch in range(1, config.epochs + 1):
                    member.train()
                    order = rng.permutation(train_count)
                    for start in range(0, train_count, config.batch_size):
                        indices = order[start:start + config.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(member(ensemble._batch(inputs, indices)), labels_torch[indices])
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite training loss for seed {seed}, epoch {epoch}.")
                        loss.backward()
                        optimizer.step()
                    predictions = ensemble._predict(member, inputs, validation)
                    val_loss = criterion(
                        torch.from_numpy(predictions), torch.from_numpy(labels[validation]),
                    ).item() / val_count
                    if not np.isfinite(val_loss):
                        raise RuntimeError(f"Nonfinite validation loss for seed {seed}, epoch {epoch}.")
                    if val_loss < best_loss:
                        best_loss, best_epoch = val_loss, epoch
                        best_state = {key: tensor.detach().cpu().clone() for key, tensor in member.state_dict().items()}
                    elif epoch - best_epoch >= config.patience:
                        break
                member.load_state_dict(best_state)
                ensemble.members.append(member.eval())
                audits.append({
                    "seed": seed, "epochs": epoch, "best_epoch": best_epoch,
                    "validation_huber_sum_per_row": best_loss,
                })
            ensemble.metadata = {
                "model_version": pm.MODEL_VERSION, "full_paper_configuration": True,
                "views": list(views), "seeds": list(seeds), "fusion": config.fusion,
                "architecture": {
                    "projection_width": 512, "trunk": [512, 512], "dropout": 0.2, "heads": 3,
                    "graph_width": 256, "graph_layers": 2, "graph_bases": 4,
                    "graph_relations": list(ensemble.relations), "graph_readout": "mean_and_max",
                    "graph_normalization": "mean over incoming edges by relation, learned self transform, no added reverse edges",
                    "graph_data_types_train_vocabulary": ensemble.graph_vocabulary,
                    "symbolic_mlp": [config.symbolic_hidden, config.symbolic_hidden],
                    "gate_hidden": config.gate_hidden,
                },
                "training": {
                    "optimizer": "Adam", "learning_rate": 1e-4, "betas": [0.9, 0.999],
                    "batch_size": config.batch_size, "patience": config.patience, "max_epochs": config.epochs,
                    "loss": "summed Huber, delta=1, standardized natural-log targets", "members": audits,
                },
                "text_encoders": {name: encoder.metadata for name, encoder in ensemble.encoders.items()},
                "training_fold_statistics": {
                    "log_target_mean": ensemble.target_mean.tolist(),
                    "log_target_scale": ensemble.target_scale.tolist(),
                    "descriptor_mean": ensemble.descriptor_mean.tolist(),
                    "descriptor_scale": ensemble.descriptor_scale.tolist(),
                },
                "device": str(device), "environment": _backend_environment(),
            }
        ensemble.metadata["llama_backend"] = backend
        ensemble.metadata["historical_quantizer_verified"] = False
        return ensemble, ensemble._predict_members(inputs, test), ensemble._predict_members(inputs, validation)

    def evaluate_cross_system(records, measurements, config, backend, output, dataset_meta):
        """Run prediction evaluation: 11 held-out systems, three five-member fits each."""
        import csv
        import sys
        import time
        from pathlib import Path

        import numpy as np

        from ymap_research.common import TARGETS, json_ready, write_json
        from ymap_research.paper_evaluation import (
            PAPER_SEEDS, _ids, _reports, _uncertainty_metrics, _write_predictions,
            gaussian_multiplier, prediction_metrics,
        )
        from ymap_research.paper_predict import _estimate

        started = time.perf_counter()
        output = Path(output)
        records = list(records)
        if not records or len(records) != len(measurements):
            raise ValueError("Every prepared candidate must align with a measured objective row.")
        ids = measurements["schema_id"].tolist()
        groups = measurements["object_model"].tolist()
        if len(set(ids)) != len(ids) or any(
            row.get("schema_id") != sid or row.get("object_model") != group
            for row, sid, group in zip(records, ids, groups)
        ):
            raise ValueError("Prepared candidate IDs and systems must uniquely align with measurements.")

        order = _ids(config.get("system_order"), "system_order")
        if len(order) != 11 or set(order) != set(groups):
            raise ValueError("The prediction protocol requires all 11 measured systems in an explicit cyclic order.")
        shared = _ids(config.get("shared_systems", []), "shared_systems")
        if set(shared) - set(order):
            raise ValueError("shared_systems must be drawn from system_order.")
        repeats = config.get("repetition_seeds", PAPER_SEEDS)
        if (
            not isinstance(repeats, (list, tuple))
            or not all(isinstance(r, (list, tuple)) and all(type(s) is int for s in r) for r in repeats)
            or tuple(tuple(r) for r in repeats) != PAPER_SEEDS
        ):
            raise ValueError("Prediction evaluation requires exactly the seed sets 0–4, 5–9, and 10–14.")

        y = measurements[list(TARGETS)].to_numpy(dtype=np.float64)
        if y.shape != (len(records), len(TARGETS)) or not np.isfinite(y).all() or np.any(y <= 0):
            raise ValueError("The three measured objectives must be finite and strictly positive.")
        scenario = {}
        for key in ("workload_id", "environment_id", "label_source"):
            distinct = set(measurements[key].tolist())
            if len(distinct) != 1 or any(not isinstance(v, str) or not v.strip() for v in distinct):
                raise ValueError(f"Evaluation requires one declared {key}.")
            scenario[key] = next(iter(distinct))

        scope = config.get("scope", "paper")
        model = "ymap_native_five_view" if scope == "native-five-view" else "ymap"
        positions = {s: np.flatnonzero(np.asarray(groups) == s) for s in order}
        splits, folds = [], []
        for i, test_system in enumerate(order):
            validation_system = order[(i + 1) % len(order)]
            train_systems = [s for s in order if s not in (test_system, validation_system)]
            train = np.concatenate([positions[s] for s in train_systems])
            val, test = positions[validation_system], positions[test_system]
            splits.append((train, val, test))
            folds.append({
                "fold": i + 1, "test_system": test_system, "validation_system": validation_system,
                "training_systems": train_systems, "training_candidates": len(train),
                "validation_candidates": len(val), "test_candidates": len(test),
            })
        write_json(output / "folds.json", folds)

        metrics, uncertainty, fits = [], [], []
        prediction_count = 0
        fields = ["scope", "fold", "repeat", "object_model", "validation_system",
                  "workload_id", "environment_id", "model", "schema_id"]
        fields += [f"{prefix}_{target}" for prefix in ("true", "pred", "raw_std", "recalibrated_std")
                   for target in TARGETS]
        with (output / "oof_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for split, (train, val, test) in zip(folds, splits):
                for repeat, seeds in enumerate(PAPER_SEEDS, start=1):
                    repeat_start = time.perf_counter()
                    print(
                        f"Fold {split['fold']}/11, repeat {repeat}/3: "
                        f"test={split['test_system']}, validation={split['validation_system']}",
                        file=sys.stderr, flush=True,
                    )
                    ensemble, test_members, val_members = fit_fold(
                        [records[i] for i in train], y[train],
                        [records[i] for i in val], y[val],
                        [records[i] for i in test], config, seeds, backend,
                    )
                    _, val_mean, val_std, _ = _estimate(val_members, len(seeds), len(val), np.ones(len(TARGETS)))
                    factors = gaussian_multiplier(y[val], val_mean, val_std)
                    _, mean, std, calibrated = _estimate(test_members, len(seeds), len(test), factors)
                    context = {
                        "scope": scope, "fold": split["fold"], "repeat": repeat,
                        "object_model": split["test_system"], "validation_system": split["validation_system"],
                        "workload_id": scenario["workload_id"], "environment_id": scenario["environment_id"],
                    }
                    _write_predictions(writer, context, model, [ids[i] for i in test], y[test], mean, std, calibrated)
                    prediction_count += len(test)
                    metrics.extend({**context, "model": model, **m} for m in prediction_metrics(y[test], mean))
                    uncertainty.extend({**context, "model": model, **m} for m in _uncertainty_metrics(y[test], mean, std, factors))
                    fits.append({
                        **context, "model": model, "backend": backend, "member_seeds": seeds,
                        "calibration_factors": dict(zip(TARGETS, factors)),
                        "details": ensemble.metadata,
                        "elapsed_seconds": time.perf_counter() - repeat_start,
                    })
                    handle.flush()
                    write_json(output / "training_details.json", fits)
                    del ensemble, test_members, val_members

        # Match the manuscript's unit of aggregation: average repeated runs within
        # each system first, then average systems without candidate-count weighting.
        _reports(output, "prediction_metrics", metrics, ("model", "objective"), shared)
        _reports(output, "uncertainty_metrics", uncertainty, ("model", "objective"), shared)
        report = {
            "status": "evaluated", "scope": scope, "backend": backend,
            "claim": "Prediction evaluation of the declared measured dataset; no claim of historical result reproduction.",
            "dataset": dataset_meta, **scenario, "system_order": order, "shared_systems": shared,
            "repetition_seeds": PAPER_SEEDS, "folds": len(folds), "fits": len(fits),
            "evaluated_candidates": len(records), "oof_prediction_rows": prediction_count,
            "aggregation": "three repetitions averaged within each test system before unweighted system means",
            "calibration": "validation-system Gaussian multiplier on original-unit sample spread; zero validation spread is unavailable",
            "files": {
                "out_of_fold_predictions": "oof_predictions.csv",
                "prediction_metrics": "prediction_metrics_summary.csv",
                "uncertainty_metrics": "uncertainty_metrics_summary.csv",
                "folds": "folds.json", "training": "training_details.json",
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(output / "summary.json", report)
        return json_ready(report)

    return SimpleNamespace(
        normalize_backend=normalize_backend,
        validate_backend=validate_backend,
        ensemble_class=ensemble_class,
        fit_fold=fit_fold,
        evaluate_cross_system=evaluate_cross_system,
    )


if __name__ == "__main__":
    raise SystemExit(main())
