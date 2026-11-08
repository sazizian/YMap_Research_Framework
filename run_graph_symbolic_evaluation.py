"""Run the fixed graph + symbolic CPU diagnostic on the native measured study.

After installing the framework, validate the repository inputs without PyTorch:
    python run_graph_symbolic_evaluation.py --check-only

Uses existing measurements, fresh held-out models, and a new output directory.
The four-candidate software-check checkpoint is never loaded. No database query,
checkpoint download, or modification of original inputs is performed.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run(project_root: Path, check_only: bool = False, *, dataset_root: Path | None = None) -> None:
    root = project_root.expanduser().resolve()
    baseline_config = root / "config" / "evaluation.yaml"
    for path in (root / "pyproject.toml", baseline_config):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required project file not found: {path}. "
                "Run from the YMap project folder or supply --project-root."
            )

    import yaml
    from ymap_research.evaluation import evaluate, load_dataset
    from ymap_research.paper_data import file_sha256, resolve_dataset
    from ymap_research.representations import (
        schema_graph,
        symbolic_feature_names,
        symbolic_matrix,
    )

    dataset = resolve_dataset(root, dataset_root=dataset_root)
    candidates, measurements = dataset.candidates_path, dataset.measurements_path
    config = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The existing evaluation configuration must be a mapping.")
    config.update({
        "models": ["linear", "ridge", "random_forest", "extra_trees", "mlp", "neural"],
        "ensemble_base": "neural",
        "ensemble_size": 5,
        "reference_model": "ensemble_risk",
        "seed": 42,
        "risk_lambda": 1.0,
        "calibration": True,
        "calibration_alpha": 0.10,
        "calibration_fraction": 0.25,
        "selection_mode": "pareto",
        "neural": {
            "device": "cpu",
            "text_encoders": [],
            "use_graph": True,
            "use_symbolic": True,
            "hidden_dim": 32,
            "graph_layers": 2,
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "weight_decay": 0.0001,
            "deterministic": True,
            "target_names": ["insert_ms", "query_ms", "storage_mb"],
        },
    })
    config.update(dataset.evaluation_config())

    print("Checking the existing measured dataset and all schema graphs...", flush=True)
    records, data, _ = load_dataset(candidates, measurements, config)
    counts = dataset.check_alignment(data)
    system_count = counts["source_systems"]
    for record in records:
        schema_graph(record["graph"])
    symbolic_matrix(records, symbolic_feature_names(records))
    print(f"Input checks passed: {len(records)} candidates, {system_count} systems.", flush=True)
    if check_only:
        print("Check-only mode completed; no model fitting was performed.", flush=True)
        return

    import torch

    torch.set_num_threads(2)
    input_hashes = dataset.input_hashes()
    neural_fit_count = system_count * (1 + config["ensemble_size"])
    output = root / "results" / "YMap" / (
        "graph_symbolic_cpu_" + uuid.uuid4().hex[:12]
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", config)
    write_json(output / "run_plan.json", {
        "scope": "Exploratory graph + symbolic CPU diagnostic; no pretrained text encoders.",
        "dataset": dataset.metadata(),
        "input_hashes": {
            "candidates.jsonl": input_hashes["candidates"],
            "measurements.csv": input_hashes["measurements"],
            "datasets.yaml": input_hashes["dataset_manifest"],
            "baseline_evaluation.yaml": file_sha256(baseline_config),
        },
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "candidate_count": len(records),
        "source_system_count": system_count,
        "neural_member_fit_count": neural_fit_count,
        "epochs_per_neural_member": 10,
        "epoch_policy": "Fixed budget; no calibration or test outcomes used for early stopping.",
        "comparison_scope": "Structural model parameters retained from the existing configuration.",
        "interpretation": "Ten epochs are a diagnostic budget, not a convergence guarantee. "
                          "Changes motivated by previous outer-test inspection are exploratory.",
    })
    print(f"Output directory: {output}", flush=True)
    print(f"Starting {system_count} held-out folds, including {neural_fit_count} neural member fits.", flush=True)

    started = time.monotonic()
    stop = threading.Event()
    status_path = output / "run_status.json"
    write_json(status_path, {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })

    def heartbeat() -> None:
        while not stop.wait(30):
            minutes = (time.monotonic() - started) / 60
            print(f"Evaluation running: {minutes:.1f} minutes elapsed.", flush=True)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        summary = evaluate(candidates, measurements, output / "evaluation", config)
    except (Exception, KeyboardInterrupt) as error:
        write_json(status_path, {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise
    else:
        write_json(status_path, {
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
            "summary_file": "evaluation/summary.json",
        })
    finally:
        stop.set()
        worker.join(timeout=1)

    print("Evaluation completed.", flush=True)
    print(f"Summary file: {output / 'evaluation' / 'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, help="Override Dataset/YMap with a copy of the data root.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and graphs without fitting.")
    arguments = parser.parse_args()
    try:
        run(arguments.project_root, arguments.check_only, dataset_root=arguments.dataset_root)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Y-Map: {exc}\n")


if __name__ == "__main__":
    main()
