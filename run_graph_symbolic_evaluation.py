r"""Run the fixed graph + symbolic CPU diagnostic for the current YMap dataset.

Run from the YMap project folder with its existing virtual environment:
    .\.venv\Scripts\python.exe run_graph_symbolic_evaluation.py --project-root .

Uses existing measurements, fresh held-out models, and a new output directory.
The four-candidate software-check checkpoint is never loaded. No database query,
checkpoint download, or modification of original inputs is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run(project_root: Path, check_only: bool = False) -> None:
    root = project_root.expanduser().resolve()
    source = root / "runs" / "all_systems_20261022_205921_551"
    candidates = source / "candidates" / "candidates.jsonl"
    measurements = source / "measurements" / "measurements.csv"
    baseline_config = root / "config" / "evaluation.yaml"
    for path in (root / "pyproject.toml", candidates, measurements, baseline_config):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required project file not found: {path}. "
                "Run from the YMap project folder or supply --project-root."
            )

    import torch
    import yaml
    from ymap_research.evaluation import evaluate, load_dataset
    from ymap_research.representations import (
        schema_graph,
        symbolic_feature_names,
        symbolic_matrix,
    )

    torch.set_num_threads(2)
    config = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The existing evaluation configuration must be a mapping.")
    config.update({
        "mode": "research",
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

    print("Checking the existing measured dataset and all schema graphs...", flush=True)
    records, data, _ = load_dataset(candidates, measurements, config)
    system_count = int(data["object_model"].nunique())
    if len(records) != 728 or system_count != 11:
        raise ValueError("This runner expects 728 aligned candidates across 11 source systems.")
    for record in records:
        schema_graph(record["graph"])
    symbolic_matrix(records, symbolic_feature_names(records))
    print("Input checks passed: 728 candidates, 11 systems.", flush=True)
    if check_only:
        print("Check-only mode completed; no model fitting was performed.", flush=True)
        return

    output = root / "results" / "YMap" / (
        "graph_symbolic_cpu_" + uuid.uuid4().hex[:12]
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", config)
    write_json(output / "run_plan.json", {
        "scope": "Exploratory graph + symbolic CPU diagnostic; no pretrained text encoders.",
        "input_hashes": {
            "candidates.jsonl": sha256(candidates),
            "measurements.csv": sha256(measurements),
            "baseline_evaluation.yaml": sha256(baseline_config),
        },
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "candidate_count": len(records),
        "source_system_count": system_count,
        "neural_member_fit_count": 66,
        "epochs_per_neural_member": 10,
        "epoch_policy": "Fixed budget; no calibration or test outcomes used for early stopping.",
        "comparison_scope": "Structural model parameters retained from the existing configuration.",
        "interpretation": "Ten epochs are a diagnostic budget, not a convergence guarantee. "
                          "Changes motivated by previous outer-test inspection are exploratory.",
    })
    print(f"Output directory: {output}", flush=True)
    print("Starting 11 held-out folds, including 66 neural member fits.", flush=True)

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
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and graphs without fitting.")
    arguments = parser.parse_args()
    run(arguments.project_root, arguments.check_only)


if __name__ == "__main__":
    main()
