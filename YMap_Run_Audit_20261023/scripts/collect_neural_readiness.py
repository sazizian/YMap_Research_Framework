"""Report local prerequisites without installing packages or downloading models."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import io
import json
import shutil
import subprocess
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_machine(project_root: Path) -> dict:
    report = {
        "purpose": "Local readiness only; no model fitting or checkpoint download.",
        "project_root": str(project_root),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "safetensors", "sentencepiece")
        },
        "nvidia_gpus": [],
        "torch_cuda_available": None,
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if result.returncode == 0:
                report["nvidia_gpus"] = list(csv.DictReader(io.StringIO(result.stdout)))
            else:
                report["nvidia_status"] = result.stderr.strip() or result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            report["nvidia_status"] = str(error)
    else:
        report["nvidia_status"] = "nvidia-smi is not on PATH; this alone does not prove that no GPU exists."

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            report["torch_cuda_available"] = bool(torch.cuda.is_available())
            report["torch_cuda_build"] = torch.version.cuda
        except Exception as error:
            report["torch_import_error"] = str(error)

    configuration_path = project_root / "config" / "neural.yaml"
    report["configuration_exists"] = configuration_path.is_file()
    if configuration_path.is_file():
        try:
            import yaml

            configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
            neural = configuration.get("neural", {})
            report["configured_device"] = neural.get("device", "cpu")
            report["configured_views"] = {
                "graph": neural.get("use_graph", True),
                "symbolic": neural.get("use_symbolic", True),
            }
            checkpoints = []
            for encoder in neural.get("text_encoders", []):
                value = encoder.get("model_name_or_path", "")
                directory = Path(value)
                if not directory.is_absolute():
                    directory = project_root / directory
                checkpoints.append({
                    "family": encoder.get("family"),
                    "configured_path": value,
                    "local_directory_exists": directory.is_dir(),
                    "local_files_only": encoder.get("local_files_only", True),
                })
            report["configured_checkpoints"] = checkpoints
            report["checkpoint_note"] = "Directory existence is not checkpoint completeness or compatibility."
        except Exception as error:
            report["configuration_error"] = str(error)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, help="Optional new JSON file; existing files are preserved.")
    arguments = parser.parse_args()
    report = inspect_machine(arguments.project_root.expanduser().resolve())
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.out.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
