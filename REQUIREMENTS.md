# System requirements

## Core package and dataset tools

| Item | Requirement |
|---|---|
| Python | 3.10 or newer |
| Operating system | Windows PowerShell, Linux or macOS; use platform-appropriate environment paths |
| Core Python packages | NumPy, pandas, SciPy, scikit-learn, PyYAML, joblib, OpenPyXL and SQLGlot `>=30.24,<31`; install from `pyproject.toml`. SQLGlot supports physical SQL import. |
| Disk | Imported files occupy approximately 1.34 GB before indexes and new outputs. Allow additional space for the download, extraction, temporary files and experiments. |
| GPU | Not needed for indexing, browsing, historical summaries or structural baselines |

Run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ymap_research doctor
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Network access is needed to install dependencies unless compatible packages are already available locally. Browsing the supplied dataset does not require downloading it again.

## Optional research integrations

| Capability | Additional requirements | Scope |
|---|---|---|
| PostgreSQL profiling | A running research PostgreSQL server and `python -m pip install -e ".[postgres]"` | Actual SQL execution; connection supplied locally through `YMAP_POSTGRES_DSN` |
| Container database | Docker Desktop or Docker Engine with Compose | Optional `docker-compose.yml`; exposed local port is 55432 |
| Alloy generation | Java JDK with both `java` and `javac`, plus a local Alloy distribution JAR | Explicit `--backend alloy --alloy-jar PATH` |
| Neural models | `python -m pip install -e ".[neural]"`, actual local checkpoints/tokenizers and sufficient memory | Full three-text-encoder configuration is separate from graph/symbolic ablations |
| XGBoost | `python -m pip install -e ".[xgboost]"` | Optional additional structural baseline |
| Figures | `python -m pip install -e ".[plots]"` | Matplotlib for `reproduction/build_ymap_figures.py` |
| Release packaging | `python -m pip install -e ".[release]"` | Optional build and Markdown-to-HTML dependencies for release scripts |

Current dependency ranges are in `pyproject.toml`. Large pretrained checkpoints may need substantial RAM/VRAM; the default neural configuration requests CUDA. Choose checkpoint sizes and the device for the actual machine. Dependencies do not include weights, and the loader does not silently replace missing checkpoints.

Earlier integration checks used Alloy 6.2.0/JDK 17, PGlite 0.5.10/PostgreSQL 18.6 WebAssembly and a CPU neural environment. These recorded environments do not establish that all platforms or native PostgreSQL deployments have been validated. See `VALIDATION_STATUS.json` and `validation/` for exact boundaries.

## Preserved upstream tooling

Imported ReinforceORM requirements are preserved in `provenance/Imported_ReinforceORM_Root/REQUIREMENTS.md`, `reproduction/requirements.txt` and `Minimal_Example_CLI_Tool/Imported_ReinforceORM/requirements.txt`. They apply to upstream entry points, not to the lightweight Y-Map dataset overview. Their Docker/notebook/frozen-export workflow is separate from `reproduction/run_ymap.py`.

Use the current core installation first. Install optional integrations only for the operation being run; historical dependencies are unnecessary for browsing the collection.
