# System requirements

## Core package

| Item | Requirement |
|---|---|
| Python | 3.10 or newer |
| Operating system | Windows PowerShell, Linux or macOS |
| Core Python packages | NumPy, pandas, SciPy, scikit-learn, PyYAML, joblib, OpenPyXL and SQLGlot `>=30.24,<31`; install from `pyproject.toml` |
| Disk | About 150 MB for the repository, plus space for new run outputs |
| GPU | Not needed for candidate generation, profiling or structural baselines |

Run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ymap_research doctor
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Optional integrations

| Capability | Additional requirements | Scope |
|---|---|---|
| PostgreSQL profiling | A running research PostgreSQL server and `python -m pip install -e ".[postgres]"` | Actual SQL execution; connection supplied locally through `YMAP_POSTGRES_DSN` |
| Container database | Docker Desktop or Docker Engine with Compose | Optional `docker-compose.yml`; exposed local port is 55432 |
| Alloy generation | Java JDK with `java` and `javac`, plus a local Alloy distribution JAR | Explicit `--backend alloy --alloy-jar PATH` |
| Neural models | `python -m pip install -e ".[neural]"`, local checkpoints/tokenizers and sufficient memory | The full three-text-encoder configuration is separate from the graph + symbolic model |
| XGBoost | `python -m pip install -e ".[xgboost]"` | Optional additional structural baseline |
| Figures | `python -m pip install -e ".[plots]"` | Matplotlib |

Dependency ranges are in `pyproject.toml`. Large pretrained checkpoints may need substantial RAM/VRAM; the default neural configuration requests CUDA. Weights are not included, and the loader does not silently replace missing checkpoints.

Earlier integration checks used Alloy 6.2.0/JDK 17, PGlite 0.5.10/PostgreSQL 18.6 WebAssembly and a CPU neural environment. See `VALIDATION_STATUS.json` and `validation/` for their exact scope.
