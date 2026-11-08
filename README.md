# Y-Map Research Framework

Version **0.5** adds the manuscript workflow: bounded schema synthesis,
PostgreSQL profiling, five-view prediction, system-held-out evaluation and
incremental risk-aware recommendation. Start with
[the workflow and reproduction limits](docs/PAPER_WORKFLOW.md). All released
data is accessed under `Dataset/YMap` with a named measurement scenario:

```sh
ymap paper check --dataset native
ymap paper check --dataset archive
```

Research software for **Y-Map**: generating relational schema candidates for an object model under ORM mapping rules, measuring them on PostgreSQL, and evaluating learned performance predictors and Pareto/preference-based schema recommendation with leave-one-system-out validation.



## Repository layout

| Path | Contents |
|---|---|
| `src/ymap_research/` | Framework source: candidate generation (`synthesis.py`), schema representations, PostgreSQL profiling, evaluation, recommendation, statistics, optional neural backend |
| `tests/` | Unit and integration tests |
| `config/` | Profiling, evaluation, training and neural configurations |
| `scripts/` | `check_framework.py` (test runner) and `neural_train_recommend.py` |
| `Dataset/YMap/` | Unified SQL and measurement catalog; see the dataset manifest for separate scenarios |
| `results/YMap/` | Outputs of the measured study (see below) |
| `YMap_Run_Audit_20261023/` | Independent consistency audit of the measured study |
| `run_graph_symbolic_evaluation.py` | Runs the graph + symbolic neural evaluation on the existing measurements |
| `docs/` | Profiling protocol, supported mapping semantics, neural backend, technical references |
| `validation/` | Recorded software-validation logs for releases 0.3 and 0.4 |

## Installation

Python 3.10 or newer is required. Run all commands from the project root.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ymap_research doctor
```

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m ymap_research doctor
```

Optional extras: `.[postgres]` for profiling, `.[neural]` for the neural backend, `.[xgboost]`, `.[plots]`.

## Running a study

Set the PostgreSQL connection locally (never commit credentials). `docker-compose.yml` starts an optional local PostgreSQL 14 server.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[postgres]"
$env:YMAP_POSTGRES_DSN = "host=localhost port=55432 dbname=ymap user=ymap password=YOUR_LOCAL_PASSWORD"

.\.venv\Scripts\python.exe -m ymap_research generate --models Dataset/YMap/Object_Models --out runs/candidates --backend python
.\.venv\Scripts\python.exe -m ymap_research profile  --candidates runs/candidates/candidates.jsonl --config config/profile.yaml --out runs/measurements
.\.venv\Scripts\python.exe -m ymap_research validate --candidates runs/candidates/candidates.jsonl --measurements runs/measurements/measurements.csv
.\.venv\Scripts\python.exe -m ymap_research evaluate --candidates runs/candidates/candidates.jsonl --measurements runs/measurements/measurements.csv --config config/evaluation.yaml --out runs/evaluation
```

Use a new output folder for every run. The `python` backend applies the structural checks in [`docs/SUPPORTED_MAPPING_SEMANTICS.md`](docs/SUPPORTED_MAPPING_SEMANTICS.md); `--backend alloy` additionally requires a JDK and an Alloy JAR. The neural backend requires PyTorch, Transformers and local pretrained checkpoints, which are not included (see [`docs/NEURAL_BACKEND.md`](docs/NEURAL_BACKEND.md)).

## Included measured study

| Item | Value |
|---|---|
| Candidates | 728 schemas from 11 object models, generated with the `python` backend |
| Database | PostgreSQL 14.24 (Docker), default server settings, Windows 11 client |
| Workload | `ymap_logical_workload_v1`, 100 rows per class, 5 repetitions per candidate (see [`docs/PROFILING_PROTOCOL.md`](docs/PROFILING_PROTOCOL.md)) |
| Raw measurements | `results/YMap/evaluation_review_20261023_113149_551.zip` (`measurements/`) |
| Evaluation (structural baselines + 5-member ensemble) | `results/YMap/evaluation_20261023_105939_243/` |
| Evaluation adding the graph + symbolic neural model (CPU) | `results/YMap/graph_symbolic_cpu_ff2ed12dbc97/` |

Macro means over the 11 held-out systems (`summary.json`; lower nRMSE/IGD+ and higher HV/F1 are better):

| Model | nRMSE | HV | IGD+ | F1 |
|---|---|---|---|---|
| Ridge | 0.107 | 0.875 | 0.154 | 0.444 |
| Linear | 0.126 | 0.909 | 0.115 | 0.459 |
| Extra trees | 0.143 | 0.898 | 0.130 | 0.463 |
| Ensemble (5 × extra trees), mean | 0.143 | 0.896 | 0.132 | 0.390 |
| Neural (graph + symbolic) | 0.137 | 0.882 | 0.161 | 0.373 |

No paired comparison is significant after Holm adjustment in either evaluation (`paired_statistics.json`). This is a small study and does not establish that the methods are equivalent.

## Tests

### Preference regret

`recommend()` records `preference_index`, chosen from the returned candidates
using predicted weighted mean-plus-risk scores only. Evaluation measures that
specific decision's realized weighted utility against the best feasible measured
utility, using the same proper-training normalization and weights for both.
The selected schema ID and normalization are retained in `recommendations.json`.
Without an explicit prediction-based decision, `preference_regret` is null.

The previous best-measured-member diagnostic is now named `oracle_set_regret`
and retains the fixed measured-pool normalization used by Pareto metrics.
Historical result folders used this oracle diagnostic under the name
`preference_regret`; recompute regret and any affected statistics in a new output
folder before comparing them with corrected runs. This code fix does not
recalculate the included research results or establish the manuscript's results.

### Running the tests

```powershell
.\.venv\Scripts\python.exe scripts/check_framework.py
```

Tests that need PostgreSQL, Alloy or neural checkpoints are skipped when those are unavailable; skipped tests are not passes.

## License

See [`LICENSE.md`](LICENSE.md). Third-party dependencies, PostgreSQL, Alloy and pretrained checkpoints remain under their own licenses.
