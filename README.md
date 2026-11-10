# Y-Map Research Framework

**Y-Map** explores relational schemas that implement the same object model under object–relational mapping constraints. It can generate candidate schemas, profile them in PostgreSQL, predict insert latency, query latency, and storage, and recommend schemas using Pareto ranking and user preferences.

This guide covers the version 0.5 release and its five-view runner. It provides entry points for inspecting the released measurements, checking model inputs, training predictors, and recommending schemas.

## Manuscript reference study

The manuscript reports an evaluation of 12,500 profiled relational designs across 11 object models under a reference PostgreSQL 14.10 protocol with five profiling repetitions per design. Use the `paper-reference` workflow with the experiment's matching candidate records, measured objectives, and study configuration. The [manuscript results index](docs/PAPER_REPORTED_VALUES.md) identifies the source table and evaluation scope for each listed result. The [run status](STATUS.md) indexes the released measurements and software checks by run.

The checked-in `paper-reference` configuration is a template. Supply a manifest that ties the reference study's candidate records and measured objectives to its profiling protocol. The prepared local SQL cohort is documented below, alongside the released `native` and `archive` datasets.

## Choose the right inputs

| Input | Coverage | Intended use |
| --- | ---: | --- |
| `paper-reference` | Manuscript reference study, 12,500 profiled designs across 11 object models | Select the study with a [dataset manifest](config/paper_dataset.example.yaml) and [evaluation configuration](config/paper_study.example.yaml). |
| `native` | 728 measured schemas across 11 systems | Start here to run the released PostgreSQL 14.24 study workflow. Each schema has one measurement row summarizing five repetitions at 100 rows per class. |
| `archive` | 27,488 distinct measured schema records across 15 systems, with 31,393 measurement rows | Use for separate analyses of the ReinforceORM archive. Interpret objectives in the context of the source records, whose measurement units and protocol are unconfirmed. |
| Local PostgreSQL SQL inventory (`runs/ymap_cohort_12500_postgresql_ready/`) | 12,500 DDL files across 26 catalog system IDs, comprising 728 native and 11,772 converted archive files | Inspect the candidate inventory and verify its manifest and SQL file hashes. |

The default data root is `Dataset/YMap/`; [the dataset registry](config/datasets.yaml) selects a scenario. Native inputs are `Generated/candidates.jsonl` and `Measurements/ymap_postgresql_measurements.csv` within that root. Use `--dataset-root` if the data are stored elsewhere. See [data and provenance](docs/DATA_AND_PROVENANCE.md) for source attribution, licenses, and the distinction between SQL files and measured schemas.

## Install and verify on Windows

Use Python 3.10 or newer and run these commands from the version 0.5 repository root. Skip virtual environment creation if `.venv` already exists.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ymap_research doctor
.\.venv\Scripts\python.exe -m ymap_research paper check --dataset native
```

On Linux or macOS, use `python3 -m venv .venv`, activate with `source .venv/bin/activate`, and use `python` in place of the PowerShell interpreter path.

The package check verifies the registered native measurements and their candidate IDs. Use the [five-view runner](src/ymap_research/ymap_paper_main.py) to also validate the native model inputs:

```powershell
.\.venv\Scripts\python.exe .\src\ymap_research\ymap_paper_main.py check-data --dataset native --config .\config\native_five_view.example.yaml
```

Expect `measured_stage_b_inputs_verified` and observed counts of **728 candidates, 11 systems, and 728 measurement rows**. The example configuration can be used as-is for this check. `neural_execution_tested: false` is expected: this command does not load or train the model.

### Verify the prepared local SQL inventory

The prepared development copy of the 12,500-candidate cohort is at `runs/ymap_cohort_12500_postgresql_ready/`, relative to the repository root. The folder contains the SQL files, `candidates.csv`, `postgresql_bundle_manifest.json`, and source and conversion records under `provenance/`. The `sql_path` entries in `candidates.csv` are relative to the cohort folder.

From the repository root, verify the inventory with `check-cohort`.

```powershell
.\.venv\Scripts\python.exe .\src\ymap_research\ymap_paper_main.py check-cohort .\runs\ymap_cohort_12500_postgresql_ready
```

The recorded local check returned `verified_sql_files: 12500`, `catalog_systems: 26`, and `can_use_inventory_directly_for_fit_or_evaluate: false`. This is a file and hash check that runs without a GPU or a PostgreSQL server. Fitting or evaluating the manuscript's reference study requires matching Stage B candidate records, measured objectives under the study protocol, and the study manifest.

### Prepare the local SQL inventory

To prepare the inventory in another checkout, extract the original `ymap-cohort-12500-with-sql.zip` bundle under `runs/ymap_cohort_12500_sql/`. Keep the SQL, CSV, manifests, and `provenance/` together. Run these commands from the repository root in an activated Python environment.

```bash
python scripts/count_ymap_sql.py runs/ymap_cohort_12500_sql
python scripts/convert_ymap_archive_to_postgresql.py runs/ymap_cohort_12500_sql runs/ymap_cohort_12500_postgresql
python scripts/prepare_ymap_postgresql_cohort.py runs/ymap_cohort_12500_sql runs/ymap_cohort_12500_postgresql runs/ymap_cohort_12500_postgresql_ready
```

The converter targets the archive's DDL forms. The preparation script preserves the native files and checks that every converted file matches its conversion report. The SQL bundle and generated `runs/` folders are kept outside Git. To rebuild the candidate **list** from the pinned source catalog, use `python scripts/build_ymap_cohort.py --dataset-root Dataset/YMap --out-dir runs/ymap_cohort_12500` with the source SQL files present. That command produces selection metadata, not the SQL bundle. The [status report](STATUS.md) records the checks and their limits.

## Train and evaluate the five-view predictor

The model combines GraphCodeBERT, CodeT5+ 220M, Llama 2 7B, a typed schema graph, and 12 structural descriptors. The five views are fused through a learned gate. The model predicts three objectives and uses a five-member ensemble to estimate uncertainty, with calibration fitted on the validation system.

On a CUDA machine, install the `neural` extra with a CUDA-enabled PyTorch build. The standalone runner defaults to `gptq-int4` and requires compatible `accelerate`, `optimum`, and `gptqmodel` installations and a local, prequantized GPTQ INT4 Llama 2 7B checkpoint. Use the dependency ranges in [`pyproject.toml`](pyproject.toml) and provide local checkpoint paths. The explicit `--llama-backend fp4` option selects a different quantization format. Record the quantization backend and checkpoint identities alongside each evaluation.

Copy [the example configuration](config/native_five_view.example.yaml) to `runs/native_five_view.yaml` and set `neural.checkpoints.graphcodebert`, `neural.checkpoints.codet5`, and `neural.checkpoints.llama` to the three local checkpoint directories. Keep the embedding cache enabled across runs. From an activated Python environment on the CUDA machine, first check the setup, then fit one fold:

```bash
python src/ymap_research/ymap_paper_main.py check-model --config runs/native_five_view.yaml
python src/ymap_research/ymap_paper_main.py fit --dataset native --config runs/native_five_view.yaml --test-system banking --out runs/native_paper_fit
```

The example holds out `banking`, uses `blog_platform` for validation and calibration, and trains on the other nine systems. It saves the five fitted members and `fit_manifest.json`. A successful `check-model` checks configuration and runtime availability; the fit exercises checkpoint loading and training.

To score prepared inputs with the fitted ensemble:

```bash
python -m ymap_research paper prepare-native --out runs/native_prediction_inputs
python src/ymap_research/ymap_paper_main.py predict --candidates runs/native_prediction_inputs/candidates.jsonl --model-dir runs/native_paper_fit --out runs/native_paper_predictions
```

The output `predictions.csv` contains member predictions, ensemble means, and raw and calibrated spreads. This example scores the entire native set, including training systems. Use the following command for **held-out performance**: it runs 11 cyclic folds, with three repetitions of five members per fold, and writes `oof_predictions.csv` plus prediction and uncertainty reports.

```bash
python src/ymap_research/ymap_paper_main.py evaluate --dataset native --config runs/native_five_view.yaml --out runs/native_five_view_evaluation
```

Use a new or empty output directory for each run. Scoring external candidates requires Stage B JSONL with matching schema and system IDs, canonical DDL, the typed graph, and all 12 descriptors. The SQL inventory CSV cannot replace that JSONL. For training or evaluation, the candidate file must also align with a measured objectives CSV. See the [reference manifest example](config/paper_dataset.example.yaml) and [study configuration](config/paper_study.example.yaml) for the `paper-reference` inputs.

## Other workflows and results

- **CPU smoke test:** with `.[neural]` installed, run `python -m ymap_research paper evaluate --dataset native --config config/paper_prediction_smoke.yaml --out runs/prediction_cpu_smoke`. This exercises graph and descriptor training on 20 measured designs from three systems. It does not exercise the text encoders.
- **Generation, profiling, and recommendation:** see the [manuscript workflow](docs/PAPER_WORKFLOW.md), [mapping semantics](docs/SUPPORTED_MAPPING_SEMANTICS.md), and [profiling protocol](docs/PROFILING_PROTOCOL.md). Actual database profiling requires the `postgres` extra and an authorized PostgreSQL instance. The Alloy backend additionally requires a JDK and an Alloy JAR. The package CLI has its own model configuration and fit flags; use the standalone runner to score ensembles it fitted.
- **Recorded results.** See the [run status](STATUS.md) for native structural baselines, graph-plus-descriptor CPU results, and the scope of each software check. See the [manuscript results index](docs/PAPER_REPORTED_VALUES.md) for the source and scope of reference-study values.
- **Preference regret:** in corrected evaluations, regret compares the recommendation selected using predictions with the best feasible measured choice. Older result folders used the best measured member of the returned set under the old `preference_regret` label; that diagnostic is now `oracle_set_regret`. Recompute affected outputs before comparing regret values.

## Tests and license

Run `python scripts/check_framework.py` from the project root using the installed environment. Integration tests requiring unavailable services or checkpoints can be skipped; the test summary records skips separately.

See [`LICENSE.md`](LICENSE.md) for the framework and [data and provenance](docs/DATA_AND_PROVENANCE.md) for third-party data terms. Dependencies and pretrained checkpoints have their own licenses.
