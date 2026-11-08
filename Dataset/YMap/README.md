# Y-Map dataset

One dataset with two sources. Every schema, SQL file and measurement keeps a `source` label (`ymap` or `reinforceorm`). The sources were measured with different databases and protocols, so they are evaluated as separate scenarios and never pooled.

| Source | Systems | SQL files | Distinct measured schemas | Measurement rows | Measured by |
|---|---|---|---|---|---|
| `ymap` | 11 (`banking` … `university`) | 728 | 728 | 728 (each the median of 5 repetitions) | This project, PostgreSQL 14.24 |
| `reinforceorm` | 15 (`rorm_bank` … `rorm_university_payroll`) | 31,376 | 27,488 | 31,393 | Hasan and Bagheri (ReinforceORM artifact); protocol and units not documented |
| **Total** | **26** | **32,104** | **28,216** | **32,121** | |

## Layout

| Path | Contents |
|---|---|
| `all_candidates.jsonl` | **Merged input.** One record per distinct measured schema: `schema_id`, `object_model`, `source`, `sql_path`, common DDL-derived `features` |
| `all_measurements.csv` | **Merged input.** One row per measurement: Y-Map loader columns (`insert_ms`, `query_ms`, `storage_mb`, `label_source`, `workload_id`, `environment_id`, `seed`) plus `source`, `units_confirmed` and the original ReinforceORM values |
| `catalog.csv` | Every SQL file with its source, original name, schema it belongs to, number of measurements and status |
| `merged_dataset_summary.json` | Counts and SHA-256 of the inputs used to build the merged files |
| `Object_Models/` | 11 Y-Map JSON models and 15 ReinforceORM Alloy models (`rorm_*.als`) |
| `All_SQL_Solutions/<system>/SQL_Solutions/` | DDL of every schema; ReinforceORM systems are prefixed `rorm_` |
| `Generated/` | Y-Map generator output (`candidates.jsonl`, `generation_manifest.json`), unchanged |
| `Measurements/ymap_postgresql_measurements.csv` | Y-Map raw measurements, byte-identical to the input recorded for the October 2026 runs |
| `Measurements/rorm_supplied_measurements.csv` | ReinforceORM `All_Designs_Raw_Data.csv`, byte-identical to the artifact |
| `LICENSE_ReinforceORM_CC_BY_NC_4.0.md` | License of all `rorm_*` material |

The merged files are rebuilt from the raw files with `python scripts/build_merged_dataset.py`.

## How the ReinforceORM records were merged

- A ReinforceORM **schema is one distinct DDL body** within a system. SQL files with identical DDL (after removing the `CREATE DATABASE FOR …`/`USE …` header, comments and whitespace) are copies; `catalog.csv` marks them `duplicate_ddl_of_<file>`.
- Each supplied record names a database. It is matched to SQL files by the file's own `USE` identifier, or by file name when the identifier differs (Flagship files say `USE flagship_re_Sol_N`). All 31,393 records matched exactly one schema.
- All records that map to the same schema are kept as **repeated measurements** (`seed` 1…k); the evaluation averages them. Original values are unchanged.
- Three SQL files have no measurement (`sql_without_measurement`).

## Common features

Both sources use the same features, counted from the DDL text: `table_count`, `column_count`, `foreign_key_count`, `pk_column_count`, `nullable_column_count`, `unique_count`, `check_count`, `index_count`, `trigger_count`, `link_table_count`, `max_columns_per_table`. On the 728 Y-Map schemas these counts equal the generator's own values exactly; `link_table_count` is a heuristic (tables with at least two foreign keys whose primary-key columns are all foreign-key columns). Y-Map records also keep the generator's `generator_features`, which are not used as model inputs here.

## Using the dataset

The evaluation code requires exactly one scenario and refuses a pooled run.

```powershell
# Y-Map scenario
.\.venv\Scripts\python.exe -m ymap_research evaluate --candidates Dataset/YMap/all_candidates.jsonl --measurements Dataset/YMap/all_measurements.csv --config config/evaluation_ymap_scenario.yaml --out runs/eval_ymap

# ReinforceORM scenario
.\.venv\Scripts\python.exe -m ymap_research evaluate --candidates Dataset/YMap/all_candidates.jsonl --measurements Dataset/YMap/all_measurements.csv --config config/evaluation_reinforceorm_scenario.yaml --out runs/eval_rorm
```

The two config files equal `config/evaluation.yaml` plus the `workload_id`/`environment_id` of their scenario. Use a new `--out` folder for every run.

## Caveats

- ReinforceORM values are stored in the `insert_ms`, `query_ms` and `storage_mb` columns because the loader requires those names, but their **units are not confirmed** (`units_confirmed = False`; storage values range from 6,518 to 112,783). Results are compared within a scenario, where units cancel after normalization; values must not be compared across sources.
- The ReinforceORM paper reports 31,302 candidate designs; the supplied files contain 31,376 SQL files, 31,393 measurement records and 27,488 distinct measured schemas.
- The ReinforceORM held-out Rocket Controller data are not included: in the artifact they are generated from formulas with random noise, not measured.
- The two sources use different object models; same-named systems (for example `library` and `rorm_library`) are different models.

## Attribution

All `rorm_*` files and `Measurements/rorm_supplied_measurements.csv` come from:

- Md Rashedul Hasan and Hamid Bagheri. *Formal Verification-Integrated Reinforcement Learning for Automated ORM Schema Synthesis.* ASE 2026. https://doi.org/10.1145/3832783.3837501
- Artifact: https://doi.org/10.6084/m9.figshare.30071260

They are licensed under CC BY-NC 4.0. Changes made here: files reorganized and renamed (`rorm_` prefix), duplicate DDL identified, features derived; no measurement value was changed. Any publication using them must cite the paper and the artifact and state that these measurements come from ReinforceORM, not from Y-Map profiling.
