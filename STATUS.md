# Evidence and execution status

This document distinguishes preserved records, newly generated files, executed software checks and research results.

| Material or operation | Meaning |
|---|---|
| Supplied ReinforceORM collection | Preserved study artifacts and historical records; import mapping in `provenance/imported_archive_manifest.json` |
| Earlier Y-Map archive | Preserved under `legacy/` with provenance |
| Current Y-Map models and 728 generated candidates | Actual inputs and generator outputs, not latency/storage measurements |
| Dataset index, verification and overview | Cataloging and consistency checks; not model training or database benchmarking |
| Historical-data summaries | Calculations from supplied records, retaining source units and identity; not new database runs |
| Paper-reported values | Transcribed from the paper, kept separate from executed analysis |
| A new profiled and evaluated study | Requires actual measurements, environment/configuration, fitted models, predictions and execution records |

## Authoritative status records

- `VALIDATION_STATUS.json`: overall recorded software validation and limits.
- `validation/v0_4/unit_tests.log`: current combined-package test output.
- `validation/v0_4/sql_corpus_validation.json`: validation of the current generated SQL corpus and export path.
- `validation/v0_4/source_preservation.json`: preservation checks against supplied archive records.
- `validation/v0_4/installed_distribution.json`: installed-package verification.
- `validation/v0_3/unit_tests.log`: preserved earlier test-suite output; skipped integration tests are not passes.
- `validation/v0_3/synthesis_integration.json`: preserved solver and SQL integration details.
- `validation/v0_3/postgresql_static_parse.json`: preserved parsing evidence, distinct from execution.
- `validation/v0_3/neural_integration.json`: preserved actual neural integration checks and environment.
- `provenance/imported_archive_manifest.json`: archive identity and per-file import mapping.

Earlier checks executed actual Alloy 6.2.0 finite-choice solving under JDK 17. SQL checks used PGlite 0.5.10/PostgreSQL 18.6 in WebAssembly, covering 10 positive mapping cases and six direct-engine negative integrity probes. This is not native PostgreSQL server validation; expected-error transaction recovery through the socket wrapper remains unverified in that evidence record.

Actual neural loading, encoding, training, saving and loading were tested using tiny randomly initialized local RoBERTa, T5 and LLaMA checkpoints with real tokenizers. These checks establish bounded software behavior. Production pretrained-model quality and full measured-corpus reconstruction of the paper were not evaluated by those fixtures.

The exact latest results and current log locations belong in `VALIDATION_STATUS.json`. The `validation/v0_3/` files retain their original validation scope rather than certifying later changes. A successful overview must not be interpreted as a blanket statement that all paper results were reproduced.

All 728 current canonical SQL candidates passed PostgreSQL parsing with pglast and were exported through the canonical adapter with a common physical feature projection. Exported records are a physical SQL view; use `Dataset/YMap/Generated/candidates.jsonl` for profiling that requires the generator's logical mapping metadata. Parsing and export checks do not execute a new database performance experiment.

The combined catalog's 80,248 SQL occurrences and 38,606 distinct byte sequences include imported files, repeated paths and earlier archives. They are separate from the 728 current Y-Map candidates. Plots under `figures/YMap/` identify their sources; `paper_reported_hypervolume` copies manuscript Table 6 values rather than reporting a new benchmark.

## Imported records and attribution

The ReinforceORM paper, result tables, frozen exports and original code retain their ReinforceORM attribution. Some upstream scripts reconstruct tables from constants or generate objectives by formulas; preserve their provenance. Their availability is not proof of a new Y-Map measurement run. This classification describes the supplied source code and does not change the authorship or reported status of the original study.

The original ReinforceORM `STATUS.md` is preserved under `provenance/Imported_ReinforceORM_Root/`. It describes artifact badge applications. This combined package does not claim an awarded badge based on that application.

## Claims requiring additional study evidence

Software checks alone do not establish the submitted paper's numerical results, cross-workload robustness, application-level ORM overhead, production-hardware speedups, an unconditional semantic-preservation theorem or independent reproduction of other research methods. Experimental evidence must be linked to the corresponding version, schemas, workload, units, environment, models and recorded predictions.
