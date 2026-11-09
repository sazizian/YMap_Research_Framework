# Audit of the supplied older archive

This audit covers one older software ZIP identified by SHA-256 `4d7af8e30b2e486a66b66505222acaa17e04b01f45de49e84eba845715f4d0a0`. The findings in the table are scoped to that exact snapshot.

| Observed implementation | Consequence | Revised implementation |
|---|---|---|
| `SyntheticProfiler.profile` computes target values from feature formulas plus seeded noise. The ZIP's included CSV has 864 rows across 11 example models. | Its CSV is a software fixture, not PostgreSQL evidence. | Real profiling is an explicit command; demo outputs are labelled `synthetic_demo` and rejected in research mode. |
| `PostgresProfiler.profile` raises `NotImplementedError`. | Real profiling is not executable in this older file. | Dedicated schema creation, logical data loading, timed repetitions and environment/raw-log recording are implemented. External DB integration must be exercised in the research environment. |
| `evaluate_recommendation` copies true objectives and adds random perturbations. | This is not held-out predictor evaluation. | Fit actual models within leave-one-object-model-out folds and save held-out predictions. |
| `recommend` reads objective labels from the existing CSV and ranks them. | It does not demonstrate prediction for an unseen source model. | Fitted-model inference reads schema features, without a measurement file. |
| `train_model` uses a random candidate-level split. | Variants of one source model can enter both train and test. | Group evaluation by source object model, with separate calibration groups. |
| Default text branches use hash embeddings, and the graph branch is a hash serialization. | The default is not three pretrained encoders plus a trained graph branch. | Explicit structural baselines and a separate optional real neural backend; no silent hash substitution. |
| Alloy adapter launches a JAR but does not establish parsed, solver-backed schema synthesis. | Merely writing `.als` files does not prove that the emitted DDL was synthesized by Alloy. | Explicit solver/backend provenance, retained solver artifacts and bounded guarantee statements. |

## Artifact scope

For the current release, the [manuscript workflow](PAPER_WORKFLOW.md)
specifies the datasets, checkpoints, and comparator records that each
evaluation uses. The [results index](PAPER_REPORTED_VALUES.md) traces
manuscript values to their source tables. Record newly fitted baselines and
profiling runs under their own configuration and provenance.
