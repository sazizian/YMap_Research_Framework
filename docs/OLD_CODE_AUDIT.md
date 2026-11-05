# Audit of the supplied older archive

This audit concerns the exact uploaded older ZIP. It does not determine whether a different implementation produced the manuscript's reported measurements. The older archive is not included in this repository; its SHA-256 was `4d7af8e30b2e486a66b66505222acaa17e04b01f45de49e84eba845715f4d0a0`.

| Observed implementation | Consequence | Revised implementation |
|---|---|---|
| `SyntheticProfiler.profile` computes target values from feature formulas plus seeded noise. | Its CSV is a software fixture, not PostgreSQL evidence. | Real profiling is an explicit command; demo outputs are labelled `synthetic_demo` and rejected in research mode. |
| The included CSV has 864 rows across 11 example models, not the manuscript's 12,500 pairs. Several subject names and structural sizes differ. | The ZIP alone cannot reproduce the manuscript corpus or establish its provenance. | Preserve original measurements separately; import only with schema IDs, measurement sources and scenario metadata. |
| `PostgresProfiler.profile` raises `NotImplementedError`. | Real profiling is not executable in this older file. | Dedicated schema creation, logical data loading, timed repetitions and environment/raw-log recording are implemented. External DB integration must be exercised in the research environment. |
| `evaluate_recommendation` copies true objectives and adds random perturbations. | This is not held-out predictor evaluation. | Fit actual models within leave-one-object-model-out folds and save held-out predictions. |
| `recommend` reads objective labels from the existing CSV and ranks them. | It does not demonstrate prediction for an unseen source model. | Fitted-model inference reads schema features, without a measurement file. |
| `train_model` uses a random candidate-level split. | Variants of one source model can enter both train and test. | Group evaluation by source object model, with separate calibration groups. |
| Default text branches use hash embeddings, and the graph branch is a hash serialization. | The default is not three pretrained encoders plus a trained graph branch. | Explicit structural baselines and a separate optional real neural backend; no silent hash substitution. |
| Alloy adapter launches a JAR but does not establish parsed, solver-backed schema synthesis. | Merely writing `.als` files does not prove that the emitted DDL was synthesized by Alloy. | Explicit solver/backend provenance, retained solver artifacts and bounded guarantee statements. |

## Boundaries for a journal revision

The new source code is a new implementation. Its existence does not retroactively verify published numbers. Any new empirical run must be reported with its actual configuration and outcomes. Original paper numbers must be traced to their original code, raw measurements and scripts. Reanalysis can use existing records; fitting new baselines or re-profiling is a new experiment.

This package does not bundle trained GraphCodeBERT/CodeT5+/LLaMA checkpoints, the original raw PostgreSQL corpus, framework-default ORM application implementations, or reproductions of TradeMaker, Leant, DTS or TriORM. Those methods must not be represented by the supplied generic structural baselines.
