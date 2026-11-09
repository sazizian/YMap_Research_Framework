# Evidence and execution status

This file separates released measurements, software checks, and manuscript-reported results. The 728-design native study and the 12,500-file SQL inventory serve different purposes.

## Released material

| Material | Meaning |
| --- | --- |
| `Dataset/YMap/Object_Models/` | 11 JSON object models for the native study. |
| `Dataset/YMap/Generated/candidates.jsonl` and the native subset of `Dataset/YMap/All_SQL_Solutions/` | 728 native candidate schemas produced by the Python backend, with their SQL. The full SQL catalog also contains archive files. |
| `Dataset/YMap/Measurements/ymap_postgresql_measurements.csv` | 728 native measurement rows: one median per design. PostgreSQL 14.24, 100 rows per class, five repetitions per design. |
| `results/YMap/evaluation_review_20261023_113149_551.zip` | Raw native measurements, workload, and environment records. |
| `results/YMap/evaluation_20261023_105939_243/` | Native leave-one-system-out evaluation of structural baselines and a five-member ensemble. |
| `results/YMap/graph_symbolic_cpu_ff2ed12dbc97/` | Native evaluation that adds the graph and structural-descriptor neural model on CPU. |
| `YMap_Run_Audit_20261023/` | Read-only consistency audit of the native evaluation. |
| `results/YMap/paper_reported_values.csv` | Values transcribed from the manuscript; see the [values register](docs/PAPER_REPORTED_VALUES.md) for scope and caveats. |
| `validation/` | Recorded software and DDL checks. |

The separate archive catalog contains 27,488 distinct measured design records in 15 systems, with 31,393 measurement rows. The protocol and measurement units are unconfirmed. See [data and provenance](docs/DATA_AND_PROVENANCE.md).

## Checks completed for the v0.5 workflow

| Check | Observed result |
| --- | --- |
| Native model input validation (`check-data`) | The runner verified Stage B inputs and their alignment with 728 measured designs in 11 systems. This check does not train the neural model. |
| Graph and descriptor CPU smoke run | Five seeds, 20 native designs, three systems. No text encoders were used. |
| Local PostgreSQL inventory (`check-cohort`) | 12,500 SQL files matched the recorded hashes: 728 unchanged native files and 11,772 converted archive files across 26 catalog system IDs. |
| Converted DDL execution | All 11,772 converted files executed in isolated transactions on PostgreSQL 18.6 via PGlite 0.5.15. See the [DDL validation report](validation/ymap-v0.5-postgresql-ddl.json). No workload was run and no latency or storage labels were produced. |

The [five-view runner](src/ymap_research/ymap_paper_main.py) provides input and checkpoint checks, single-fold fitting, prediction, and full cyclic evaluation. The [README](README.md) lists the commands and runtime requirements.

## What remains unverified

- Full five-view training and evaluation on CUDA with GraphCodeBERT, CodeT5+ and Llama 2. The necessary checkpoints are not bundled.
- Reproduction of the manuscript's numerical results or its 12,500 profiled designs across 11 systems. The locally prepared SQL inventory contains 26 catalog system IDs and lacks the archive mappings and measurements under a common protocol.
- Full-space Alloy synthesis for the original experiment. The bundled native study used the Python generator.
- Performance at larger data scales, on other database systems, or through an ORM at application level. The structural checks apply within the supported mapping rules and bounds; they do not prove unconditional semantic preservation.

Link each new experimental claim to the run outputs, configuration, environment, and measured inputs that support it.
