# Evidence and execution status

This file separates what the repository contains, what has been executed, and what it does not establish.

| Material | Meaning |
|---|---|
| `Dataset/YMap/Object_Models/` | 11 JSON object models used by the current implementation |
| `Dataset/YMap/Generated/`, `Dataset/YMap/All_SQL_Solutions/` | 728 candidate schemas produced by the `python` generator backend, with their SQL files |
| `results/YMap/evaluation_review_20261023_113149_551.zip` | Raw PostgreSQL measurements for the 728 candidates (5 repetitions each), environment and workload records |
| `results/YMap/evaluation_20261023_105939_243/` | Leave-one-system-out evaluation of structural baselines and a 5-member ensemble on those measurements |
| `results/YMap/graph_symbolic_cpu_ff2ed12dbc97/` | The same evaluation with the graph + symbolic neural model added (CPU) |
| `YMap_Run_Audit_20261023/` | Read-only consistency audit of the evaluation (no changes to inputs or labels) |
| `results/YMap/paper_reported_values.csv` | Numbers transcribed from a manuscript draft; not produced by this code |
| `validation/` | Recorded software checks for releases 0.3 and 0.4 |

## Not established by this repository

- The numerical results of earlier manuscript drafts, including the 12,500-candidate corpus.
- Results of the full three-text-encoder neural model (GraphCodeBERT, CodeT5+, LLaMA); checkpoints are not included.
- Alloy-solver synthesis of the full candidate space; the included candidates come from the `python` backend.
- Performance at larger data scales, on other database systems, or through an ORM framework at application level.
- An unconditional semantic-preservation guarantee; the structural checks are bounded.

New experimental claims must be linked to their own run folder, configuration, environment and measurements.
