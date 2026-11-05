# Dataset — ReinforceORM Artifacts

Self-contained research data for RQ1–RQ4 reproduction. Files are copied or distilled from
`ReinforceORM_Artifacts_March_2026` (September 2025 / March 2026 artifact release).

**Archival DOI:** https://doi.org/10.6084/m9.figshare.30071260.v8  
**License:** CC BY-NC 4.0 (see package-root `LICENSE.md`).

This `Dataset/` directory is part of the **executable** AE package. The sibling folder
`ReinforceORM_Old_Artifacts_Initial_Submission/` is **read-only transparency storage**
from the initial submission and is **not** required for reproduction.

## Layout

| Directory | RQ | Contents |
|-----------|-----|----------|
| `RQ1_Effectiveness/` | RQ1 | Tradespace counts, GD/HV PKLs/CSVs, GD/HV workbook |
| `RQ2_Performance/` | RQ2 | Time/memory PKL/CSV, ReinforceORM solution JSONs, TradeMaker xlsx |
| `RQ3_LM_Comparison/` | RQ3 | LM metrics PKL/CSVs, metrics workbook + script |
| `RQ4_Ablation_Study/rocket_controller/` | RQ4 | Held-out Alloy model, full tradespace, 2,520 SQL solutions |
| `object_models/` | Shared | 15 training `.als` models + held-out compatibility copy |
| `correlation/` | Setup | Pearson correlation among optimization objectives |

Each RQ folder has its own `README.md` with file descriptions and notes on excluded large artifacts.

## Reproduction wiring

Scripts resolve data in this order:

1. **Bundled `Dataset/`** (this directory — works standalone on Figshare)
2. **External March 2026 path** (optional sibling checkout or `REINFORCEORM_MARCH_ARTIFACTS`)

Frozen PKL files under `reproduction/frozen_states/` are built from these sources via
`reproduction/scripts/build_frozen_states.py`. The same PKLs are mirrored in `Dataset/RQ*/` for direct access.

Notebooks use `frozen_or_dataset_pkl()` from `reinforceorm.paths` to load from either location.

## object_models/

| File | Description |
|------|-------------|
| `manifest.json` | Training system list + held-out design-space summary |
| `Bank.als` … `University-Payroll.als` | 15 training object models |
| `rocket_controller_operations.als` | Compatibility copy of held-out model |

## RQ4_Ablation_Study/rocket_controller/

Canonical held-out Rocket Controller ablation artifacts:

| File | Description |
|------|-------------|
| `rocket_controller_operations.als` | Formal Alloy object model (8 classes, 4 inheritance, 6 associations) |
| `rocket_controller_tradespace_full.csv` | Full tradespace (2,520 solutions) |
| `rocket_controller_pareto_optimal.csv` | Pareto subset |
| `rocket_controller_tradespace.json` | Tradespace metadata + strategy dimensions |
| `source/` | Domain rationale and NASA avionics references |
| `sql_solutions/` | SQL exports for all 2,520 solutions |

## correlation/

Pearson correlation coefficients among query time, insert time, and storage across 31,302 candidate designs.
Source: March 2026 `Corelation_Analysis/Correlation_Results.csv`.

Frozen copy: `reproduction/frozen_states/correlation_analysis.pkl`.

See `object_models/README.md` and `components/reinforceorm/data/object_models.py`.
