# Historical ReinforceORM data reanalysis

This report analyzes supplied historical records. It does not report new Y-Map model or database experiments.

Source: `ReinforceORM_Old_Artifacts_Initial_Submission/Corelation_Analysis/All_Designs_Raw_Data.csv`
Source SHA-256: `e677dc1113f655e822c64b156c529d02af7890f440c8d3d268fa13123b477e28`

The file contains 31,393 raw records across 15 systems and 31,313 distinct system/solution ID pairs. There are 68 repeated-ID groups, 68 with conflicting objective vectors. All source rows and original cost strings were preserved. Repeated records were not averaged or declared independent schemas.

Query, insert and storage units and the historical profiling protocol remain unconfirmed. SQL joins do not verify measurement provenance. Original SQL is MySQL-style; it was read, never executed or converted.

## Within-system nondominance

All three supplied raw costs are minimized. Equal objective vectors remain nondominated together. Counts refer to records; an ID with conflicting observations may have both dominated and nondominated records. No normalized hypervolume is reported because common reference-point and normalization semantics have not been established.

| System | Records | Distinct IDs | Nondominated records | Conflicting IDs |
| --- | ---: | ---: | ---: | ---: |
| Bank | 62 | 30 | 7 | 21 |
| CSOS | 2320 | 2320 | 13 | 0 |
| Camping | 44 | 29 | 7 | 14 |
| Canteen | 2688 | 2688 | 11 | 0 |
| Customer - Order | 9 | 8 | 2 | 1 |
| Decider | 80 | 80 | 9 | 0 |
| E-Commerce | 8144 | 8144 | 20 | 0 |
| Flagship | 256 | 256 | 8 | 0 |
| Hospital | 9600 | 9600 | 16 | 0 |
| Library | 2688 | 2688 | 8 | 0 |
| Onlinestore | 8 | 8 | 6 | 0 |
| Order-Processing | 56 | 56 | 7 | 0 |
| Store | 2686 | 2686 | 20 | 0 |
| Traffic-Control | 2688 | 2688 | 15 | 0 |
| University-Payroll | 64 | 32 | 10 | 32 |

## SQL identity audit

Verified original USE identifiers for 31,057 distinct system/solution pairs (31,137 raw records). Exact filenames were preferred; case-only matching and the explicit Hospital filename alias were audited against internal USE identifiers. See `schema_join_audit.csv` for unresolved or conflicting matches.

## Rocket Controller data

```json
{
  "status": "analyzed",
  "source": "Dataset/RQ4_Ablation_Study/rocket_controller/rocket_controller_tradespace_full.csv",
  "source_sha256": "5aeccf3f5820e2e04a66fd46ccbe5d2e99f77875b485ded2d3e07d52f7c77b9b",
  "row_count": 2520,
  "supplied_marked_pareto_count": 13,
  "recomputed_nondominated_count": 337,
  "marked_and_nondominated_count": 11,
  "marked_but_dominated_ids": [
    "RCO-0952",
    "RCO-0975"
  ],
  "unmarked_nondominated_count": 326,
  "generator_script": "reproduction/build_rocket_controller_artifact.py",
  "matches_recognized_bundled_generator_formula": true,
  "label_provenance": "formula_generated_objectives_not_database_measurements",
  "flags_policy": "Preserved original flags; separately recomputed actual nondominance. No supplied values or files were changed."
}
```

The original CSV flags and objective values were retained. Recomputed nondominance is a separate derived result. Formula-generated objectives cannot be used as real database measurements.

## Reported outputs and attribution

```json
{
  "status": "compared",
  "source": "results/rq1_tradespace_effectiveness.csv",
  "reported_total_solutions": 31302,
  "raw_records": 31393,
  "systems_with_record_count_differences": [
    {
      "subject_system": "Library",
      "reported_records": 2600,
      "raw_records": 2688,
      "raw_minus_reported": 88
    },
    {
      "subject_system": "Traffic-Control",
      "reported_records": 2685,
      "raw_records": 2688,
      "raw_minus_reported": 3
    }
  ],
  "interpretation": "These supplied files describe different counts; version/protocol reconciliation is unresolved. No source totals were changed."
}
```

Original `results/` CSVs, workbooks and frozen-state reproduction outputs are reported historical summaries. They are distinct from the raw-record reanalysis above. The analysis does not unpickle frozen states or rerun imported scripts.

Benchmark: ReinforceORM ASE 2026 artifact; DOI: 10.6084/m9.figshare.30071260.v8. License: CC BY-NC 4.0. Preserve the original attribution and license. These are not newly measured Y-Map results.
