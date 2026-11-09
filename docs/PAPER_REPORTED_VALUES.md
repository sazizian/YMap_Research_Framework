# Paper-reported Y-Map values



## Source identity and page numbering

- Source: manuscript version v8 (the PDF is not included in this repository).
- Document title: *Y-Map: Performance-Aware Relational Schema Recommendation under Object-Relational Mapping Constraints*.
- SHA-256: `efadae1dcf94a541cd219810ae31bc313f391bf8a2352cdacd9ee10ea2dcba2e`.
- Source length: 41 physical PDF pages. `source_page` uses the one-based physical PDF page, including the highlights page. `source_printed_page` records the manuscript footer where unambiguous; for the selected pages it is one less than the physical page. Physical page 27 has an overlaid earlier footer as well as the current footer 26.
- Each CSV row retains its source table, method, metric, value, unit, scenario, and verification status. Numeric spellings preserve the displayed precision. A copied PDF with a different hash is a different source and requires a new provenance entry.

## What the selected values describe

| Evidence | Physical page / table | Scope | Selected reported values |
| --- | --- | --- | --- |
| Prediction | 19 / Table 5 | Leave-one-system-out mean over 11 held-out systems under the reference protocol | Y-Map insert/query/storage nRMSE: 0.17 / 0.27 / 0.08; corresponding R-squared: 0.68 / 0.21 / 0.92; Spearman rho: 0.83 / 0.64 / 0.95. XGBoost query nRMSE 0.31 and rho 0.52. |
| Recommendation | 21 / Table 6 | Mean over 9 shared systems; Y-Map risk-neutral ensemble-mean front | HV: Y-Map 0.91, DTS 0.83, TriORM 0.78, Leant 0.61, measured front 0.96. Y-Map HV ratio 0.95, IGD+ 0.018, GD 0.01, F1 0.90. |
| Paired statistic | 25 / Table 8 | Y-Map versus DTS HV; 9 system pairs; 9-test Holm family | Adjusted exact two-sided Wilcoxon p = 0.035. Transcribed, not recalculated. |
| Protocol shift | 25 / Table 9 | Median over 5 sampled systems; reference-trained predictions | HV ratios: Wlookup 0.86, Wjoin 0.81, Dskew 0.89, D10x 0.87. |
| Online cost | 26 / Table 10 | Mean over 9 shared systems; enumeration included; offline model construction excluded | Total wall-clock minutes: Y-Map 12.5, DTS 11.0, TriORM 8.0, Leant 561, exhaustive profiling 2,785. |
| Offline cost | 26 / Table 11 | Corpus profiling versus one-fold training, separately | Profiling: 687 machine-hours; training all 5 members once in one fold: 2.1 GPU-hours. |
| Incremental mode | 27 / Table 12 | Individual large systems | HIS full/incremental: 74.5 / 31.2 minutes; ERP full/incremental: 97.5 / 40.9 minutes. |
| Uncertainty selection | 29 / Table 14 | Ensemble, kappa = 1; 9 shared systems | Pareto F1 0.89. This is a different front from the risk-neutral F1 0.90 condition. |
| Recalibration | 29 / Table 15 | All 11 held-out systems; nominal 95% intervals | Insert/query/storage coverage: 93.1% / 92.4% / 94.0%. |

nRMSE, correlation coefficients, R-squared, normalized front metrics, and p-values are dimensionless. The raw objectives are latency and database storage; their physical units must not be assigned to the normalized errors. HV and HV ratio are distinct columns in the paper. Online minutes, machine-hours, and GPU-hours also describe different accounting boundaries and cannot be added without an explicit cost model.

## Separate corpora and runs

These sources describe different collections. None establishes that the others have the same schema identities, labels, scenarios, or provenance.

| Source | Count and meaning | Relationship to this register |
| --- | --- | --- |
| v8 source PDF, physical page 15, Table 3 | Reports 12,500 unique/profiled candidates across 11 systems. Its listed shared-system unique counts sum to 7,506, with the remaining 4,994 assigned to HIS and ERP. | Count transcribed from manuscript Table 3. |
| Current package profiling/training outputs | Counts and metrics must be read from their own manifests and generated output files. | Fresh results must carry their own run IDs and provenance, and must never be relabeled from this CSV. |

## Source caveats preserved

These notes document how the index represents values and annotations from the specific v8 source identified above.

- The gated single-model F1 differs between Table 13 (0.87, marked) and Table 14 (0.86, marked). That ambiguous single-model F1 is intentionally absent from the CSV; the risk-neutral ensemble and kappa = 1 ensemble conditions are kept distinct.
- Table 15 marks NLL entries for checking against its stated definition. No NLL value is included here.
- Table 11 explicitly describes the all-fold/all-repetition training total as conditional on a per-fold interpretation. This register includes only the paper's one-fold 2.1 GPU-hour entry, with that interpretation noted; it does not assert a measured all-study GPU total or a deployment break-even point.
- Table 16's reported diagnostic round-trip pass rate is described as pre-correction; the text says that this does not establish the post-correction validation outcome. No post-correction success rate is inferred or included.
- Physical page 27 has duplicated/overlaid Table 12 content and figure/footer text. Its selected runtime values agree with the surrounding prose on physical page 26; no new measurements were used to repair the page.
- The statistics discussion immediately before Table 5 asks that a sign convention be checked against the analysis script. The index transcribes the adjusted p-value from Table 8.


