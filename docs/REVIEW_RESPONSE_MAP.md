# Evidence map for the seven editorial priorities

| Priority | What the revised code provides | Evidence still needed from actual study |
|---|---|---|
| 1. Objective/workload scope | Scenario identifiers, actual profiling protocol, explicit generated-data mode, separate scenario evaluation | Original workload frequencies/distributions; actual workload-shift runs if broad claims retained |
| 2. Schema versus ORM | Direct PostgreSQL schema measurement clearly labelled | ORM applications if application-level ORM runtime claims are retained |
| 3. Bounded formal claims | Explicit Python/Alloy backend distinction, supported mapping rules, actual Alloy finite-choice checks and bounded PGlite mapping/integrity checks | Original encoding and source-to-mapping validation for the study's target models; native PostgreSQL execution and socket expected-error recovery remain unverified; no universal preservation theorem is supplied |
| 4. Scalability/cost | Streamed candidate enumeration; explicit caps; final-stage finite-pool search; separate manifests | Comparable actual stage timings for all research methods, including original corpus construction and per-new-system processing |
| 5. Preference/uncertainty | Train-only objective normalization; dimensionless standard-deviation penalty; same-ensemble mean/risk comparisons; uncertainty diagnostics | Actual held-out runs and interpretation of calibration/coverage; ensemble disagreement is not automatically calibrated probability |
| 6. Empirical/statistical | Source-group folds; per-system errors and recommendation measures; exact two-sided signed-rank tests, zeros/ties and one Holm family; rank-biserial effect | Original 12,500 measured pairs and original benchmark provenance; checks of the manuscript's particular table computations |
| 7. Reproducibility/exposition | Configurations, manifests, raw profiling logs, saved predictions, dependency checks and English step-by-step instructions | Versioned public anonymous archive, accessible original artifacts, actual study model checkpoints/checksums, manuscript figures and textual revisions |

## Claims this deliverable does not establish

- The original reported numbers are reproduced.
- New profiling matches the original paper hardware/workload controls.
- Python structural checks are formal Alloy proofs.
- Finite strategy-choice solving proves all object-state semantics.
- A capped candidate prefix is a representative sample.
- Small pure-Python tests establish PostgreSQL integration correctness.
- Production pretrained GraphCodeBERT/CodeT5+/LLaMA quality or measured-corpus research results are established by the tiny randomly initialized local checkpoint tests.
- A baseline comparison reproduces other named research methods.

The response letter should describe only corrections and executions actually completed, using actual file/version references. Source code availability does not replace empirical validation.

`VALIDATION_STATUS.json` is the authoritative overall status; `validation/unit_tests.log` records the final suite output. `validation/synthesis_integration.json` records actual Alloy 6.2.0/JDK 17 checks and PGlite 0.5.10/PostgreSQL 18.6 WebAssembly evidence, including 10 positive mapping cases and six direct-engine negative integrity probes. These checks do not establish native PostgreSQL execution or expected-error transaction recovery through the socket wrapper. `validation/neural_integration.json` and `docs/NEURAL_BACKEND.md` document actual neural loading, encoding, training and save/load tests with tiny randomly initialized local RoBERTa, T5 and LLaMA fixtures, not production pretrained-model evaluation or manuscript-result replication.
