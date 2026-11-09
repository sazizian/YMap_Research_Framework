# Y-Map manuscript workflow

Version 0.5 adds a pipeline for the manuscript's method and retains the
released 728-schema PostgreSQL experiment. A workflow implementation, a
software smoke run, and reproduction of the published numbers are distinct
claims. This repository does not contain the full original 12,500-design
profiling run, its original object fixtures, or the original model weights.

## Start from the released data

From the repository root, install the framework and verify each scenario:

```sh
python -m pip install -e .
ymap paper check --dataset native
ymap paper check --dataset archive
```

Both commands use `Dataset/YMap` and `config/datasets.yaml`. Each run chooses one
workload, environment, and measurement source. The archive's measurement units
are unconfirmed, so report any within-archive analysis separately. See
[Data and provenance](DATA_AND_PROVENANCE.md) for counts, source attribution,
and the required manifest for a paper protocol. The raw SQL catalog can include
duplicates and unmeasured schemas; neither contributes extra measured designs.
Use `--dataset-root` to relocate the whole data root.

```sh
ymap paper evaluate --dataset native --out runs/native_evaluation
ymap paper evaluate --dataset archive --out runs/archive_evaluation
python run_graph_symbolic_evaluation.py --check-only
```

The first two commands evaluate the released structural baselines in their
own scenarios, with legacy evaluation settings or an explicit `--config`.
The third validates the released graph/descriptor runner's inputs without
loading Torch. To check the manuscript evaluator on the 728-schema release, use
`scope: smoke` and `views: []` in the native evaluation config. This
exercises system-held-out folds, measured Pareto metrics and preference regret
without training the five-view model. The archived candidate catalog does not supply the complete
canonical object mappings required by the manuscript's graph and 12 descriptors.

To exercise actual neural training on a CPU with PyTorch installed:

```sh
ymap paper evaluate --dataset native --config config/paper_prediction_smoke.yaml --out runs/prediction_cpu_smoke
```

This trains the graph and symbolic branches with five seeds per fold on three
small systems (20 measured candidates). It uses the verified physical tables of
the already-measured schemas without changing their keys, nullability, or SQL.
Read `oof_predictions.csv` for predictions and spread in the measured objective
units and `summary.json` for the run scope. No solver, fresh database
measurements, or pretrained text models are involved in this smoke run.

## Run the five-view predictor on the measured 728 designs

The native study can exercise all five model views separately from the
manuscript reference experiment. Prepare and verify its inputs on any machine:

```sh
ymap paper prepare-native --out runs/native_prediction_inputs
```

The export attaches the typed graph, twelve descriptors, and a canonical-order
text view of the existing constraints to each of the 728 measured IDs. The
original executable SQL and measurements stay unchanged. The export is for
prediction input; it does not constitute a PostgreSQL or Alloy rerun.

On a CUDA machine with the three local checkpoints and the neural extras
installed, copy [the native five-view config](../config/native_five_view.example.yaml)
and fill in the checkpoint paths:

```sh
ymap paper evaluate --dataset native --config path/to/native_five_view.yaml --out runs/native_five_view_evaluation
ymap paper fit --dataset native --config path/to/native_five_view.yaml --validation-system banking --out runs/native_five_view_ensemble
```

Evaluation uses 11 cyclic held-out systems and the manuscript's three
repetitions of five seeds. Fit uses one held-out validation system to produce a
deployable five-member ensemble; `paper predict` can then score prepared inputs.
These are native-study results. They use the October 2026 measurement protocol
and the implemented Llama FP4 loading path; they do not establish agreement
with the manuscript's historical INT4 run or comparator cohort.

## Run a documented manuscript-protocol study

1. Install the runtime extras: `python -m pip install -e ".[postgres,neural,paper-int4]"`.
   Provide a compatible JDK, an Alloy distribution JAR, local GraphCodeBERT,
   CodeT5+ 220M and Llama-2 7B checkpoints (weights and tokenizer files), and a
   CUDA device. The paper does not specify its exact four-bit format; this
   implementation uses bitsandbytes FP4 with FP16 compute for Llama. Checkpoint
   files are not fetched by the workflow.
2. Supply object-model JSONs and a YAML mapping from each system ID to its
   maximum table count. Generate canonical SQL and its typed graph and 12
   descriptors. The scope is a table-count bound enforced during compilation.
3. Copy [the profiler config](../config/paper_profile.example.yaml), fill in
   actual object populations, query bindings, warm-up mix, seeds and environment
   evidence, and point `YMAP_POSTGRES_DSN` at an authorized disposable local
   PostgreSQL server. Reference profiling requires a trusted cache-reset command
   for a dedicated host. Use a separate `mode: smoke` run to check configuration.
4. Place the generated records and measured CSV under a subdirectory of
   `Dataset/YMap`. Copy [the dataset manifest](../config/paper_dataset.example.yaml)
   and record the actual workload, environment, counts, protocol provenance and
   input SHA-256 hashes. Use `raw_repetitions` for five measured rows per schema
   or `protocol_medians` for documented medians. A fresh run is a new experiment.
5. Copy [the study config](../config/paper_study.example.yaml), fill the actual
   cyclic order of 11 system IDs and 9 shared comparison IDs, and set local
   checkpoint paths. Use a text embedding cache across folds. Optionally provide
   recorded paired comparator selections; the evaluator requires matching
   measured candidates and protocol and does not generate selections for other
   methods.

```sh
ymap paper generate --models path/to/models --scopes path/to/table_scopes.yaml \
  --alloy-jar path/to/alloy.jar --out Dataset/YMap/PaperReference/generated
ymap paper profile --candidates Dataset/YMap/PaperReference/generated/candidates.jsonl \
  --config path/to/protocol.yaml --out Dataset/YMap/PaperReference/profiled
ymap paper check --dataset paper-reference --manifest path/to/dataset.yaml
ymap paper evaluate --dataset paper-reference --manifest path/to/dataset.yaml \
  --config path/to/study.yaml --out runs/paper_evaluation
```

Adjust the paths inside the dataset manifest to match the generated files.
The profiler writes one median measurement row per schema by default; its
per-repetition evidence and realized protocol are saved alongside the CSV.
Use `--backend python` to test the bounded compiler without a JDK/Alloy; it
uses the same finite decision space, but does not perform formal solving.
The Alloy bridge encodes finite mapping strategy choices. It does not recover
the paper's original relational translator or all of its constraints. The
four recommendation streams have seeded declaration and choice orders; they
are not guaranteed to use independent SAT solver seeds. SQL deduplication
checks canonical DDL bytes, not full relational equivalence.

Full evaluation fits the five views with cyclic system-held-out validation,
three repetitions of five members, validation-only Gaussian recalibration,
prediction baselines and the 66 manuscript preference vectors. Use
`scope: smoke` for a smaller software check or `scope: ablation` with an
explicit view subset. These scopes have different outputs and are identified
in the reports. Use `ymap paper transfer --help` and
`ymap paper compare-orm --help` to analyze measured workload and ORM comparisons.
The diagnostics consume documented measurements and out-of-fold predictions;
they do not run the original external systems.

## Fit once and recommend on a new model

If the candidate schemas have already been produced by Alloy or another
enumerator, you can score them directly after `paper fit`:

```sh
ymap paper predict --candidates path/to/prepared_candidates.jsonl --model-dir runs/paper_ensemble --out runs/existing_predictions
```

The candidate file must contain the original schema and system identifiers,
canonical `paper_ddl`, `paper_graph` with the five typed relations, and the
twelve `paper_features` when those views are used by the fitted model. Preserve
the mappings from the exact physical schemas. Raw SQL alone does not establish
the inheritance edges or mean join pressure required by the manuscript.
`predictions.csv` contains the individual members, ensemble means, sample
standard deviations, and validation-calibrated spreads in the original target
units. `prediction_manifest.json` records the inputs and model. The command
does not enumerate, measure, train, or estimate calibration from the scoring
candidates. If you set `--batch-size`, use an embedding cache to avoid repeated
checkpoint loads.

```sh
ymap paper fit --manifest path/to/dataset.yaml --config path/to/study.yaml \
  --validation-system SYSTEM_ID --out runs/paper_ensemble
ymap paper recommend --model path/to/new_system.json --model-dir runs/paper_ensemble \
  --max-tables 12 --alloy-jar path/to/alloy.jar --out runs/new_system
```

The fit leaves the selected validation system out of training to estimate
per-objective uncertainty multipliers. Use repeatable `--holdout-system`
options to exclude additional systems. The recommendation interleaves four
bounded streams, only encodes distinct SQL, and stops when its mean Pareto
front stabilizes or its unique-candidate budget is used. It returns the
risk-adjusted Pareto front or, with `--weights`, a single candidate. It writes
predictions, selected DDL and separate enumeration, inference and ranking
timings. Use `--exhaustive` for an unbounded-by-count comparison within the same
finite table scope. To reuse saved weights from moved local checkpoints, use
`--config` to override checkpoint paths, device, or embedding cache.
