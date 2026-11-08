# Dataset scenarios and provenance

`Dataset/YMap/` is the data root. [`config/datasets.yaml`](../config/datasets.yaml)
selects inputs under that root. Keeping the files in one location makes them
easier to find; their measurements still belong to separate workload and
environment scenarios.

| Dataset | Candidate and measurement inputs relative to the root | Measured schemas | Systems | Measurement rows |
| --- | --- | ---: | ---: | ---: |
| `native` (default) | `Generated/candidates.jsonl`, `Measurements/ymap_postgresql_measurements.csv` | 728 | 11 | 728 |
| `archive` (explicit selection) | `all_candidates.jsonl`, `all_measurements.csv`, filtered to the archival workload and environment | 27,488 | 15 | 31,393 |
| `paper-reference` (explicit manifest) | Original or newly profiled candidate and measurement files, with a documented reference protocol | Declared and checked against supplied data | Declared and checked | Declared and checked |

The default inputs retain the native graph, symbolic structure and DDL. Native
measurements were collected on PostgreSQL 14.24, with 100 rows per class and
five timed repetitions summarized by the median for each candidate. See the
[profiling protocol](PROFILING_PROTOCOL.md). This is the repository's October
2026 study, distinct from the manuscript's reference experiment. The dataset
manifest records both the repository hashes (LF line endings) and the October
run archive hashes (CRLF). The candidate and measurement files are otherwise
byte-for-byte identical.

The archive's storage and timing units and its profiling protocol are
unconfirmed. Analysis must remain within that scenario. Its values are stored
under the loader's existing objective column names; those names do not verify
physical units. The merged files are a catalog. Selecting `archive` filters
measurements by the exact workload, environment and label source from the
manifest before joining measured outcomes to candidate IDs. The archive
contains the common DDL-derived features and does not supply the complete
native graph, symbolic inputs or profiling fixtures for each schema.
Rebuilding the merged files from the archived SQL and measurements matched
the committed inputs after normalizing CSV line endings; the manifest
records both sets of hashes.

The repository catalog records 32,104 SQL files, including 728 native SQL
files and 31,376 archive SQL files. The archive groups copies with identical
normalized DDL text within each source system under one schema ID; multiple records
per ID remain measurement rows. Across both scenarios the catalog records
28,216 distinct measured schema records and 32,121 measurement rows. These
counts differ from 31,302 designs reported by the cited archive paper, and
from the manuscript's reported reference corpus. The manuscript also lists
80,248 SQL entries for the wider artifact package, including duplicates and
files inside ZIPs. That inventory does not count distinct or measured schemas. See the
[dataset inventory](../Dataset/YMap/README.md) and the
[manuscript value register](PAPER_REPORTED_VALUES.md).

## Check the inputs

Install the base package, then run from the project root:

```sh
ymap paper check --dataset native
ymap paper check --dataset archive
ymap paper check --dataset paper-reference --manifest /path/to/reference.yaml
```

Each check validates candidate alignment and counts for one scenario. To use
a copy of the data root, add `--dataset-root /path/to/YMap`. To save the full
provenance and observed input hashes, add `--out /path/to/dataset_check.json`.

To also check the native graph and symbolic representations without PyTorch
or a database, run `python run_graph_symbolic_evaluation.py --check-only`.
To fit the exploratory graph diagnostic, install `.[neural]` and omit
`--check-only`. Its run plan records actual input SHA-256 values, the
selected scenario, and the scope.

## Supply measured inputs for the manuscript workflow

Use `paper-reference` for verified original inputs or a new measured study
with documented protocol and units. A fresh study has its own provenance.
There is no bundled default or fallback. Fill in the
[example manifest](../config/paper_dataset.example.yaml) with your measured
counts, exact workload and environment identifiers, source citation, protocol
description, and each input's SHA-256. The JSONL candidate file and CSV
measurement file use the same alignment fields as native files. Input paths
in the manifest must be relative to the selected dataset root and resolve
inside it.

The resolver verifies these hashes and requires a measured label source.
File identity and internally consistent counts establish which files the run
consumed. Claiming reproduction of manuscript values additionally requires
verifying the original experimental protocol, splits, methods, and outputs.

## Archive attribution

The archived inputs are from the ReinforceORM artifact. Cite Md Rashedul
Hasan and Hamid Bagheri, *Formal Verification-Integrated Reinforcement Learning
for Automated ORM Schema Synthesis*, ASE 2026
([paper](https://doi.org/10.1145/3832783.3837501),
[artifact](https://doi.org/10.6084/m9.figshare.30071260)). The source files
remain under [CC BY-NC 4.0](../Dataset/YMap/LICENSE_ReinforceORM_CC_BY_NC_4.0.md).
The repository's existing importer reorganized and renamed files, grouped
duplicate DDL, and extracted structural features. It retained the original
measurement values. New results using the archive should identify its source
and cite its paper and artifact.
