# PostgreSQL profiling protocol

Profiling requires a PostgreSQL server and optional `psycopg[binary]>=3.1`. Set a connection string in the environment variable named by `dsn_env` (default `YMAP_POSTGRES_DSN`). Credentials are never copied to manifests or printed by this code. A connection failure is fatal; no synthetic timing fallback exists.

Set `fixture_mode: deterministic_generated` to generate the input objects and associations for a profiling run. Each run has its own fixture, environment, workload, and measurement records. See the [manuscript workflow](PAPER_WORKFLOW.md) for configuring the reference protocol.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `dsn_env` | `YMAP_POSTGRES_DSN` | Environment-variable name, not a literal DSN |
| `fixture_mode` | Required | Must be `deterministic_generated` |
| `rows_per_class` | 100 | Positive exact-class object count; alternatively a complete concrete-class-to-count dictionary |
| `seed` | 42 | Fixture and association reproducibility seed |
| `repetitions` | 5 | Measured repetitions, summarized by median |
| `warmup` | 1 | Complete unreported warmup repetitions, retained in raw evidence |
| `batch_size` | 1000 | Client `executemany` batch size |
| `query_limit` | 100 | Maximum fetched rows for scan/join templates |
| `statement_timeout_ms` | 30000 | Session statement timeout |
| `environment_notes` | Uncharacterized server hardware | Supply server CPU/RAM/storage and isolation information manually |

Every concrete class gets its configured count; abstract classes have no own instances. Values follow the source scalar types and deterministic reference distributions; nullable attributes have a 0.1 null probability. Logical IDs are client-assigned and identical across mappings. 1:N assigns one source to every mandatory target. Optional target participation skips every fifth target. 1:1 uses distinct sources and rejects infeasible required cardinalities, rather than silently changing the fixture. M:N links each target to up to two distinct sources. The fixture, source model, DDL, and mapped query text/parameters are preserved per candidate.

## Shared workload and mapping realization

The logical template contains one point lookup and one ordered limited scan per logical class, with polymorphic projections of that class's inherited and declared attributes. It contains one ordered limited source/target join per association. SQL differs by mapping; logical inputs and expected rows do not. Result rows are completely fetched before stopping each query timer. An independent logical oracle computes expected result hashes directly from the source objects/links; database results must match those hashes on every repetition and mapping. The first physical mapping is not treated as ground truth. A mismatch rejects the profile and prevents a final `measurements.csv` from being certified.

`workload_id` hashes the shared logical template/data-distribution protocol, not the model name/content or candidate. Thus the same protocol applied to different systems has the same scenario ID. `realization_id` separately hashes the specific source model, objects, and links. The seed is explicit and does not become part of `workload_id`, so repeated seeds remain in the same scenario. Query count differs with source-system size and is recorded. The template equally weights its logical queries; it is not an observed production frequency mix and does not establish robustness under alternative workloads or skew.

## Measured boundaries and units

1. Each candidate gets a new unpredictable UUID-named PostgreSQL schema. It is created by this run; an existing schema name is never accepted. All candidate objects are created there. Only that schema is later dropped, using `DROP SCHEMA <owned-uuid> CASCADE`. A per-candidate lifecycle file records the owned name, successful creation, and cleanup status, including connection-loss failures requiring manual cleanup of only that schema. Imported candidate SQL/mapping metadata must match canonical regeneration before any SQL executes. Use an account with schema creation permission in a dedicated research database.
2. DDL, source-fixture generation, candidate enumeration, and query construction occur outside metric timers. These setup costs must be separately measured if reporting an end-to-end recommendation cost.
3. Each warmup/measured repetition truncates the candidate tables outside the insert timer. `insert_ms` measures wall time of inserting the **whole logical fixture in one transaction**, including client conversion/batching, network transfer, indexes, foreign keys, deferred participation triggers, and COMMIT. It is not average latency of one insert operation and is not directly interchangeable with a paper metric defined per operation. Exact logical object counts, physical row counts, and batch settings should accompany any interpretation.
4. Tables are analyzed outside both timers. Query timing includes client execution, database execution, network, and full result fetching. `query_ms` is the arithmetic mean per logical SQL query for that repetition, then the median across measured repetitions. Each individual query's wall time is retained. No application ORM hydration, session management, lazy/eager loading, cache behavior, or transactional production concurrency is measured.
5. `storage_mb` is the sum of `pg_total_relation_size` for all candidate ordinary tables, including their indexes and TOAST, divided by **1,000,000**. It is decimal MB of relation storage, not process memory or heap RAM. Raw `storage_bytes` are retained. Schema catalog/function metadata and WAL volume are outside this measure.
6. Prepared statements and automatic preparation are disabled. Insert batching is fixed by configuration. Query order is deterministic. Server settings and client details are recorded. Query cache conditions are warm/mixed after insert and ANALYZE; no operating-system or PostgreSQL cache eviction is attempted. Repeating schema loads does not establish a cold-cache experiment. Concurrent server load, hardware, and candidate-order effects require independent experimental control. Candidate order can be randomized externally before profiling and should then be recorded by the calling experiment.

`measurements.csv` has one median summary row per candidate and seed, with `label_source=postgresql_measured`. `raw_measurements.jsonl` retains all repetitions including warmups, query timings/result hashes, storage bytes, workload/environment/realization IDs, and seeds. `environment.json` records PostgreSQL version/settings and client details; its stable hash excludes candidate names, order, timestamps, and measured timings. `workload.json` records the protocol and measurement definitions. A failed run retains `.partial`/raw evidence but never turns it into a complete measurement file. Output evidence is never overwritten automatically.

## Verification status

Pure model validation, hierarchy and association construction, deterministic fixtures, physical load plans, independent logical result oracles, and logical SQL plans have automated checks. Optional `pglast` 8.7 static validation parsed DDL, PL/pgSQL trigger bodies, SELECTs, and INSERT templates for 84 candidate examples across all 11 supplied models. The helper `validate_postgresql_syntax` reports `database_execution: false`; parsing alone cannot verify catalog resolution, constraint enforcement, or driver behavior. See the [official pglast API documentation](https://pglast.readthedocs.io/en/latest/api.html) for the separate SQL and PL/pgSQL parsers.

Live integration tests are enabled with `YMAP_TEST_POSTGRES_DSN`. They execute multilevel polymorphic mappings, all three inheritance strategies with supported leaf associations, exact decimal/date/timestamp roundtrips, schema cleanup, and negative subtype/one-to-one/participation constraints. Native server validation must be completed in the target research environment before reporting findings. The optional Alloy 6.2/JDK 17 bridge has separately passed actual solver solution-set comparisons; that does not validate SQL execution or performance claims.

Additional build verification used the official PGlite 0.5.10 PostgreSQL 18.6 WebAssembly engine. Positive psycopg socket roundtrips passed four multilevel/polymorphic mappings and six mappings spanning all three supported inheritance strategies with leaf associations and scalar types. Its direct JavaScript API rejected six invalid-data probes with the expected PostgreSQL SQLSTATEs, including subtype violations, one-to-one duplicates, partially null composite foreign keys, and missing mandatory target links. The PGlite socket wrapper did not correctly recover expected-error transactions under psycopg in this environment, so negative native-driver error recovery remains unverified; the native integration tests are retained without weakening constraints. These checks are execution verification against a WASM PostgreSQL engine, not native-server performance experiments or reproduction of the original manuscript's results. Runtime class/version and psycopg version are recorded in profiling environment manifests.

The software validation environment for the PGlite checks provided root access only, which prevented starting a native PostgreSQL server there. The native integration suite uses a separate PostgreSQL environment configured through `YMAP_TEST_POSTGRES_DSN`.
