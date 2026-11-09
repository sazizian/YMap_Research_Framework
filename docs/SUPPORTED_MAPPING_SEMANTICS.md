# Supported object/relational mapping semantics

This guide documents the mapping constraints and validation checks implemented in this release. The [profiling guide](PROFILING_PROTOCOL.md) describes how this implementation measures PostgreSQL performance.

## Input contract

Models use the supplied JSON format: `name`, `classes`, and `associations`. A class has a unique string `name`, optional single string `extends` parent, optional boolean `abstract`, and `attributes`. Attributes have unique names, a supported scalar `type`, and optional boolean `nullable` (default false). Supported types are integer/int, string/text, float/double/decimal, date/datetime, and bool/boolean. Cyclic inheritance, multiple inheritance, attribute overriding/shadowing, unknown types, abstract classes without a concrete realization, and ambiguous physical attribute-name encodings are rejected. Unsupported semantic fields such as custom uniqueness/cardinality/index flags are rejected rather than silently ignored. Nonsemantic annotations may be placed under `metadata`.

Associations have a unique `name`, `source`, `target`, and `multiplicity`: `one-to-one`, `one-to-many` (default), or `many-to-many`. For 1:1 and 1:N, `nullable: false` (default) means every target object must have exactly one source; true permits zero or one source per target. 1:1 additionally permits at most one target per source. Source participation may be zero. M:N means zero-to-many at both endpoints; omit `nullable`, which has no unambiguous meaning for both ends and is rejected when present. General minimum/maximum cardinalities, ordered collections, association attributes, ownership/cascade semantics, and collections of scalar values are outside this implementation.

## Inheritance

Mapping strategy is selected per complete hierarchy, not independently per subclass. Arbitrary inheritance depth is supported.

| Strategy | Physical representation | Explicit limits |
|---|---|---|
| `single_table` | One table per hierarchy, discriminator, all declared attributes. Conditional checks require nonnullable attributes only for applicable concrete types and prohibit irrelevant values. | No runtime ORM framework or lazy loading behavior is implemented. |
| `joined` | One table per class; each child has a composite `(id,dtype)` foreign key to its direct parent. Own attributes remain in the owning class table. | SQL constraints do not prove that every root discriminator row has all required child rows; the supported loader supplies and checks the complete ancestor path. |
| `table_per_class` | One table per concrete class, containing all inherited and own attributes; polymorphic reads use unions. Abstract classes have no own table. | A hierarchy choice is explicitly excluded if an association endpoint within it has descendants. Cross-table polymorphic referential constraints are not implemented. No identity registry is added. |

Object IDs are globally unique within the logical fixture and preserved across all mappings. Cross-table global identity for TPC is a loader invariant; a database-global uniqueness constraint across its separate concrete tables is not claimed. IDs are generated client-side, not measured as sequence allocation.

For STI/joined, typed association references use `(id,dtype)` composite foreign keys plus checks limiting the allowed concrete discriminator values of the declared endpoint. A source of static type `Student`, for example, may reference `Student` and descendants, and cannot reference an unrelated sibling or direct `Person` object. Joined parent/child composite keys also prevent discriminator disagreement between stored parent and child rows.

## Associations and integrity

`direct_fk` places a typed source reference on the target class's physical table (or the hierarchy table for STI). Conditional checks scope it to the declared target type. A 1:1 reference has a unique source-ID constraint. M:N never uses `direct_fk`.

`association_table` creates a pair-keyed link table with typed foreign keys to both endpoints. A 1:N link has a unique target-ID constraint; a 1:1 link has unique target-ID and source-ID constraints. Required target participation for nonnullable 1:N/1:1 is enforced by deferred PostgreSQL constraint triggers on both the target storage table and the link table. These triggers scan for missing participation at transaction end, and their cost is included in insert/commit timing. They are a correct reference implementation, not an optimized bulk-loading strategy. Foreign keys are `MATCH FULL DEFERRABLE INITIALLY DEFERRED` so complete cyclic logical fixtures can be loaded in one transaction. Delete cascade policy is not assumed.

`validate_model` checks the supported source syntax; `validate_candidate` checks primary keys, foreign-key target existence/uniqueness, column types, generated name collisions, and graph endpoint integrity. Before profiling, `canonical_candidate` independently regenerates supported SQL from the source model/decisions and rejects modified or stale executable candidate metadata. The profiler does not execute arbitrary DDL imported in JSON. Neither function proves complete object/relational semantic equivalence. Logical-fixture validation checks global object IDs, inherited attributes, subtype membership, association uniqueness, participation, and realization hashes. An independent logical oracle computes the expected reference-query rows before comparing database results. These are finite checks for the supplied fixture, not a proof for all possible object states or application queries.

## Candidate enumeration and Alloy

The Python generator lazily enumerates the finite Cartesian product of admissible inheritance and association choices. It does not first materialize every candidate. `max_candidates` selects a deterministic prefix; this is neither random sampling nor evidence that the best candidates are in the prefix. Candidate IDs include the entire source model, mapping decisions, and mapping-format version. Each candidate records excluded unsupported choices.

`backend="alloy"` invokes the real Alloy Java API using a user-supplied local distribution JAR and JDK. It writes the `.als` model, exact decision encoding/scope, JAR/bridge SHA-256, Java version, compiler diagnostics, solver identity, and solver decisions to the requested artifact directory. The bounded encoding solves finite admissible strategy choices. Python constructs and structurally checks each corresponding DDL. It does **not** encode the whole SQL translation, prove query preservation, or justify a claim of complete semantic preservation. It never automatically falls back to Python if Java/JAR execution fails. The bridge was compiled and executed against the official Alloy 6.2 distribution with JDK 17 during verification: its solution-ID sets/counts matched Python for zero decision dimensions, a three-strategy hierarchy, and a polymorphic-association example. Other installed Alloy versions still require local verification.

The implementation retains distinct decision variants; it does not claim SQL-semantic equivalence deduplication, exhaustive schema design beyond this supported strategy space, or scalability to an unbounded space. The graph representation includes table and column nodes, containment edges, and individual foreign-key-column edges. Composite-FK grouping remains in the candidate's `tables[*].foreign_keys` metadata.

## Execution verification and remaining limits

PostgreSQL syntax and trigger bodies were statically parsed with pglast 8.7 for 84 candidates across all 11 supplied example models. Actual bounded Alloy solution comparisons used Alloy 6.2 and JDK 17. PostgreSQL execution verification used the official PGlite 0.5.10 WebAssembly build of PostgreSQL 18.6: ten positive physical-mapping cases matched the independent logical oracle, and six direct-engine negative integrity probes failed with the expected SQLSTATEs. These are software verification checks, not reproduction of the paper's experiments or validation of native PostgreSQL timing performance.

A native PostgreSQL instance could not be started in this root-only sandbox: PostgreSQL forbids running its server as root, and the container does not permit switching to another UID. The PGlite socket wrapper also failed psycopg error-recovery framing after intentionally invalid transactions. Positive psycopg roundtrips and direct-engine constraint failures passed separately; native-driver negative transaction recovery remains to be verified on a normal PostgreSQL installation. The native integration test remains intact and no SQL constraint was weakened for the WASM wrapper.
