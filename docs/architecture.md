# Architecture and development direction

## Current layers

- `column.mojo`: typed windows (offset, length) onto reference-counted value
  buffers and LSB-first validity bitmaps, as in Arrow arrays.
- `string_column.mojo`: string columns in the Arrow `large_utf8` layout (one
  UTF-8 byte buffer, Int64 offsets, validity), read as borrowed `StringSlice`s
  and built with `StringBuilder`.
- `bool_column.mojo`: Boolean columns with bit-packed values (one bit per
  row), sharing the bitmap helpers with validity.
- `io/`: CSV, Parquet, GeoJSON, and Shapefile readers and writers, and
  (`io/arrow.mojo`) Arrow C Data Interface export (zero-copy where layouts
  agree) and import.
- `series.mojo`: named, runtime-tagged columns and batch slicing/concatenation.
- `expr.mojo`: flat expression nodes and composition, independent of data.
- `binding.mojo`: schema resolution, dtypes, shape, and aggregate dependencies.
- `execution.mojo`: bounded eager evaluation and grouped accumulator scheduling.
- `expr_kernels.mojo`: specialized binary kernels, including explicit Float64 SIMD.
- `aggregation/scalar_states.mojo`: mergeable exact-integer and reassociable floating-point states.
- `frame.mojo`: validated eager dataframe operations and expression entry points.
- `aggregation/`: eager grouping implementations and shared streaming reduction state.

The only runtime dependency is the Mojo standard library. This is an eager CPU
implementation, not a production query engine. Buffers are immutable once
shared, so copying a column, `select`, `rename`, `drop`, `head`, `slice`,
`column()`, typed extraction, GroupBy snapshots, and expression batch slices are
O(1) per column and share storage. The only in-place growth (batch reassembly)
copies first unless the column owns its buffers outright. Column lookup at binding
is O(schema width), but execution uses resolved source indices.

Sorting is O(n log n) with O(n) index workspace. Hash grouping is expected O(n).
Join index construction is expected O(left + right + output); materialization
also scales with column count. Variable-length string costs depend on their length.

See [expression semantics](expressions.md) for the broadcasting, ordering, and
reduction contracts that deliberately leave room for parallelism and SIMD.
No universal dataframe trait or foreign-backend abstraction is imposed.

## Current state

The milestones this page used to list are in place: lazy plans with
predicate, projection, and slice pushdown ([lazy queries](lazy.md)); parallel
hash grouping, joins, and sorts on worker threads, with `maintain_order` as an
explicit request; chunked columns and Arrow C Data import and export
([Arrow interchange](arrow.md)); CSV and Parquet input ([CSV](csv.md),
[Parquet](parquet.md)); and multi-column sorts with multiple numeric or string
grouping and join keys. Performance work is measured query by query against
Polars and DuckDB on external suites; see [benchmarks](design/benchmarks.md).

Parquet reading goes through Arrow C++ behind a C shim (`native/dfparquet`)
until a Mojo-native reader exists; `dataframe/io/parquet.mojo` is the seam.

Optional GPU execution on Apple silicon is covered in
[Apple Metal](apple-metal.md). Distributed execution, arbitrary Python-object
columns, pandas index compatibility, and cross-engine adapters are outside the
scope.

## Lazy query layers

`dataframe.lazy` remains the import surface for `LazyFrame`, its grouped
builders, scan functions, and existing plan-inspection symbols. It re-exports
the implementations below. `dataframe.__init__` continues to expose the same
public query API.

The plan is a flat, topologically ordered node list: children precede their
parents, and a scan's offset selects matching slots in the context's source
frames and schemas. Graph rewrites restore that ordering through `plan_reorder`.

- `lazy_api.mojo` owns query construction, public collection/profiling/explain
  entry points, and accelerator provider dispatch. A `LazyFrame` owns one
  internal `QueryContext`; private plan storage is accessed through `_plan`.
- `planning/model.mojo` defines `PlanNode` and operation tags. It imports
  neither optimizer passes nor execution code.
- `planning/context.mojo` holds nodes, source frames, schemas, and per-run
  caches/counters. Execution and provider ports are supplied by the public
  API when a context is constructed and retained when it is copied.
- `planning/graph.mojo`, `properties.mojo`, and `scans.mojo` implement graph
  traversal, identity, schema queries, expression properties, and scan-node
  encoding. `strategies.mojo` holds shared sampling and memory-budget rules;
  `explain.mojo` describes nodes and their streaming capabilities.
- `optimizer/expressions.mojo` and `optimizer/passes.mojo` own expression
  folding and logical rewrites, including predicate, projection, and slice
  pushdown, common subexpressions, and shared-subplan filter handling.
- `planning/physical.mojo` chooses CPU strategies and join order, including
  decisions informed by sampled or measured inputs.
- `query_execution/` interprets the plan, lowers and runs streaming pipelines,
  materializes shared inputs, records observed execution, and offers regions
  for accelerator placement. This namespace is distinct from
  `execution.mojo`, which evaluates bounded expressions over columns.

Dependencies within these layers point from the public API to execution and
physical planning, then to logical optimization and shared plan analysis, and
finally to the node model. Lower layers never import `lazy`, `lazy_api`, or
`query_execution`. Some planning operations require a schema or measured input:
they request it through the context's execution port instead of importing the
executor. Independent-input jobs use that same port. Shared-subplan execution
requests collection through a separate port, preserving the public collection
settings and execution-report behavior. Provider adapters stay in `lazy_api`;
the executor receives only provider-neutral context ports.

`scripts/check_lazy_boundaries.py` checks these import directions and rejects
cycles among the lazy implementation modules. The existing eager
`DataFrame`/`LazyFrame` type relationship remains at the public integration
boundary; this check does not impose a new dependency model on eager operators.

The public class retains private method delegates used by planner tests and
provider integrations. Algorithm bodies live in their owning modules; delegates
do not implement a second planner or executor. Plan copying still shares the
existing reference-counted column storage and observed counters, while worker
jobs retain their own prefetch dictionaries.

Benchmark runner freshness, trace discovery, and query-diagram source
fingerprints traverse the implementation recursively so moving code into
subpackages preserves those checks.

## Aggregation implementation

`aggregation/group_by.mojo` owns `GroupBy` and `GroupingSets`, grouping strategy
selection, and the range, perfect-hash and bucket worker jobs. The kernels below live alongside the eager and streaming implementation.

`aggregation/state.mojo` owns `_ReductionPlan`, `_StreamReduction`, merge and
finish jobs, and aggregate-table input/output adapters. Both eager grouping and
streaming execution import this state directly. `aggregation/keys.mojo` handles
struct expansion/repacking and categorical key normalization;
`aggregation/expressions.mojo` provides binding and row-local/reduction checks
also used by ordinary frame operations and query planning.

The extracted code still uses the existing `DataFrame` value type and `concat`
from `frame.mojo`; `GroupingSets` also retains its `LazyGroupingSets` plan type.
These dependencies preserve the existing representations and operations.
`frame.mojo` re-exports the moved symbols, and `streaming.mojo` re-exports its
three existing state helpers for import compatibility. Internal consumers use
the implementation modules directly. `scripts/check_aggregation_boundaries.py`
checks that aggregation does not import frame implementation helpers, streaming
or execution facades, or accelerator dispatch, and has no internal module cycles.

## Eager join implementation

`DataFrame.join` retains its public overloads, key-argument checks, and cross-join
name/overflow validation. `_join_impl` delegates to `joins/eager.mojo`, which
owns the existing strategy order and output assembly. Key dtype and struct
preparation remain in that implementation with their original error ordering.

`joins/range.mojo` owns bounded-integer/progression matching, membership and
range filtering, including their existing byte budgets and worker thresholds.
`joins/rows.mojo` owns stable pair ordering, CSR layouts and row count/fill jobs;
`joins/keys.mojo` encodes both sides into one key-id space for dictionary joins.
The existing hash/fused join kernels and as-of joins remain in their modules.

The join modules use the existing `DataFrame` value type, while frame re-exports
all moved helpers and constants for compatibility. Shared categorical dictionary
normalization for concatenation and chunk-filter jobs stay on the frame.
`scripts/check_join_boundaries.py` enforces the internal dependency direction
and prevents imports of frame implementation helpers or higher execution layers.

## Aggregation kernels

The computational aggregation modules share `aggregation/` with the eager and
streaming orchestration. Their names distinguish the state and table layouts:

| Module | Responsibility | Compatibility import |
| --- | --- | --- |
| `scalar_states.mojo` | Mergeable integer, floating, logical and moment states | `dataframe.reductions` |
| `reducer.mojo` | Typed `Reducer` update/merge/finalize and reduction kernels | `dataframe.aggregate` |
| `hash_reduce.mojo` | Per-worker high-cardinality Int64 grouped reductions | `dataframe.hash_agg` |
| `dense.mojo` | Perfect-hash layouts and dense group slots | `dataframe.perfect_aggregate` |
| `hash_table.mojo` | Radix-partitioned aggregate tables and inline group states | `dataframe.aggregate_table` |
| `selected_rows.mojo` | Reduction kernels reading inputs through selected row indices | `dataframe.indexed_reduce` |

The six compatibility modules re-export their original declarations, including
private helpers and constants. Engine consumers import the canonical modules;
contract tests retain legacy imports. The aggregation boundary gate rejects new
engine dependencies on those facades. Existing dependencies on expression
execution and pipeline buffer facts remain; this move preserves algorithms,
thresholds, state layouts, allocation, ownership and worker behavior.
