# Scalar materialization adapter

`benchmarks/tools/bench_metal_suites.py` stages a separate Metal worker. Its
scalar adapter recognizes a lexically typed DataFrame returned by a terminal
LazyFrame `collect()`, followed by `frame.select(scalar_expr).item()`. It inserts
only `.lazy()` after the receiver and `.collect()` after the unchanged SELECT.
The existing staging step forces that collection with `MetalRuntime()`.
Canonical queries, CPU workers, table schemas, and expressions are unchanged.

This is a conservative benchmark source adapter, not a general Mojo type
checker. It supports a single reviewed aggregate Expr with optional casts or
aliases. Unknown receivers, raw DataFrames, row selections, selectors/options,
borrowed aliases, nested closure captures, and invalidated assignments remain
unchanged and appear in the proof's `unadapted` list. Function annotations and
lexical scope are used to establish receiver types. No query ID, receiver name,
or data-dependent condition selects an adaptation.

Staging retains `scalar_materialization_proofs` for each canonical module. Each
proof pins the canonical and derived worker SHA256 and hashes the unchanged
SELECT/argument bytes. `adapted` records identify the source line, receiver type,
and two insertions. An audit with a mismatched canonical or pinned derived
worker hash becomes `unknown` and loses diagnostic alignment/order keys. Legacy
hybrid metadata remains hybrid; staging never promotes it automatically.

Run the derivation checks without a Mojo compiler:

```sh
python3 tests/tooling/test_metal_materialization_adapter.py
python3 tests/tooling/test_metal_suite_driver.py
```

For physical validation, stage the new workers and run both TPC-DS q24 variants
against the same inputs as the previous harness. Reuse the supplementary q24
literal oracle from the separate populated-witness increment: each variant must
return the unchanged one-row typed answer. The canonical CPU worker must stay
byte-identical to its previous derived worker. Run the final harness against
both the initial and current engine trees before attributing a coverage change
to kernels. Keep supplementary witness receipts separate from canonical counts.

The source audit finds one adaptation in the current canonical modules. This is
not a runtime answer, performance result, or whole-query GPU certificate. Host
scalar readback/control, IO, dictionary preparation, and orchestration remain
outside that claim. The existing 49-cell empty-answer backlog is unchanged.
