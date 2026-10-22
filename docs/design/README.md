# Design notes and measurement records

These notes explain why parts of the engine look the way they do. They are
for people changing the library, not for people using it. The pages for users
are in [docs/](../../README.md#using-the-package).

Each note describes the tree at the commit in the last column, the last
substantive edit on `main` when this index was written. Code may have moved
on since. Notes marked "measurement" also open with the commit their numbers
were taken at; those are a record of a decision, not current numbers.

| Note | Purpose | Describes the tree at |
| --- | --- | --- |
| [accelerator-backends.md](accelerator-backends.md) | How optional accelerator backends plug into the planner and executor. | `c411a93e` (2026-10-16) |
| [apple-metal.md](apple-metal.md) | The Apple Silicon Metal backend: runtime, kernels, supported operations. | `8d2c0566` (2026-10-22) |
| [apple-metal-benchmarks.md](apple-metal-benchmarks.md) | Measurement: Metal development mechanisms compared with the CPU paths. | `51c40c90` (2026-10-20) |
| [apple-metal-replacement-coverage.md](apple-metal-replacement-coverage.md) | Matrix of which engine operations have a Metal replacement. | `7603528e` (2026-10-20) |
| [arrow-import.md](arrow-import.md) | Measurement: parallel Arrow import (#276). | `be283d9d` (2026-10-15) |
| [benchmarks.md](benchmarks.md) | Rules for benchmarking, the external suites, and retired benchmarks. | `d05602cf` (2026-10-21) |
| [executor-design.md](executor-design.md) | Design of thread-owned pipelines over morsels (#538). | `c54d9d2c` (2026-10-10) |
| [gpu-transfer-probe.md](gpu-transfer-probe.md) | Measurement: primitive GPU transfer costs. | `51c40c90` (2026-10-20) |
| [join-cutoffs.md](join-cutoffs.md) | Measurement: join and gather cutoffs (#268, #272). | `54495318` (2026-10-15) |
| [lazy-streaming.md](lazy-streaming.md) | Measurement: lazy streaming time and resident memory (#109). | `54495318` (2026-10-15) |
| [parquet-streaming.md](parquet-streaming.md) | Measurement: Parquet row-group streaming (#277). | `be283d9d` (2026-10-15) |
| [sort-cutoffs.md](sort-cutoffs.md) | Measurement: low-cardinality sort calibration (#271). | `54495318` (2026-10-15) |
| [test-coverage-plan.md](test-coverage-plan.md) | How tests are chosen: contracts, independent expected results, dimensions. | `66fd96b0` (2026-10-16) |
| [worker-calibration.md](worker-calibration.md) | Measurement: worker-count and machine calibration (#273). | `54495318` (2026-10-15) |
