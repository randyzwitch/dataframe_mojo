Measurements in this note were taken at commit 51c40c90 (2026-10-20); they are a record of that decision, not current numbers.

# Primitive GPU transfer probe

Run on an Apple Silicon Mac with the repository's GPU environment:

```sh
pixi run -e gpu mojo -I . benchmarks/bench_gpu_transfers.mojo
```

The default covers 4,097 and 1,048,576 rows with five repetitions per layout.
Pass `rows [repetitions]` to select one size, for example
`pixi run -e gpu mojo -I . benchmarks/bench_gpu_transfers.mojo 1048576 7`.
The accepted row limit is `Int32.MAX // 42`, checked before payload sizing
and allocation.

Each whitespace-delimited `SAMPLE` line is machine-parseable as alternating
field names and values. `sample_kind=first` marks repetition zero for that
layout; later repetitions are labeled `warm`. One DeviceContext is initialized
before sampling and reused for every case. The measured phases are host List
allocation, DeviceBuffer creation/enqueue, upload enqueue, upload
synchronization wait, download enqueue, and download synchronization wait.
Input generation and full round-trip verification are outside the timed
phases.

The layouts are generic 21-byte-per-cell planes (separate low and high UInt64
words, one UInt8 validity value, and one UInt32 error value), raw four-byte
Float32, and Float32 plus bit-packed nullable validity. The generic layout uses
distinct nonzero high-word and error-word sentinels to check all four transfer
planes; these are transport patterns, not valid Float32 or fault semantics.
The raw Float32 layout has no validity or error semantics. The generic and
nullable layouts use the same every-thirteenth-row validity pattern. Transfer
payload sizes are reported for one direction. Host byte counts include input
and readback Lists; device byte counts report only the input buffers.

This isolates primitive host/device copies and buffer creation. It runs no
kernels, does not construct DataFrame columns, and does not measure the full
staging or result-materialization path. The output is diagnostic evidence about
these direct-copy layouts; it is not a GPU bandwidth result or a claim about
end-to-end performance.
