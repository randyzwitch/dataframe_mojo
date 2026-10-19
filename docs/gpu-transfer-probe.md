# Primitive GPU transfer probe

Run on an Apple Silicon Mac with the repository's GPU environment:

```sh
pixi run -e gpu mojo -I . benchmarks/bench_gpu_transfers.mojo
```

The default covers 4,097 and 1,048,576 rows with five repetitions per layout.
Pass `rows [repetitions]` to select one size, for example
`pixi run -e gpu mojo -I . benchmarks/bench_gpu_transfers.mojo 1048576 7`.

Each whitespace-delimited `SAMPLE` line is machine-parseable as alternating
field names and values. `sample_kind=first` marks repetition zero for that
layout; later repetitions are labeled `warm`. One DeviceContext is initialized
before sampling and reused for every case. The measured phases are host List
allocation and filling, DeviceBuffer creation/enqueue, upload enqueue, upload
synchronization wait, download enqueue, and download synchronization wait.
All round-trip values and validity/error bytes are checked after the timed waits.

The layouts are the current generic 21-byte-per-cell planes (two UInt64 words,
one UInt8 validity value, and one UInt32 error value), raw four-byte Float32,
and Float32 plus bit-packed nullable validity. Input values are exact binary
fractions; the generic and nullable cases use the same every-thirteenth-row
validity pattern. The transfer payload sizes are reported for one direction.
Host and device byte counts include both input and readback allocations.

This isolates primitive host/device copies and buffer creation. It runs no
kernels, does not construct DataFrame columns, and does not measure the full
staging or result-materialization path. The output is diagnostic evidence about
these direct-copy layouts; it is not a GPU bandwidth result or a claim about
end-to-end performance.
