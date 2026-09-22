# Wholesale public-reader benchmark

This is the final public `read_csv` benchmark for the CSV reader replacement.
It uses `/tmp/csv-port-mixed-million.csv`: 1,000,000 rows, four typed columns
(`id:Int64,value:Float64,active:Bool,label:String`), 46,212,703 bytes, SHA-256
`fa8eb06b7007daca362939caa992985b7792c9240cebcfe43458ae30da49e0d5`.
The full scenario reads all four columns; projected reads `id,label`.

Each entry is the median of seven read-only samples after one untimed reference
read and one warm read in its own process. The twelve processes ran sequentially
only after the validation build/test work was idle. The timed region is the
reader call. All samples retain row count, width, null count, and equality
checks against an untimed read from the same binary. Raw samples are the CSV
files in this directory.

| Read | Threads | Main b742ff2 | Public 1a0b2f0 | Polars 1.44.2 | Main/public | Public/Polars |
|---|---:|---:|---:|---:|---:|---:|
| Full | 1 | 407.89 ms | 250.29 ms | 138.25 ms | 1.63x | 1.81x |
| Projected | 1 | 343.49 ms | 147.50 ms | 93.90 ms | 2.33x | 1.57x |
| Full | 32 | 35.22 ms | 20.30 ms | 11.86 ms | 1.73x | 1.71x |
| Projected | 32 | 29.53 ms | 22.21 ms | 10.89 ms | 1.33x | 2.04x |

The `main` binary is `/tmp/csv-bench-main-b742ff2`, invoked through its former
`legacy` selector. It is an archived `b742ff2` public-reader baseline. The
replacement binary is `/tmp/bench_csv_port_public`, compiled from this source
snapshot (`1a0b2f0`) and invokes only public `read_csv`; its raw label is
`mojo-public`. Polars was the installed pinned 1.44.2 oracle with
`POLARS_MAX_THREADS` set before import.

These results establish an end-to-end improvement over the archived baseline
on this fixture. They do not attribute the gain to one CSV component or claim
broad parity: current public semantics follow Polars where the former reader
differed, and the workload has one fixed schema and projection. A preliminary
set collected while validation builds were active was excluded and moved to
`/tmp/wholesale-public-invalid-during-build`; it is not part of this report.
