# Head-to-head after the public CSV reader switch

Polars 1.44.2, Linux x86-64, 32 threads, warm input, five timed repetitions
per process. Run `pixi run -e oracle bench-polars`; the runner verifies output
heights and checksums before reporting times. The table below is one complete
run on the `perf/chunked-slice-batches` branch after the fix. Times are ms;
Mojo/Polars ratios below 1 mean Mojo is faster. This is a snapshot, not a
performance guarantee: CSV and Polars times varied across nearby runs.

| Workload | Rows | Mojo | Polars | Mojo / Polars |
|---|---:|---:|---:|---:|
| csv_read | 100,000 | 6.69 | 2.43 | 2.8x |
| arithmetic_chain | 100,000 | 2.36 | 3.34 | 0.7x |
| nullable_compare | 100,000 | 2.22 | 3.27 | 0.7x |
| filter | 100,000 | 7.27 | 4.13 | 1.8x |
| global_sum | 100,000 | 0.24 | 0.14 | 1.7x |
| grouped_low | 100,000 | 4.50 | 1.80 | 2.5x |
| grouped_high | 100,000 | 5.24 | 2.87 | 1.8x |
| grouped_skew | 100,000 | 4.53 | 2.27 | 2.0x |
| grouped_str | 100,000 | 8.25 | 2.03 | 4.1x |
| join_inner | 100,000 | 14.57 | 5.10 | 2.9x |
| sort_multi | 100,000 | 21.60 | 7.04 | 3.1x |
| csv_read | 1,000,000 | 19.50 | 17.74 | 1.1x |
| arithmetic_chain | 1,000,000 | 8.25 | 4.13 | 2.0x |
| nullable_compare | 1,000,000 | 5.81 | 4.17 | 1.4x |
| filter | 1,000,000 | 52.20 | 7.45 | 7.0x |
| global_sum | 1,000,000 | 0.57 | 0.20 | 2.9x |
| grouped_low | 1,000,000 | 24.77 | 13.56 | 1.8x |
| grouped_high | 1,000,000 | 33.11 | 13.13 | 2.5x |
| grouped_skew | 1,000,000 | 24.29 | 15.51 | 1.6x |
| grouped_str | 1,000,000 | 114.85 | 14.94 | 7.7x |
| join_inner | 1,000,000 | 118.65 | 27.08 | 4.4x |
| sort_multi | 1,000,000 | 135.91 | 55.53 | 2.4x |

The original full-suite run on `main` showed 1M arithmetic, comparison and
filter times near 3.2 seconds each. That was a real regression introduced when
#161 made CSV results chunked, not a table-format error. `fusion.fused` was
rechunking all eight input columns for every 1,024-row expression batch.
`Series.slice` also made a List of every chunk for each batch; that separate
cost affected grouping. This branch finds overlapping chunks by cumulative
offset and prepares only referenced Float64 fusion inputs once per expression.

Two alternating comparisons used separate optimized binaries from unchanged
`main` (`5e05b84`) and this branch, the same 1M-row fixture, 32 threads,
and best of three warmed reads per process:

| Workload | Main range | This branch range |
|---|---:|---:|
| arithmetic_chain | 3,182–3,184 ms | 8.08–8.15 ms |
| nullable_compare | 3,162–3,186 ms | 5.84–5.91 ms |
| filter | 3,213–3,227 ms | 50.75–60.67 ms |
| grouped_low | 59.70–61.65 ms | 25.63–26.06 ms |

A separate ablation rechunked the input once before all non-CSV workloads and
measured 4.31 ms arithmetic, 2.14 ms comparison, and 22.52 ms filter at 1M
rows. This isolates further costs of chunked input; it is not the proposed
public behavior. The remaining 1M filter and string-grouping gaps in the full
table are real and need separate investigation. The older pre-#161 table in
[cpu-performance-evaluation.md](cpu-performance-evaluation.md) used contiguous
CSV output, so its roughly 5x maximum at 1M rows does not describe this
post-port checkout.
