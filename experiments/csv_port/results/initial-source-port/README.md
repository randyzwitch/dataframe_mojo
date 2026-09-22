# Initial CSV source-port timings

One million mixed rows, 46,212,703 bytes, explicit schema
`id:Int64,value:Float64,active:Bool,label:String`. Includes numeric nulls,
quoted separators, escaped quotes, Unicode, and embedded newlines. Projection
reads `id,label`. Same machine, file, schema, and requested thread counts;
separate sequential processes, warm file cache, seven retained samples each.
No compiler/test jobs ran concurrently. These are medians in milliseconds.

| Read | Threads | Main | Source port | Polars | Main/port | Port/Polars |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 | 394.29 | 381.62 | 138.00 | 1.03× | 2.77× |
| full | 4 | 166.62 | 104.98 | 36.26 | 1.59× | 2.90× |
| full | 32 | 35.37 | 20.73 | 11.81 | 1.71× | 1.76× |
| projected | 1 | 345.32 | 290.58 | 94.78 | 1.19× | 3.07× |
| projected | 4 | 124.58 | 79.47 | 24.65 | 1.57× | 3.22× |
| projected | 32 | 30.88 | 16.41 | 10.03 | 1.88× | 1.64× |

This is one initial mixed-string workload, not a general parity claim or an
attribution of the gains to individual commits. The remaining single-thread
gap demonstrates that pool startup cannot explain the whole difference.
The scoped shared-queue pool, temporal conversion, and native Arrow view export
remain documented source differences. The public reader is not switched.

Every run checked shape, null counts, and full output equality against its
untimed reference. Separately, `verify_reader_oracle.py` compared actual Mojo
outputs with Polars for 6,000 rows across explicit/inferred schemas,
full/projected reads, and one/four threads (eight passing comparisons).

`metadata.json` records source commits, input hash, machine, and build commands.
`baseline_driver.mojo` is the benchmark driver compiled against an unchanged
archive of merged main; it imports no port modules. `main_*.csv` uses the
historical `legacy` engine label; `port_*.csv` uses `explicit`.

`before-harness-lifetime-fix/` preserves superseded Polars samples. The Python
harness originally retained a warmup result and could destroy the preceding
output inside the next assignment. Corrected runs explicitly release both
outside the timed read. Only top-level CSV files contribute to this report.

Reproduce the input with `prepare.py ... --rows 1000000`. Use the build commands
in metadata, then the commands in the parent README with 7 iterations and
thread counts 1, 4, and 32. Summarize only this directory's top-level CSV files.
