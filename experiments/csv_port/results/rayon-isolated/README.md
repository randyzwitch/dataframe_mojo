# Isolated native Rayon experiment — not adopted

A temporary reader outside the repository linked Rayon 1.12.0 /
rayon-core 1.13.0, using a global OnceLock ThreadPool, ThreadPool.scope, and
Scope.spawn. It preserved CountLines chunking and the builder-alignment
parser (before typed integer/trusted-span/float frontend changes).
The native dependency and alternate reader are not part of production code.

Same fixture and seven-sample methodology as ../initial-source-port.
Explicit 1/32 thread settings matched between engines. Separate processes,
warm cache, no concurrent builds/tests. Medians in milliseconds:

| Read | Threads | Scoped pool | Native Rayon | Polars |
|---|---:|---:|---:|---:|
| Full | 1 | 273.95 | 291.60 | 138.11 |
| Projected | 1 | 173.61 | 187.80 | 93.74 |
| Full | 32 | 17.22 | 28.34 | 11.34 |
| Projected | 32 | 22.43 | 18.26 | 10.76 |

Rayon improves the projected 32-thread case but is worse in the other cases.
This does not establish that scheduling accounts for the remaining gap or
justify adding a native dependency. The bridge adds FFI calls per spawn/task;
this is not a controlled measurement of Polars' entire runtime.

The bridge passed 20 AOT and 20 JIT process exits. The isolated CSV reader
passed 24 actual Polars comparisons and seven reader tests including drained
worker errors. Temporary reproduction sources are /tmp/rayon_bridge and
/tmp/dataframe-mojo-rayon-probe; raw measurements are retained here, but the
prototype is not a supported or repository-reproducible reader.
